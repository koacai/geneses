from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from omegaconf import DictConfig
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

import wandb
from dialogue_separator.model.components import MMDiT


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        x_merged = extras.get("x_merged", None)
        assert x_merged is not None
        x_1 = x[:, 0, :, :]
        x_2 = x[:, 1, :, :]
        res_1, res_2 = self.model.forward(x_merged, t.unsqueeze(0), x_1, x_2)
        return torch.stack([res_1, res_2], dim=1)


class SSLFeatureExtractor(nn.Module):
    def __init__(self, model_name: str, layer: int, add_adapter: bool) -> None:
        super(SSLFeatureExtractor, self).__init__()

        self.model = Wav2Vec2BertModel.from_pretrained(
            model_name, add_adapter=add_adapter
        )
        for param in self.model.feature_projection.parameters():
            param.requires_grad = False
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        self.layer = layer

    @torch.no_grad()
    def extract(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**inputs, output_hidden_states=True).hidden_states[self.layer]


class DACVAE:
    def __init__(self, ckpt_path: str) -> None:
        self.model = torch.jit.load(ckpt_path)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.decode(features)

    def to(self, device: torch.device) -> None:
        self.model = self.model.to(device)


class DialogueSeparatorLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(DialogueSeparatorLightningModule, self).__init__()
        self.cfg = cfg

        self.mmdit = MMDiT(**cfg.model.mmdit)
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.ssl_feature_extractor = SSLFeatureExtractor(
            cfg.model.ssl_model.name,
            cfg.model.ssl_model.layer,
            cfg.model.ssl_model.add_adapter,
        )

        self.dacvae = DACVAE(cfg.model.vae.ckpt_path)

        self.save_hyperparameters(cfg)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = hydra.utils.instantiate(
            self.cfg.model.optimizer, params=self.parameters()
        )
        lr_scheduler = hydra.utils.instantiate(
            self.cfg.model.lr_scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }

    def on_fit_start(self) -> None:
        self.dacvae.to(self.device)

    def on_test_start(self) -> None:
        self.dacvae.to(self.device)

    def calc_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_merged = self.ssl_feature_extractor.extract(batch["ssl_input_merged"])

        x_1 = batch["vae_feature_1"].permute(0, 2, 1)
        x_2 = batch["vae_feature_2"].permute(0, 2, 1)

        batch_size = x_merged.size(0)

        t = self.sampling_t(batch_size)
        x = torch.stack([x_1, x_2], dim=1)
        noise = torch.randn_like(x)
        path_sample = self.path.sample(x_0=noise, x_1=x, t=t)

        est_dxt_1, est_dxt_2 = self.mmdit.forward(
            x_merged, t, path_sample.x_t[:, 0, :, :], path_sample.x_t[:, 1, :, :]
        )

        loss, unweighted_loss_per_sample = self.loss_fn(
            est_dxt_1,
            est_dxt_2,
            path_sample.dx_t[:, 0, :, :],
            path_sample.dx_t[:, 1, :, :],
            t,
        )

        return loss, unweighted_loss_per_sample, t

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        _ = batch_idx

        loss, _, _ = self.calc_loss(batch)

        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        _ = batch_idx

        loss, _, _ = self.calc_loss(batch)

        self.log("validation_loss", loss, sync_dist=True)

        wav_sr = self.cfg.model.vae.sample_rate
        if batch_idx < 5 and self.global_rank == 0 and self.local_rank == 0:
            wav_len = batch["wav_len"][0]
            source_1 = batch["wav_1"][0][:wav_len].cpu().numpy()
            source_2 = batch["wav_2"][0][:wav_len].cpu().numpy()
            source_merged = batch["wav_merged"][0][:wav_len].cpu().numpy()

            self.log_audio(source_1, f"source_1/{batch_idx}", wav_sr)
            self.log_audio(source_2, f"source_2/{batch_idx}", wav_sr)
            self.log_audio(source_merged, f"source_merged/{batch_idx}", wav_sr)

            est_feature1, est_feature2 = self.forward(batch)

            with torch.no_grad():
                estimated_1 = (
                    self.dacvae.decode(est_feature1)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                estimated_2 = (
                    self.dacvae.decode(est_feature2)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

            self.log_audio(estimated_1, f"estimated_1/{batch_idx}", wav_sr)
            self.log_audio(estimated_2, f"estimated_2/{batch_idx}", wav_sr)

        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        est_feature1, est_feature2 = self.forward(batch, step_size=0.01)
        with torch.no_grad():
            decoded_1 = self.dacvae.decode(est_feature1)
            decoded_2 = self.dacvae.decode(est_feature2)

        wav_sr = self.cfg.model.vae.sample_rate
        batch_size = batch["wav_1"].size(0)
        for i in range(batch_size):
            sample_dir = Path("test_output") / f"{batch_idx}" / f"{i}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            wav_len = batch["wav_len"][i]
            source_1 = batch["wav_1"][i][:wav_len].cpu()
            source_2 = batch["wav_2"][i][:wav_len].cpu()
            source_merged = batch["wav_merged"][i][:wav_len].cpu()

            torchaudio.save(sample_dir / "source_1.wav", source_1.unsqueeze(0), wav_sr)
            torchaudio.save(sample_dir / "source_2.wav", source_2.unsqueeze(0), wav_sr)
            torchaudio.save(
                sample_dir / "source_merged.wav", source_merged.unsqueeze(0), wav_sr
            )

            estimated_1 = decoded_1[i].squeeze()[:wav_len].to(torch.float32).cpu()
            estimated_2 = decoded_2[i].squeeze()[:wav_len].to(torch.float32).cpu()

            torchaudio.save(
                sample_dir / "estimated_1.wav", estimated_1.unsqueeze(0), wav_sr
            )
            torchaudio.save(
                sample_dir / "estimated_2.wav", estimated_2.unsqueeze(0), wav_sr
            )

    def sampling_t(
        self, batch_size: int, m: float = 0.0, s: float = 1.0
    ) -> torch.Tensor:
        schema = self.cfg.model.t_sampling_schema

        if schema == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif schema == "logit_normal":
            u = torch.randn((batch_size,), device=self.device) * s + m
            t = torch.sigmoid(u)
        else:
            raise ValueError(f"Unknown t sampling schema: {schema}")

        return t

    def loss_fn(
        self,
        est_dxt1: torch.Tensor,
        est_dxt2: torch.Tensor,
        dxt_1: torch.Tensor,
        dxt_2: torch.Tensor,
        t: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mse_loss = torch.nn.MSELoss(reduction="none")

        loss_1 = mse_loss(est_dxt1, dxt_1).mean(dim=(1, 2))
        loss_2 = mse_loss(est_dxt2, dxt_2).mean(dim=(1, 2))

        unweighted_loss_per_sample = loss_1 + loss_2

        schema = self.cfg.model.weighting_schema
        if schema == "none":
            weight = 1.0
        elif schema == "sigma_sqrt":
            weight = (1 - t + epsilon) ** (-2.0)
        elif schema == "parabolic_midpoint":
            weight = 4 * t * (1 - t) + epsilon
        else:
            raise ValueError(f"Unknown weighting schema: {schema}")

        loss_weighted = unweighted_loss_per_sample * weight

        return loss_weighted.mean(), unweighted_loss_per_sample

    def forward(
        self, batch: dict[str, Any], step_size: float = 0.01
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_merged = self.ssl_feature_extractor.extract(batch["ssl_input_merged"])

        vae_size = (
            batch["wav_merged"].size(0),
            1000,  # 20秒の音声のVAEは長さ1000（ハードコーディング）
            self.cfg.model.vae.hidden_size,
        )

        noise_1 = torch.randn(vae_size, device=self.device)
        noise_2 = torch.randn(vae_size, device=self.device)
        noise = torch.stack([noise_1, noise_2], dim=1)

        time_grid = torch.tensor([0.0, 1.0])

        solver = ODESolver(velocity_model=WrappedModel(self.mmdit))

        res = solver.sample(
            x_init=noise,
            step_size=step_size,
            time_grid=time_grid,
            x_merged=x_merged,
        )
        assert isinstance(res, torch.Tensor)

        res_1 = res[:, 0, :, :].permute(0, 2, 1)
        res_2 = res[:, 1, :, :].permute(0, 2, 1)

        return res_1, res_2

    def log_audio(self, audio: np.ndarray, name: str, sampling_rate: int) -> None:
        if isinstance(self.logger, loggers.WandbLogger):
            wandb.log({name: wandb.Audio(audio, sample_rate=sampling_rate)})

    def separate(self, wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        if sr != self.cfg.model.vae.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.cfg.model.vae.sample_rate
            )

        wav_merged = torch.zeros(
            1,
            self.cfg.model.vae.sample_rate * self.cfg.model.vae.max_duration,
        )
        wav_len = wav.shape[-1]
        wav_merged[0, : wav.shape[-1]] = wav

        wav_merged_ssl = torchaudio.functional.resample(
            wav_merged,
            self.cfg.model.vae.sample_rate,
            self.cfg.model.ssl_model.sample_rate,
        )

        processor = AutoFeatureExtractor.from_pretrained(self.cfg.model.ssl_model.name)
        with torch.no_grad():
            ssl_input_merged = processor(
                [w.cpu().numpy() for w in wav_merged_ssl],
                sampling_rate=self.cfg.model.ssl_model.sample_rate,
                return_tensors="pt",
            )

        batch = {"wav_merged": wav_merged, "ssl_input_merged": ssl_input_merged}
        est_feature1, est_feature2 = self.forward(batch)

        with torch.no_grad():
            estimated_1 = (
                self.dacvae.decode(est_feature1)[0]
                .squeeze()[:wav_len]
                .to(torch.float32)
                .cpu()
            )
            estimated_2 = (
                self.dacvae.decode(est_feature2)[0]
                .squeeze()[:wav_len]
                .to(torch.float32)
                .cpu()
            )

        return estimated_1.unsqueeze(0), estimated_2.unsqueeze(0)
