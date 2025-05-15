import json

import hydra
import numpy as np
import torch
import torchaudio
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from huggingface_hub import hf_hub_download
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from moshi.models import loaders
from omegaconf import DictConfig

import wandb

from .mmdit_model import MMDiT


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        x_merged = extras.get("x_merged", None)
        assert x_merged is not None
        x_1 = x[:, 0, :, :]
        x_2 = x[:, 1, :, :]
        res_1, res_2 = self.model.forward(x_merged, t.unsqueeze(0), x_1, x_2)
        return torch.stack([res_1, res_2], dim=1)


class DialogueSeparatorLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(DialogueSeparatorLightningModule, self).__init__()
        self.cfg = cfg

        self.mmdit = MMDiT(**cfg.model.mmdit)

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(self.cfg.model.mimi.num_codebooks)
        for param in self.mimi.parameters():
            param.requires_grad = False

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        with open(f"{cfg.model.stats_path}", "r") as f:
            self.stats = json.load(f)

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

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        _ = batch_idx

        loss = self.calc_loss(batch)

        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        _ = batch_idx

        loss = self.calc_loss(batch)

        self.log("validation_loss", loss)

        wav_sr = self.cfg.model.mimi.sr
        if batch_idx < 5 and self.global_rank == 0 and self.local_rank == 0:
            wav_len = batch["wav_len"][0]
            source_1 = batch["wav_1"][0][:wav_len].cpu().numpy()
            source_2 = batch["wav_2"][0][:wav_len].cpu().numpy()
            source_merged = batch["wav_merged"][0][:wav_len].cpu().numpy()

            self.log_audio(source_1, f"source_1/{batch_idx}", wav_sr)
            self.log_audio(source_2, f"source_2/{batch_idx}", wav_sr)
            self.log_audio(source_merged, f"source_merged/{batch_idx}", wav_sr)

            with torch.no_grad():
                feature_1 = self.denormalize_feature(batch["feature_1"])
                code_1 = self.mimi.quantizer.encode(feature_1)
                decoded_1 = (
                    self.mimi.decode(code_1)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                feature_2 = self.denormalize_feature(batch["feature_2"])
                code_2 = self.mimi.quantizer.encode(feature_2)
                decoded_2 = (
                    self.mimi.decode(code_2)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                feature_merged = self.denormalize_feature(batch["feature_merged"])
                code_merged = self.mimi.quantizer.encode(feature_merged)
                decoded_merged = (
                    self.mimi.decode(code_merged)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

            self.log_audio(decoded_1, f"decoded_1/{batch_idx}", wav_sr)
            self.log_audio(decoded_2, f"decoded_2/{batch_idx}", wav_sr)
            self.log_audio(decoded_merged, f"decoded_merged/{batch_idx}", wav_sr)

            est_feature1, est_feature2 = self.forward(batch)

            with torch.no_grad():
                code_1 = self.mimi.quantizer.encode(est_feature1)
                estimated_1 = (
                    self.mimi.decode(code_1)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                code_2 = self.mimi.quantizer.encode(est_feature2)
                estimated_2 = (
                    self.mimi.decode(code_2)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

            self.log_audio(estimated_1, f"estimated_1/{batch_idx}", wav_sr)
            self.log_audio(estimated_2, f"estimated_2/{batch_idx}", wav_sr)

        return loss

    def calc_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_1 = batch["feature_1"].permute(0, 2, 1)
        x_2 = batch["feature_2"].permute(0, 2, 1)
        x_merged = batch["feature_merged"].permute(0, 2, 1)

        batch_size = x_merged.size(0)

        t = torch.rand((batch_size,), device=self.device)
        noise_1 = torch.randn_like(x_1)
        path_sample1 = self.path.sample(x_0=noise_1, x_1=x_1, t=t)
        noise_2 = torch.randn_like(x_2)
        path_sample2 = self.path.sample(x_0=noise_2, x_1=x_2, t=t)

        est_dxt_1, est_dxt_2 = self.mmdit.forward(
            x_merged, t, path_sample1.x_t, path_sample2.x_t
        )

        loss = self.loss_fn(est_dxt_1, est_dxt_2, path_sample1.dx_t, path_sample2.dx_t)

        return loss

    def loss_fn(
        self,
        est_dxt1: torch.Tensor,
        est_dxt2: torch.Tensor,
        dxt_1: torch.Tensor,
        dxt_2: torch.Tensor,
    ) -> torch.Tensor:
        l1_loss = torch.nn.L1Loss()
        return l1_loss(est_dxt1, dxt_1) + l1_loss(est_dxt2, dxt_2)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_merged = batch["feature_merged"].permute(0, 2, 1)

        noise_1 = torch.randn_like(x_merged)
        noise_2 = torch.randn_like(x_merged)
        noise = torch.stack([noise_1, noise_2], dim=1)

        step_size = 0.01
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

        return self.denormalize_feature(res_1), self.denormalize_feature(res_2)

    def separate_wav(
        self, wav: torch.Tensor, sr: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wav = wav.to(self.device)

        if sr != self.cfg.model.mimi.sr:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.model.mimi.sr)

        feature_merged = self.mimi.encode_to_latent(wav.unsqueeze(0), quantize=False)
        batch = {"feature_merged": self.normalize_feature(feature_merged)}

        est_feature1, est_feature2 = self.forward(batch)

        with torch.no_grad():
            code_1 = self.mimi.quantizer.encode(est_feature1)
            estimated_1 = self.mimi.decode(code_1)[0].to(torch.float32).cpu()
            code_2 = self.mimi.quantizer.encode(est_feature2)
            estimated_2 = self.mimi.decode(code_2)[0].to(torch.float32).cpu()

        return estimated_1, estimated_2

    def log_audio(self, audio: np.ndarray, name: str, sampling_rate: int) -> None:
        if isinstance(self.logger, loggers.WandbLogger):
            wandb.log({name: wandb.Audio(audio, sample_rate=sampling_rate)})

    def normalize_feature(self, feature: torch.Tensor) -> torch.Tensor:
        return (feature - self.stats["mean"]) / self.stats["std"]

    def denormalize_feature(self, feature: torch.Tensor) -> torch.Tensor:
        return feature * self.stats["std"] + self.stats["mean"]
