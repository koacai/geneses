from typing import Any

import hydra
import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from omegaconf import DictConfig
from transformers import Wav2Vec2BertModel

import wandb

from .components import MMDiT


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        x_merged = extras.get("x_merged", None)
        assert x_merged is not None
        x_1 = x[:, 0, :, :]
        x_2 = x[:, 1, :, :]
        res_1, res_2 = self.model.forward(x_merged, t.unsqueeze(0), x_1, x_2)
        return torch.stack([res_1, res_2], dim=1)


class SSLFeatureExtractor:
    def __init__(self, model_name: str, layer: int, add_adapter: bool) -> None:
        self.model = Wav2Vec2BertModel.from_pretrained(
            model_name, add_adapter=add_adapter
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer = layer

    @torch.no_grad()
    def extract(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**inputs, output_hidden_states=True).hidden_states[self.layer]

    def to(self, device: torch.device) -> None:
        self.model = self.model.to(device)  # type: ignore


class DACVAE:
    def __init__(self, ckpt_path: str) -> None:
        self.model = torch.jit.load(ckpt_path)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model.encode(wav)

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

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        _ = batch_idx

        loss = self.calc_loss(batch)

        self.log("train_loss", loss)

        return loss

    def on_fit_start(self) -> None:
        self.ssl_feature_extractor.to(self.device)
        self.dacvae.to(self.device)

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        _ = batch_idx

        loss = self.calc_loss(batch)

        self.log("validation_loss", loss)

        wav_sr = self.cfg.model.sample_rate
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

    def calc_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            x_1, _, _, _ = self.dacvae.encode(batch["wav_1"].unsqueeze(1))
            x_2, _, _, _ = self.dacvae.encode(batch["wav_2"].unsqueeze(1))
            x_1 = x_1.permute(0, 2, 1)
            x_2 = x_2.permute(0, 2, 1)
            x_merged = self.ssl_feature_extractor.extract(batch["ssl_input_merged"])

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

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_merged = self.ssl_feature_extractor.extract(batch["ssl_input_merged"])
            vae_merged, _, _, _ = self.dacvae.encode(batch["wav_merged"].unsqueeze(1))
            vae_merged = vae_merged.permute(0, 2, 1)

        noise_1 = torch.randn_like(vae_merged)
        noise_2 = torch.randn_like(vae_merged)
        noise = torch.stack([noise_1, noise_2], dim=1)

        step_size = 0.1
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
