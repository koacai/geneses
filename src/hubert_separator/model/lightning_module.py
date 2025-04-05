from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from transformers import HubertModel

import wandb
from hubert_separator.utils.model import fix_len_compatibility, sequence_mask

from .feature_extractor import FeatureExtractor
from .flow_predictor import FlowPredictor


class HuBERTSeparatorLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(HuBERTSeparatorLightningModule, self).__init__()
        self.cfg = cfg

        self.feature_extractor = FeatureExtractor(**cfg.model.hubert)
        self.hubert_model = HubertModel.from_pretrained(
            cfg.model.hubert.model_name
        ).train(True)
        self.flow_predictor_1 = FlowPredictor(**cfg.model.flow_predictor)
        self.flow_predictor_2 = FlowPredictor(**cfg.model.flow_predictor)

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.save_hyperparameters(cfg)

    def on_fit_start(self) -> None:
        self.feature_extractor.to(self.device)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return hydra.utils.instantiate(
            self.cfg.model.optimizer, params=self.parameters()
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        _ = batch_idx
        loss = self.calc_loss(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        _ = batch_idx
        loss = self.calc_loss(batch)

        self.log("validation_loss", loss)

        sr = 22050
        if batch_idx < 5 and self.global_rank == 0 and self.local_rank == 0:
            wav_len = batch["wav_len"][0]
            source_wav_1 = batch["wav_1"][0][:wav_len].cpu().numpy()
            source_wav_2 = batch["wav_2"][0][:wav_len].cpu().numpy()
            source_merged = batch["wav_merged"][0][:wav_len].cpu().numpy()

            self.log_audio(source_wav_1, f"source_wav_1/{batch_idx}", sr)
            self.log_audio(source_wav_2, f"source_wav_2/{batch_idx}", sr)
            self.log_audio(source_merged, f"source_merged/{batch_idx}", sr)

        return loss

    def calc_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        src1, src2 = self.feature_extractor(batch)

        src = self.hubert_model(
            **batch["ssl_input_merged"], output_hidden_states=True
        ).hidden_states[self.cfg.model.hubert.layer]

        batch_size = src.size(0)

        orig_len = src.size(1)
        new_len = fix_len_compatibility(orig_len)
        src = F.pad(src, (0, 0, 0, new_len - orig_len))
        src1 = F.pad(src1, (0, 0, 0, new_len - orig_len))
        src2 = F.pad(src2, (0, 0, 0, new_len - orig_len))

        lengths = orig_len * torch.ones(batch_size, device=self.device).to(self.device)
        mask = sequence_mask(lengths, new_len).unsqueeze(1).to(self.device)

        t = torch.rand((batch_size,), device=self.device)
        noise1 = torch.randn_like(src1)
        path_sample1 = self.path.sample(x_0=noise1, x_1=src1, t=t)
        noise2 = torch.randn_like(src2)
        path_sample2 = self.path.sample(x_0=noise2, x_1=src2, t=t)

        est_dxt_1 = self.flow_predictor_1.forward(path_sample1.x_t, mask, src, t)
        est_dxt_2 = self.flow_predictor_2.forward(path_sample2.x_t, mask, src, t)

        loss = self.loss(est_dxt_1, est_dxt_2, path_sample1.dx_t, path_sample2.dx_t)

        return loss

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        src1, src2 = self.feature_extractor(batch)

        src = self.hubert_model(
            **batch["ssl_input_merged"], output_hidden_states=True
        ).hidden_states[self.cfg.model.hubert.layer]

        orig_len = src.size(1)
        new_len = fix_len_compatibility(orig_len)
        src = F.pad(src, (0, 0, 0, new_len - orig_len))
        src1 = F.pad(src1, (0, 0, 0, new_len - orig_len))
        src2 = F.pad(src2, (0, 0, 0, new_len - orig_len))

        noise1 = torch.randn_like(src1)
        noise2 = torch.randn_like(src2)

        step_size = 0.001
        time_grid = torch.tensor([0.0, 1.0])

        solver_1 = ODESolver(velocity_model=self.flow_predictor_1)
        res_1 = solver_1.sample(x_init=noise1, step_size=step_size, time_grid=time_grid)
        assert isinstance(res_1, torch.Tensor)

        solver_2 = ODESolver(velocity_model=self.flow_predictor_2)
        res_2 = solver_2.sample(x_init=noise2, step_size=step_size, time_grid=time_grid)
        assert isinstance(res_2, torch.Tensor)

        return res_1, res_2

    def loss(
        self,
        est_dxt1: torch.Tensor,
        est_dxt2: torch.Tensor,
        dxt_1: torch.Tensor,
        dxt_2: torch.Tensor,
    ) -> torch.Tensor:
        l1_loss = torch.nn.L1Loss()
        return l1_loss(est_dxt1, dxt_1) + l1_loss(est_dxt2, dxt_2)

    def log_audio(self, audio: np.ndarray, name: str, sampling_rate: int) -> None:
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                wandb.log({name: wandb.Audio(audio, sample_rate=sampling_rate)})
