from typing import Any

import hydra
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from transformers import HubertModel

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
        return loss

    def calc_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        src1, src2 = self.feature_extractor(batch)

        src = self.hubert_model(
            **batch["ssl_input_merged"], output_hidden_states=True
        ).hidden_states[self.cfg.model.hubert.layer]

        batch_size = src.size(0)
        t = torch.rand((batch_size,), device=self.device)
        noise1 = torch.randn_like(src1)
        path_sample1 = self.path.sample(x_0=noise1, x_1=src1, t=t)
        noise2 = torch.randn_like(src2)
        path_sample2 = self.path.sample(x_0=noise2, x_1=src2, t=t)

        mask = torch.ones(batch_size, 1, src.size(1), device=self.device)

        est_dxt_1 = self.flow_predictor_1.forward(path_sample1.x_t, mask, src, t)
        est_dxt_2 = self.flow_predictor_2.forward(path_sample2.x_t, mask, src, t)

        loss = self.loss(est_dxt_1, est_dxt_2, path_sample1.dx_t, path_sample2.dx_t)

        return loss

    def loss(
        self,
        est_dxt1: torch.Tensor,
        est_dxt2: torch.Tensor,
        dxt_1: torch.Tensor,
        dxt_2: torch.Tensor,
    ) -> torch.Tensor:
        l1_loss = torch.nn.L1Loss()
        return l1_loss(est_dxt1, dxt_1) + l1_loss(est_dxt2, dxt_2)
