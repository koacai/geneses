from typing import Any

import hydra
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from transformers import HubertModel


class HuBERTSeparatorLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(HuBERTSeparatorLightningModule, self).__init__()
        self.cfg = cfg

        self.hubert_model = HubertModel.from_pretrained(
            cfg.model.hubert.model_name
        ).train(True)

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
        src = self.hubert_model(
            **batch["ssl_input_merged"], output_hidden_states=True
        ).hidden_states[self.cfg.model.hubert.layer]

        return src
