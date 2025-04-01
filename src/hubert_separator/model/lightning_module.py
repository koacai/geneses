import hydra
import torch
from lightning.pytorch import LightningModule
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
