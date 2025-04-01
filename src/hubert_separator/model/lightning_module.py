from lightning.pytorch import LightningModule
from omegaconf import DictConfig


class HuBERTSeparatorLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(HuBERTSeparatorLightningModule, self).__init__()
        self.cfg = cfg
