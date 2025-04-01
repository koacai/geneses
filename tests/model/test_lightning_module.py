import itertools

import pytest
import torch
from hydra import compose, initialize

from hubert_separator.data.datamodule import HuBERTSeparatorDataModule
from hubert_separator.model.lightning_module import HuBERTSeparatorLightningModule


class TestHuBERTSeparatorLightningModule:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default")
            self.lightning_module = HuBERTSeparatorLightningModule(cfg)
            self.datamodule = HuBERTSeparatorDataModule(cfg.data.datamodule)
            self.datamodule.setup("fit")

    def test_init(self, init) -> None:
        _ = init

    def test_calc_loss(self, init) -> None:
        _ = init
        for batch in itertools.islice(self.datamodule.val_dataloader(), 3):
            loss = self.lightning_module.calc_loss(batch)
            assert isinstance(loss, torch.Tensor)
