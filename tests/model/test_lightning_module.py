import itertools

import pytest
import torch
from hydra import compose, initialize

from ditse.data.datamodule import DiTSEDataModule
from ditse.model.lightning_module import DiTSELightningModule


class TestDialogueSeparatorLightningModule:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default")
            self.lightning_module = DiTSELightningModule(cfg)
            self.datamodule = DiTSEDataModule(cfg.data.datamodule)
            self.datamodule.setup("fit")

    def test_calc_loss(self, init) -> None:
        _ = init
        for batch in itertools.islice(self.datamodule.val_dataloader(), 3):
            loss = self.lightning_module.calc_loss(batch)
            assert isinstance(loss, torch.Tensor)

    def test_forward(self, init) -> None:
        _ = init
        for batch in itertools.islice(self.datamodule.val_dataloader(), 3):
            res1, res2 = self.lightning_module.forward(batch)
            assert isinstance(res1, torch.Tensor)
            assert isinstance(res2, torch.Tensor)
