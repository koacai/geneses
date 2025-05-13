import itertools

import pytest
import torch
from huggingface_hub import hf_hub_download
from hydra import compose, initialize
from moshi.models import loaders

from dialogue_separator.data.datamodule import DialogueSeparatorDataModule
from dialogue_separator.model.lightning_module import DialogueSeparatorLightningModule


@pytest.mark.skip("ローカルで実行するためのテスト")
class TestDialogueSeparatorLightningModule:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default")
            self.lightning_module = DialogueSeparatorLightningModule(cfg)
            self.datamodule = DialogueSeparatorDataModule(cfg.data.datamodule)
            self.datamodule.setup("fit")

            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            self.mimi = loaders.get_mimi(mimi_weight, device="cpu")
            self.mimi.set_num_codebooks(cfg.model.mimi.num_codebooks)

    def test_calc_loss(self, init) -> None:
        _ = init
        for batch in itertools.islice(self.datamodule.val_dataloader(), 3):
            loss = self.lightning_module.calc_loss(batch, self.mimi)
            assert isinstance(loss, torch.Tensor)

    def test_forward(self, init) -> None:
        _ = init
        for batch in itertools.islice(self.datamodule.val_dataloader(), 3):
            res1, res2 = self.lightning_module.forward(batch, self.mimi)
            assert isinstance(res1, torch.Tensor)
            assert isinstance(res2, torch.Tensor)
