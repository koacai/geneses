import itertools

import pytest
from hydra import compose, initialize

from hubert_separator.data.datamodule import HuBERTSeparatorDataModule


class TestHuBERTSeparatorDataModule:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default").data.datamodule
            self.datamodule = HuBERTSeparatorDataModule(cfg)

    def test_train_dataloader(self, init) -> None:
        _ = init
        self.datamodule.setup("fit")

        batch_size = self.datamodule.cfg.train.batch_size
        print(batch_size)

        for batch in itertools.islice(self.datamodule.train_dataloader(), 3):
            print(batch)

    def test_val_dataloader(self, init) -> None:
        _ = init
        self.datamodule.setup("fit")

        batch_size = self.datamodule.cfg.valid.batch_size
        print(batch_size)

        for batch in itertools.islice(self.datamodule.val_dataloader(), 3):
            print(batch)
