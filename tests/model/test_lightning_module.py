import pytest
from hydra import compose, initialize

from hubert_separator.model.lightning_module import HuBERTSeparatorLightningModule


class TestHuBERTSeparatorLightningModule:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default")
            self.lightning_module = HuBERTSeparatorLightningModule(cfg)

    def test_init(self, init) -> None:
        _ = init
