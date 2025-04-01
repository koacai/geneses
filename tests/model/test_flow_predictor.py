import pytest
import torch
from hydra import compose, initialize

from hubert_separator.model.flow_predictor import FlowPredictor


class TestFlowPredictor:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default").model.flow_predictor
            self.flow_predictor = FlowPredictor(**cfg)

    def test_forward(self, init) -> None:
        _ = init

        batch_size = 4
        x_merged = torch.randn(batch_size, 768, 100)
        x_t = torch.randn(batch_size, 768, 100)
        mask = torch.ones(batch_size, 1, 100)
        t = torch.rand((batch_size,))

        dx_t = self.flow_predictor.forward(x_t, mask, x_merged, t)
        assert dx_t.shape == x_t.shape
