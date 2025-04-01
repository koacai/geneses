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
        x_merged = torch.randn(batch_size, 100, 768)
        x_t = torch.randn(batch_size, 100, 768)
        t = torch.rand((batch_size,))

        dx_t = self.flow_predictor(x_merged, x_t, t)
        print(dx_t.size())
