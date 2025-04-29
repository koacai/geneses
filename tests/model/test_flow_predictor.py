import pytest
import torch
from hydra import compose, initialize

from dialogue_separator.model.flow_predictor import Decoder
from dialogue_separator.utils.model import sequence_mask


class TestDecoder:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default").model.flow_predictor
            self.decoder = Decoder(**cfg)

    def test_forward(self, init) -> None:
        _ = init

        batch_size = 4
        token_merged = torch.randint(0, 2047, (batch_size, 8, 100))
        token_t = torch.randint(0, 2047, (batch_size, 8, 100))
        mask = sequence_mask(torch.tensor([95, 96, 97, 98]), 100).unsqueeze(1)
        t = torch.rand((batch_size,))

        output = self.decoder.forward(token_t, mask, token_merged, t)
        assert output.size(0) == batch_size
        assert output.size(-1) == 2048
