import torch
from hydra import compose, initialize

from dialogue_separator.model.flow_predictor import FlowPredictor, MimiEmbedding
from dialogue_separator.utils.model import sequence_mask


def test_flow_predictor_forward() -> None:
    with initialize(config_path="../../config", version_base=None):
        cfg = compose(config_name="default").model.flow_predictor
        flow_predictor = FlowPredictor(cfg)

    batch_size = 4
    token_merged = torch.randint(0, 2047, (batch_size, 8, 100))
    token_t_1 = torch.randint(0, 2047, (batch_size, 8, 100))
    token_t_2 = torch.randint(0, 2047, (batch_size, 8, 100))
    mask = sequence_mask(torch.tensor([95, 96, 97, 98]), 100).unsqueeze(1)
    t = torch.rand((batch_size,))

    token_t = torch.stack([token_t_1, token_t_2], dim=1)

    res = flow_predictor.forward(token_t, mask, token_merged, t)
    assert res.size() == (batch_size, 2, 8, 100, 2048)


def test_mimi_embedding_forward() -> None:
    mimi_token_embedding = MimiEmbedding(
        num_codebooks=8, vocab_size=2048, hidden_size=512
    )
    token = torch.randint(0, 2047, (4, 8, 100))
    embedding = mimi_token_embedding.forward(token)
    assert embedding.size() == (4, 100, 512)
