import torch

from dialogue_separator.model.components import MMDiT


def test_mmdit() -> None:
    model = MMDiT(
        hidden_size=64,
        max_seq_len=1000,
        depth=12,
        heads=12,
    )
    x_merged = torch.randn(1, 100, 64)
    x_1 = torch.randn(1, 100, 64)
    x_2 = torch.randn(1, 100, 64)
    t = torch.rand((1,))
    res1, res2 = model.forward(x_merged, t, x_1, x_2)
    assert res1.shape == (1, 100, 64)
    assert res2.shape == (1, 100, 64)
