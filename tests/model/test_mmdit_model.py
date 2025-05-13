import torch

from dialogue_separator.model.mmdit_model import MMDiT


def test_mmdit() -> None:
    model = MMDiT(
        in_channels=512,
        out_channels=512,
        hidden_size=512,
        max_seq_len=1000,
        depth=1,
        heads=8,
    )
    x_merged = torch.randn(1, 100, 512)
    x_1 = torch.randn(1, 100, 512)
    x_2 = torch.randn(1, 100, 512)
    t = torch.rand((1,))
    res1, res2 = model.forward(x_merged, t, x_1, x_2)
    assert res1.shape == (1, 100, 512)
    assert res2.shape == (1, 100, 512)
