import torch

from geneses.model.components import MMDiT


def test_mmdit() -> None:
    model = MMDiT(
        in_channels=16,
        in_ssl_channels=1024,
        out_channels=16,
        hidden_size=768,
        max_ssl_seq_len=1000,
        max_seq_len=334,
        depth=12,
        heads=12,
    )
    x_merged = torch.randn(1, 100, 1024)
    x_1 = torch.randn(1, 100, 16)
    x_2 = torch.randn(1, 100, 16)
    t = torch.rand((1,))
    res1, res2 = model.forward(x_merged, t, x_1, x_2)
    assert res1.shape == (1, 100, 16)
    assert res2.shape == (1, 100, 16)
