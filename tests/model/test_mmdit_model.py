import torch

from dialogue_separator.model.mmdit_model import MMDiT


def test_mmdit() -> None:
    model = MMDiT(
        in_channels=64,
        out_channels=64,
        hidden_size=384,
        depth=1,
        max_seq_len=4096,
        mel_size=128,
        mel_hidden_size=512,
        max_mel_len=4096,
        ssl_size=512,
        ssl_hidden_size=512,
        max_ssl_len=4096,
    )
    x = torch.randn(1, 4, 64)
    mel = torch.randn(1, 57, 128)
    ssl = torch.randn(1, 6, 512)
    t = torch.rand((1,))
    out = model.forward(x, t, ssl_feature=ssl, mel=mel)
    out.mean().backward()
    assert out.shape == (1, 4, 64)
