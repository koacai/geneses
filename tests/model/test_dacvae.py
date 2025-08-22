import torch

from flowditse.model.dacvae import DACVAE


def test_dacvae_encode_decode() -> None:
    ckpt_path = "/groups/gag51394/users/asai/dacvae_l16_librispeech.pt"
    dacvae = DACVAE(ckpt_path)
    wav = torch.randn((4, 16000 * 20))
    feature = dacvae.encode(wav)
    assert feature.size(0) == 4
    assert feature.size(1) == 16
