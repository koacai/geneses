import torch


def test_dacvae() -> None:
    dacvae = torch.jit.load("/groups/gag51394/users/asai/dacvae.pt")
    wav = torch.randn(1, 1, 24000 * 20)  # need to be 20 seconds
    encoded, _, _, _ = dacvae.encode(wav)
    print(encoded.shape)
    reconstructed_wav = dacvae.decode(encoded)
    print(reconstructed_wav.shape)
