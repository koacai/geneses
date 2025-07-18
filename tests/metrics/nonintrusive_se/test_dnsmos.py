import torch

from dialogue_separator.metrics.nonintrusive_se.dnsmos import calc_dnsmos


def test_dnsmos() -> None:
    audio = torch.randn(16000 * 10)
    score = calc_dnsmos(audio, 16000, use_gpu=True)
    assert isinstance(score, float)
