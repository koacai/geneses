import torch

from dialogue_separator.metrics.nonintrusive_se.utmos import calc_utmos


def test_calc_utmos() -> None:
    audio = torch.randn(16000 * 10)
    score = calc_utmos(audio, 16000, True)
    assert isinstance(score, float)
