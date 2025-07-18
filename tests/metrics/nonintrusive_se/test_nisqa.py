import torch

from dialogue_separator.metrics.nonintrusive_se.nisqa import calc_nisqa


def test_calc_nisqa() -> None:
    audio = torch.randn(16000 * 10)
    score = calc_nisqa(audio, 16000, True)
    assert isinstance(score, float)
