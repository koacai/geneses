import torch

from dialogue_separator.metrics.intrusive_se.estoi import calc_estoi


def test_calc_estoi() -> None:
    ref = torch.randn(16000 * 10)
    inf = torch.randn(16000 * 10)
    estoi = calc_estoi(ref, inf, 16000)
    assert isinstance(estoi, float)
