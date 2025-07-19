import torch

from dialogue_separator.metrics.intrusive_se.pesq import calc_pesq


def test_calc_pesq() -> None:
    ref = torch.randn(16000 * 10)
    inf = torch.randn(16000 * 10)
    pesq = calc_pesq(ref, inf, 16000)
    assert isinstance(pesq, float)
