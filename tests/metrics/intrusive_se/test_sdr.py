import torch

from dialogue_separator.metrics.intrusive_se.sdr import calc_sdr


def test_calc_sdr() -> None:
    ref = torch.randn(16000 * 10)
    inf = torch.randn(16000 * 10)
    sdr = calc_sdr(ref, inf)
    assert isinstance(sdr, float)
