import fast_bss_eval
import numpy as np
import torch


def calc_sdr(ref: torch.Tensor, inf: torch.Tensor) -> float:
    ref_np = ref.cpu().numpy()
    inf_np = inf.cpu().numpy()

    assert ref.shape == inf.shape
    if ref.ndim == 1:
        ref_np = ref_np[None, :]
        inf_np = inf_np[None, :]
    else:
        assert ref_np.ndim == 2, ref_np.shape

    sdr, _, _ = fast_bss_eval.bss_eval_sources(
        ref_np, inf_np, compute_permutation=False, clamp_db=50.0
    )

    return float(np.mean(sdr))  # type: ignore
