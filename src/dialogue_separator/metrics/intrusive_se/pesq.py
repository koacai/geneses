import torch
import torchaudio
from pesq import PesqError, pesq


def calc_pesq(ref: torch.Tensor, inf: torch.Tensor, fs: int) -> float:
    assert ref.shape == inf.shape

    if fs == 8000:
        mode = "nb"
    elif fs == 16000:
        mode = "wb"
    elif fs > 16000:
        mode = "wb"
        ref = torchaudio.functional.resample(ref, fs, 16000)
        inf = torchaudio.functional.resample(inf, fs, 16000)
        fs = 16000
    else:
        raise ValueError(
            f"sample rate must be 8000 or 16000+ for PESQ evaluation, but got {fs}"
        )

    pesq_score = pesq(
        fs,
        ref,
        inf,
        mode=mode,
        on_error=PesqError.RETURN_VALUES,
    )

    return pesq_score
