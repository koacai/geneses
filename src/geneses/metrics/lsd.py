import librosa
import numpy as np
import torch


def lsd_metric(
    ref: torch.Tensor,
    inf: torch.Tensor,
    fs: int,
    nfft=0.032,
    hop=0.016,
    p=2,
    eps=1.0e-08,
):
    """Calculate Log-Spectral Distance (LSD).

    Args:
        ref (torch.Tensor): reference signal (time,)
        inf (torch.Tensor): enhanced signal (time,)
        fs (int): sampling rate in Hz
        nfft (float): FFT length in seconds
        hop (float): hop length in seconds
        p (float): the order of norm
        eps (float): epsilon value for numerical stability
    Returns:
        mcd (float): LSD value between [0, +inf)
    """
    ref_np = ref.cpu().numpy()
    inf_np = inf.cpu().numpy()
    scaling_factor = np.sum(ref_np * inf_np) / (np.sum(inf_np**2) + eps)
    inf_np = inf_np * scaling_factor

    nfft = int(fs * nfft)
    hop = int(fs * hop)
    # T x F
    ref_spec = np.abs(librosa.stft(ref_np, hop_length=hop, n_fft=nfft)).T
    inf_spec = np.abs(librosa.stft(inf_np, hop_length=hop, n_fft=nfft)).T
    lsd = np.log(ref_spec**2 / ((inf_spec + eps) ** 2) + eps) ** p
    lsd = np.mean(np.mean(lsd, axis=1) ** (1 / p), axis=0)
    return lsd
