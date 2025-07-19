import numpy as np
import torch
from pystoi import stoi


def calc_estoi(ref: torch.Tensor, inf: torch.Tensor, sr: int) -> float:
    np.random.seed(0)
    return stoi(ref.cpu().numpy(), inf.cpu().numpy(), fs_sig=sr, extended=True)
