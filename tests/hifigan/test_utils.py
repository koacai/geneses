from pathlib import Path

import torch

from dialogue_separator.hifigan.utils import get_vocoder


def test_get_vocoder() -> None:
    path = Path("LJ_FT_T2_V1/generator_v1")
    device = torch.device("cpu")
    hifigan = get_vocoder(path, device)
    print(hifigan)
