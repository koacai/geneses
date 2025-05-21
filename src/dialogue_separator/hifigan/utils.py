from pathlib import Path

import gdown
import torch

from .config import v1
from .env import AttrDict
from .models import Generator as HiFiGAN


def get_vocoder(vocoder_path: Path, device: torch.device):
    if not vocoder_path.exists():
        vocoder_path.parent.mkdir(exist_ok=True)

        generator_url = (
            "https://drive.google.com/uc?id=1QEBKespXTmsMzsSRBXWdpIT0Ve7nnaRZ"
        )
        gdown.download(generator_url, str(vocoder_path))

    h = AttrDict(v1)
    hifigan = HiFiGAN(h)
    hifigan.load_state_dict(torch.load(vocoder_path, map_location=device)["generator"])
    hifigan = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan
