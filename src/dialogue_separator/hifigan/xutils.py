from pathlib import Path

import gdown
import torch

from .config import v1
from .env import AttrDict
from .models import Generator as HiFiGAN


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_vocoder(vocoder_path: Path):
    if not vocoder_path.exists():
        vocoder_path.parent.mkdir(exist_ok=True)

        generator_url = (
            "https://drive.google.com/uc?id=1QEBKespXTmsMzsSRBXWdpIT0Ve7nnaRZ"
        )
        gdown.download(generator_url, str(vocoder_path))

    h = AttrDict(v1)
    hifigan = HiFiGAN(h)
    hifigan.load_state_dict(torch.load(vocoder_path)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan
