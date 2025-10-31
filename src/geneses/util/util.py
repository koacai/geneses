from pathlib import Path

import requests
import torch
from tqdm import tqdm


def create_mask(lengths: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    assert lengths.dim() == 1
    assert lengths.shape[0] == feature.shape[0]

    mask = torch.zeros_like(feature, device=feature.device, dtype=torch.bool)

    for i, length in enumerate(lengths):
        if length > 0:
            mask[i, :, :length] = True
        else:
            mask[i, :, :] = False

    return mask


def download_file(url: str, path: Path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        pbar = tqdm(
            total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True
        )
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
