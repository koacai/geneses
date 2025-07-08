import torch


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
