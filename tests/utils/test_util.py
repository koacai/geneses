import torch

from dialogue_separator.util.util import create_mask


def test_create_mask() -> None:
    lengths = torch.tensor([1, 2, 3, 4])
    feature = torch.randn(4, 2, 5)
    created_mask = create_mask(lengths, feature)
    print(created_mask[3])
    print(created_mask.shape)
