import torch


class DACVAE:
    def __init__(self, ckpt_path: str) -> None:
        self.model = torch.jit.load(ckpt_path)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        feature, _, _, _ = self.model.encode(wav.unsqueeze(1))
        return feature

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.decode(features)

    def to(self, device: torch.device) -> None:
        self.model = self.model.to(device)
