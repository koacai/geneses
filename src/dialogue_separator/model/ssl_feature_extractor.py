import torch
from torch import nn
from transformers import Wav2Vec2BertModel


class SSLFeatureExtractor(nn.Module):
    def __init__(self, ssl_model_name: str, layer: int) -> None:
        self.model = Wav2Vec2BertModel.from_pretrained(ssl_model_name).eval()
        self.layer = layer

    @torch.no_grad()
    def forward(self, input: dict[str, torch.Tensor]) -> torch.Tensor:
        feature = self.model(**input, output_hidden_states=True).hidden_states[
            self.layer
        ]
        return feature
