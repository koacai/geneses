import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from transformers import Wav2Vec2BertModel


class SSLFeatureExtractor(nn.Module):
    def __init__(self, ssl_model_name: str, layer: int) -> None:
        super(SSLFeatureExtractor, self).__init__()

        self.model = Wav2Vec2BertModel.from_pretrained(
            ssl_model_name, num_hidden_layers=8, layerdrop=0.0
        ).train()
        adapter_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="lora_only",
            target_modules=["output_dense"],
        )
        self.model = inject_adapter_in_model(adapter_config, self.model)
        self.layer = layer

    @torch.no_grad()
    def forward(self, input: dict[str, torch.Tensor]) -> torch.Tensor:
        feature = self.model(**input, output_hidden_states=True).hidden_states[
            self.layer
        ]
        return feature
