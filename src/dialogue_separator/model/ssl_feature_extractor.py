import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from transformers import Wav2Vec2BertModel


class SSLFeatureExtractor(nn.Module):
    def __init__(self, ssl_model_name: str, layer: int, fine_tuning_mode: str) -> None:
        super(SSLFeatureExtractor, self).__init__()

        if fine_tuning_mode == "none":
            self.model = Wav2Vec2BertModel.from_pretrained(ssl_model_name).eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.layer = layer

        elif fine_tuning_mode == "lora":
            self.model = Wav2Vec2BertModel.from_pretrained(
                ssl_model_name, layerdrop=0.0
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

        elif fine_tuning_mode == "hf_adapter":
            self.model = Wav2Vec2BertModel.from_pretrained(
                ssl_model_name, add_adapter=True
            )
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.adapter.parameters():  # type: ignore
                param.requires_grad = True
            self.layer = layer

        else:
            raise ValueError(f"unknown fine tuning mode: {fine_tuning_mode}")

    @torch.no_grad()
    def forward(self, input: dict[str, torch.Tensor]) -> torch.Tensor:
        feature = self.model(**input, output_hidden_states=True).hidden_states[
            self.layer
        ]
        return feature
