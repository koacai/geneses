from typing import Any

import torch
from transformers import HubertModel


class FeatureExtractor:
    def __init__(self, model_name: str, layer: int) -> None:
        self.model = HubertModel.from_pretrained(model_name)
        self.model.eval()
        self.layer = layer

    @torch.inference_mode()
    def __call__(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        input_1 = batch["ssl_input_1"]
        input_2 = batch["ssl_input_2"]
        hidden_states_1 = self.model(
            **input_1, output_hidden_states=True
        ).hidden_states[self.layer]
        hidden_states_2 = self.model(
            **input_2, output_hidden_states=True
        ).hidden_states[self.layer]

        return hidden_states_1, hidden_states_2

    def to(self, device: torch.device) -> None:
        self.model = self.model.to(device)  # type: ignore
