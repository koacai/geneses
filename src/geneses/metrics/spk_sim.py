import torch
import torchmetrics
from speechbrain.inference.speaker import EncoderClassifier


def spk_sim_metric(
    xvector: EncoderClassifier, wav1: torch.Tensor, wav2: torch.Tensor
) -> float:
    with torch.no_grad():
        xvec1 = xvector.encode_batch(wav1.unsqueeze(0)).squeeze(0)
        xvec2 = xvector.encode_batch(wav2.unsqueeze(0)).squeeze(0)

    spk_sim = torchmetrics.functional.pairwise_cosine_similarity(xvec1, xvec2)
    return spk_sim.item()
