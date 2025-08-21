import numpy as np
import torch
import torchaudio
from Levenshtein import distance
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

TARGET_FS = 16000


class PhonemePredictor(nn.Module):
    # espeak installation is required for this function to work
    # To install, try
    # https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#linux
    def __init__(
        self,
        checkpoint="facebook/wav2vec2-lv-60-espeak-cv-ft",
        sr=16000,
        device: torch.device = torch.device("cpu"),
    ):
        # https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to(device=device)  # type: ignore
        self.sr = sr
        self.device = device

    def forward(self, waveform):
        input_values = self.processor(  # type: ignore
            waveform, return_tensors="pt", sampling_rate=self.sr
        ).input_values
        # retrieve logits
        logits = self.model(input_values.to(device=self.device)).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)  # type: ignore


class LevenshteinPhonemeSimilarity:
    """Levenshtein Phoneme Similarity.

    Reference:
        J. Pirklbauer, M. Sach, K. Fluyt, W. Tirry, W. Wardah, S. Moeller,
        and T. Fingscheidt, “Evaluation metrics for generative speech enhancement
        methods: Issues and perspectives,” in Speech Communication; 15th ITG Conference,
        2023, pp. 265-269.
        https://ieeexplore.ieee.org/document/10363040
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.phoneme_predictor = PhonemePredictor(device=device)

    def __call__(self, reference: np.ndarray, sample: np.ndarray) -> float:
        sample_phonemes = self.phoneme_predictor(sample)[0].replace(" ", "")
        ref_phonemes = self.phoneme_predictor(reference)[0].replace(" ", "")
        if len(ref_phonemes) == 0:
            return np.nan
        lev_distance = distance(sample_phonemes, ref_phonemes)
        return 1 - lev_distance / len(ref_phonemes)


def lps_metrics(
    model: LevenshteinPhonemeSimilarity,
    ref: torch.Tensor,
    inf: torch.Tensor,
    fs: int = 16000,
):
    """Calculate the similarity between ref and inf phoneme sequences.

    Args:
        model (torch.nn.Module): phoneme recognition model
            Please use the model in
            https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
            to get comparable results.
        ref (torch.Tensor): reference signal (time,)
        inf (torch.Tensor): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        similarity (float): phoneme similarity value between (-inf, 1]
    """
    if fs != TARGET_FS:
        ref = torchaudio.functional.resample(ref, fs, TARGET_FS)
        inf = torchaudio.functional.resample(inf, fs, TARGET_FS)
    with torch.no_grad():
        similarity = model(ref.cpu().numpy(), inf.cpu().numpy())
    return similarity
