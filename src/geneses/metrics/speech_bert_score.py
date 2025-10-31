import numpy as np
import torch
import torchaudio
from discrete_speech_metrics import SpeechBERTScore as SBS

TARGET_FS = 16000


class SpeechBERTScore:
    """SpeechBERTScore.

    Reference:
        SpeechBERTScore: Reference-Aware Automatic Evaluation of Speech
        Generation Leveraging NLP Evaluation Metrics
        https://arxiv.org/abs/2401.16812
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.speech_bert_score = SBS(
            sr=TARGET_FS,
            model_type="mhubert-147",
            layer=8,
            use_gpu=device == torch.device("cuda"),
        )

    def __call__(self, reference: np.ndarray, sample: np.ndarray):
        precision, recall, f1_score = self.speech_bert_score.score(reference, sample)
        return precision, recall, f1_score


def speech_bert_score_metric(
    model: SpeechBERTScore, ref: torch.Tensor, inf: torch.Tensor, fs: int = 16000
):
    """Calculate the SpeechBERTScore between ref and inf.

    Args:
        model (torch.nn.Module): SpeechBERTScore model
            Please use the model with model_type="mhubert-147" and layer=8
            to get comparable results on multilingual data.
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        score (float): SpeechBERTScore precision value between [0, 1]
    """
    if fs != TARGET_FS:
        ref = torchaudio.functional.resample(ref, fs, TARGET_FS)
        inf = torchaudio.functional.resample(inf, fs, TARGET_FS)
    with torch.no_grad():
        score = model(ref.cpu().numpy(), inf.cpu().numpy())
    return score[0]
