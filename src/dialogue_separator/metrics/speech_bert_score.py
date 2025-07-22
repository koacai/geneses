import numpy as np
from discrete_speech_metrics import SpeechBERTScore as SBS

TARGET_FS = 16000


class SpeechBERTScore:
    """SpeechBERTScore.

    Reference:
        SpeechBERTScore: Reference-Aware Automatic Evaluation of Speech
        Generation Leveraging NLP Evaluation Metrics
        https://arxiv.org/abs/2401.16812
    """

    def __init__(self, device="cpu"):
        self.speech_bert_score = SBS(
            sr=TARGET_FS, model_type="mhubert-147", layer=8, use_gpu="cuda" in device
        )

    def __call__(self, reference: np.ndarray, sample: np.ndarray):
        precision, recall, f1_score = self.speech_bert_score.score(reference, sample)
        return precision, recall, f1_score
