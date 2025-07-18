import tempfile
import warnings

import torch
import torchaudio

from dialogue_separator.metrics.nonintrusive_se.nisqa_util import (
    load_nisqa_model,
    predict_nisqa,
)


def calc_nisqa(audio: torch.Tensor, sr: int, use_gpu: bool) -> float:
    nisqa_model_path = "src/dialogue_separator/lib/NISQA/weights/nisqa.tar"
    model = load_nisqa_model(nisqa_model_path, device="cuda" if use_gpu else "cpu")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        audio_path = tmp_file.name
        torchaudio.save(audio_path, audio.cpu().unsqueeze(0), sr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        nisqa_score = predict_nisqa(model, audio_path)

    return nisqa_score["mos_pred"]
