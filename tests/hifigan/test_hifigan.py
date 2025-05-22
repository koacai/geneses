from pathlib import Path

import torch
import torchaudio
from lhotse import CutSet

from dialogue_separator.hifigan.denoiser import Denoiser
from dialogue_separator.hifigan.utils import get_vocoder
from dialogue_separator.utils.mel import mel_spectrogram


def test_hifigan() -> None:
    path = Path("LJ_FT_T2_V1/generator_v1")
    device = torch.device("cpu")
    vocoder = get_vocoder(path, device)

    shar_dir = Path("/groups/gag51394/users/asai/shar/librispeech/")
    cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
    recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    n_fft = 1024
    n_mels = 80
    sample_rate = 22050
    hop_length = 256
    win_length = 1024
    f_min = 0
    f_max = 8000

    for cut in cuts.data:
        cut.save_audio("before.wav")
        wav = torch.from_numpy(cut.load_audio())
        wav = torchaudio.functional.resample(wav, cut.sampling_rate, sample_rate)

        mel = mel_spectrogram(
            wav,
            n_fft,
            n_mels,
            sample_rate,
            hop_length,
            win_length,
            f_min,
            f_max,
            center=False,
        )

        denoiser = Denoiser(vocoder, mode="zeros").eval()
        audio = vocoder(mel).clamp(-1, 1)
        audio = denoiser(audio.squeeze(0), strength=0.00025)
        torchaudio.save("after.wav", audio, sample_rate)
        break
