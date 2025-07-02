from pathlib import Path

import torch
import torchaudio
from lhotse import CutSet


def test_dacvae() -> None:
    dacvae = torch.jit.load("/groups/gag51394/users/asai/dacvae_l16_librispeech.pt")

    shar_dir = Path("/groups/gag51394/users/asai/shar/librispeech/")

    cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
    recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    sr = 16000
    for cut in cuts.data:
        wav = torch.from_numpy(cut.load_audio())
        wav = torchaudio.functional.resample(
            wav, orig_freq=cut.sampling_rate, new_freq=sr
        )

        rms_noise = 10 ** (-4)

        wav_padded = torch.randn(1, 1, sr * 20) * rms_noise
        wav_padded[0, 0, : wav.shape[-1]] = wav

        torchaudio.save("source.wav", wav_padded.squeeze(0), sr)
        encoded, _, _, _ = dacvae.encode(wav_padded)
        reconstructed_wav = dacvae.decode(encoded)
        torchaudio.save("reconstructed.wav", reconstructed_wav.squeeze(0), sr)
        break
