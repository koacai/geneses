from pathlib import Path

import torch
import torchaudio
from lhotse import CutSet


def test_dacvae() -> None:
    dacvae = torch.jit.load("/groups/gag51394/users/asai/dacvae.pt")

    shar_dir = Path("/groups/gag51394/users/asai/shar/libri2mix_clean/")

    cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
    recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    for cut in cuts.data:
        wav = torch.from_numpy(cut.load_audio())
        wav = torchaudio.functional.resample(
            wav, orig_freq=cut.sampling_rate, new_freq=24000
        )
        wav = wav[0] + wav[1]

        wav_padded = torch.zeros(1, 1, 24000 * 20)
        wav_padded[0, 0, : wav.shape[-1]] = wav
        torchaudio.save("before.wav", wav_padded.squeeze(0), 24000)
        encoded, _, _, _ = dacvae.encode(wav_padded)
        print(encoded.shape)
        reconstructed_wav = dacvae.decode(encoded)
        print(reconstructed_wav.shape)
        torchaudio.save("reconstructed.wav", reconstructed_wav.squeeze(0), 24000)
        break
