from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse import CutSet
from moshi.models import loaders

if __name__ == "__main__":
    dir = "/groups/gag51394/users/asai/shar/libri2mix"
    shar_dir = Path(dir)
    cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
    recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cpu")
    mimi.set_num_codebooks(8)

    for cut in cuts.data:
        wav = torch.from_numpy(cut.load_audio())
        wav = torchaudio.functional.resample(wav, cut.sampling_rate, 24000)
        wav = wav[0, :] + wav[1, :]
        # torchaudio.save("before.wav", wav.unsqueeze(0), 24000)
        wav = wav.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            codes = mimi.encode(wav)
            latent = mimi.decode_latent(codes)
            rec = mimi.decode(codes)

        print(rec.size())
        print(latent.size())

        # torchaudio.save("test.wav", rec.squeeze(0), 24000)

        break
