from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse import CutSet
from moshi.models import loaders

if __name__ == "__main__":
    shar_dir = Path("/groups/gag51394/users/asai/shar/jvs")
    cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
    recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cpu")
    mimi.set_num_codebooks(8)

    for cut in cuts.data:
        print(cut)
        wav = torch.from_numpy(cut.load_audio()).unsqueeze(0)
        print(wav.shape)
        with torch.no_grad():
            codes = mimi.encode(wav)  # [B, K = 8, T]
            print(codes)
            print(torch.max(codes))
            print(codes.shape)
            decoded = mimi.decode(codes)
            print(decoded.shape)

        torchaudio.save("decoded.wav", decoded.squeeze(0), 24000)

        break
