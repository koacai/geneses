from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse import CutSet

from flowditse.model.lightning_module import FlowDiTSELightningModule

if __name__ == "__main__":
    shar_dir = Path("/groups/gcb50354/kohei_asai/shar/callhome_en/")

    cut_paths = sorted(list(map(str, shar_dir.glob("cuts.*.jsonl.gz"))))
    recording_paths = sorted(list(map(str, shar_dir.glob("recording.*.tar"))))

    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    ckpt_path = hf_hub_download(
        repo_id="koacai/flowditse", filename="complex_noise/epoch=20-step=151515.ckpt"
    )
    flowditse = FlowDiTSELightningModule.load_from_checkpoint(ckpt_path)

    for i, cut in enumerate(cuts.data):
        if i == 0:
            continue
        audio = torch.from_numpy(cut.load_audio())
        if audio.size(-1) > 20 * cut.sampling_rate:
            audio = audio[:, : 20 * cut.sampling_rate]

        noisy_mixed_wav = audio[0] + audio[1]
        torchaudio.save("src1.wav", audio[0].unsqueeze(0), cut.sampling_rate)
        torchaudio.save("src2.wav", audio[1].unsqueeze(0), cut.sampling_rate)

        torchaudio.save(
            "noisy_mixed_wav.wav", noisy_mixed_wav.unsqueeze(0), cut.sampling_rate
        )

        wav_1, wav_2, sr = flowditse.separate_and_enhance(
            noisy_mixed_wav, cut.sampling_rate
        )

        torchaudio.save("est1.wav", wav_1, sr)
        torchaudio.save("est2.wav", wav_2, sr)

        break
