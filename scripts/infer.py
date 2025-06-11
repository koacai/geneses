from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse import CutSet

from dialogue_separator.model.lightning_module import DialogueSeparatorLightningModule

if __name__ == "__main__":
    ckpt_path = hf_hub_download(
        "koacai/dialogue-separator", "Libri2Mix/epoch=50-step=25449.ckpt"
    )
    dialogue_separator = DialogueSeparatorLightningModule.load_from_checkpoint(
        ckpt_path
    ).eval()

    shar_dir = Path("/groups/gag51394/users/asai/shar/librispeech/")
    cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
    recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    for cut in cuts.data:
        cut.save_audio("source.wav")
        wav = torch.from_numpy(cut.load_audio())
        wav_1, wav_2 = dialogue_separator.separate(wav, cut.sampling_rate)
        torchaudio.save(
            "output_1.wav",
            wav_1,
            dialogue_separator.cfg.data.datamodule.vae.sample_rate,
        )
        torchaudio.save(
            "output_2.wav",
            wav_2,
            dialogue_separator.cfg.data.datamodule.vae.sample_rate,
        )
        break
