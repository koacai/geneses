import io
import random
import uuid
from pathlib import Path
from typing import Any

import torch
import torchaudio
import webdataset as wds
from huggingface_hub import hf_hub_download
from lhotse import CutSet
from lhotse.cut import Cut
from moshi.models import loaders
from omegaconf import DictConfig


class PreprocessorLibri2Mix:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.device = torch.device(cfg.device)

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(cfg.mimi.num_codebooks)

    def write_webdataset(self) -> None:
        shar_dir = Path(self.cfg.shar_dir)
        cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
        recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
        cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

        Path(f"{self.cfg.webdataset_dir}/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.cfg.webdataset_dir}/valid").mkdir(parents=True, exist_ok=True)
        train_sink = wds.ShardWriter(
            f"{self.cfg.webdataset_dir}/train/data-%06d.tar",
            maxsize=self.cfg.shard_size.train,
        )
        valid_sink = wds.ShardWriter(
            f"{self.cfg.webdataset_dir}/valid/data-%06d.tar",
            maxsize=self.cfg.shard_size.valid,
        )

        cuts = cuts.shuffle(random.Random(42))
        for i, cut in enumerate(cuts.data):
            sample = self.process_cut(cut)
            if i < self.cfg.train_ratio * len(cuts):
                train_sink.write(sample)
            else:
                valid_sink.write(sample)

        train_sink.close()
        valid_sink.close()

    def process_cut(self, cut: Cut) -> dict[str, Any]:
        buf = io.BytesIO()
        audio = torch.from_numpy(cut.load_audio())
        torchaudio.save(buf, audio, cut.sampling_rate, format="flac")

        feature_1, feature_2, feature_merged = self.get_mimi_feature(cut)

        s = {
            "__key__": uuid.uuid1().hex,
            "audio.flac": buf.getvalue(),
            "feature_1.pth": wds.torch_dumps(feature_1.cpu()),
            "feature_2.pth": wds.torch_dumps(feature_2.cpu()),
            "feature_merged.pth": wds.torch_dumps(feature_merged.cpu()),
        }

        return s

    def get_mimi_feature(
        self, cut: Cut
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio = torch.from_numpy(cut.load_audio())

        if cut.sampling_rate != self.cfg.mimi.sr:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=cut.sampling_rate,
                new_freq=self.cfg.mimi.sr,
            )

        audio_1 = audio[0]
        audio_2 = audio[1]
        audio_merged = audio_1 + audio_2

        audio_stack = (
            torch.stack([audio_1, audio_2, audio_merged], dim=0)
            .unsqueeze(1)
            .to(self.device)
        )

        with torch.no_grad():
            codes = self.mimi.encode_to_latent(audio_stack)

        return codes[0], codes[1], codes[2]
