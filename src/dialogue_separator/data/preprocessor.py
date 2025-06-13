import io
import random
import uuid
from pathlib import Path
from typing import Any

import torch
import torchaudio
import webdataset as wds
from lhotse import CutSet, MultiCut
from lhotse.cut import Cut
from omegaconf import DictConfig


class Preprocessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.dacvae = torch.jit.load(cfg.vae.ckpt_path).to(self.device)

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
        for cut in cuts.data:
            sample = self.process_cut(cut)

            assert isinstance(cut, MultiCut)
            assert isinstance(cut.custom, dict)

            if cut.custom["subset"] == "dev-clean":
                valid_sink.write(sample)
            elif cut.custom["subset"] in ["train-clean-100", "train-clean-360"]:
                train_sink.write(sample)

        train_sink.close()
        valid_sink.close()

    def process_cut(self, cut: Cut) -> dict[str, Any]:
        buf = io.BytesIO()
        audio = torch.from_numpy(cut.load_audio())
        torchaudio.save(buf, audio, cut.sampling_rate, format="flac")

        vae_feature_1, vae_feature_2 = self.vae_encode(cut)

        s = {
            "__key__": uuid.uuid1().hex,
            "audio.flac": buf.getvalue(),
            "vae_feature_1.pth": wds.torch_dumps(vae_feature_1.cpu()),
            "vae_feature_2.pth": wds.torch_dumps(vae_feature_2.cpu()),
        }

        return s

    def vae_encode(self, cut: Cut) -> tuple[torch.Tensor, torch.Tensor]:
        audio = torch.from_numpy(cut.load_audio()).to(self.device)

        with torch.no_grad():
            feature, _, _, _ = self.dacvae.encode(audio.unsqueeze(1))

        return feature[0], feature[1]
