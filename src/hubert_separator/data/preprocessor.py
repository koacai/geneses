import random
import uuid
from pathlib import Path
from typing import Any

import webdataset as wds
from lhotse import CutSet
from lhotse.cut import Cut
from omegaconf import DictConfig


class Preprocessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

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
            samples = self.process_cut(cut)
            for sample in samples:
                if i < self.cfg.train_ratio * len(cuts):
                    train_sink.write(sample)
                else:
                    valid_sink.write(sample)

        train_sink.close()
        valid_sink.close()

    def process_cut(self, cut: Cut) -> list[dict[str, Any]]:
        cuts = cut.cut_into_windows(duration=self.cfg.duration)
        res = []
        for c in cuts.data:
            s = {
                "__key__": uuid.uuid1().hex,
                "audio.flac": c.recording.sources[0].source,  # type: ignore
            }
            res.append(s)
        return res
