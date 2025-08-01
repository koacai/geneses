from pathlib import Path

from lhotse import CutSet
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig

from dialogue_separator.data.dataset import LibriTTSRMixDataset


class AugmentDataModule2(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(AugmentDataModule2, self).__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        _ = stage

        shar_dir = Path(self.cfg.shar_dir)
        cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
        recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
        cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

        train_cuts = cuts.filter_supervisions(
            lambda s: s.custom is not None
            and s.custom["subset"] in ["train-clean-360", "train-clean-100"]
        )
        self.train_dataset = LibriTTSRMixDataset(train_cuts)

        valid_cuts = cuts.filter_supervisions(
            lambda s: s.custom is not None and s.custom["subset"] in ["dev-clean"]
        )
        self.valid_dataset = LibriTTSRMixDataset(valid_cuts)

        test_cuts = cuts.filter_supervisions(
            lambda s: s.custom is not None and s.custom["subset"] in ["test-clean"]
        )
        self.test_dataset = LibriTTSRMixDataset(test_cuts)
