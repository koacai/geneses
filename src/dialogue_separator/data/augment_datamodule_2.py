from pathlib import Path
from typing import Any

from lhotse import CutSet, MultiCut
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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

        def _in_subset(cut: Any, subsets: list[str]) -> bool:
            assert isinstance(cut, MultiCut)
            assert cut.custom is not None
            return cut.custom["subset"] in subsets

        train_cuts = cuts.filter(
            lambda c: _in_subset(c, ["train-clean-360", "train-clean-100"])
        )
        self.train_dataset = LibriTTSRMixDataset(train_cuts)

        valid_cuts = cuts.filter(lambda c: _in_subset(c, ["dev-clean"]))
        self.valid_dataset = LibriTTSRMixDataset(valid_cuts)

        test_cuts = cuts.filter(lambda c: _in_subset(c, ["test-clean"]))
        self.test_dataset = LibriTTSRMixDataset(test_cuts)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
            persistent_workers=True,
        )

    def identity(self, x):
        return x[0]
