from functools import partial
from pathlib import Path
from typing import Any

import torch
import torchaudio
from lhotse import CutSet, MultiCut
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dialogue_separator.data.dataset import LibriTTSRMixDataset


class PreprocessDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(PreprocessDataModule, self).__init__()
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
        valid_cuts = cuts.filter(lambda c: _in_subset(c, ["dev-clean"]))
        test_cuts = cuts.filter(lambda c: _in_subset(c, ["test-clean"]))

        self.train_dataset = self.setup_dataset_pipeline(
            LibriTTSRMixDataset(train_cuts)
        )
        self.valid_dataset = self.setup_dataset_pipeline(
            LibriTTSRMixDataset(valid_cuts)
        )
        self.test_dataset = self.setup_dataset_pipeline(LibriTTSRMixDataset(test_cuts))

    def setup_dataset_pipeline(
        self, dataset: LibriTTSRMixDataset
    ) -> LibriTTSRMixDataset:
        return dataset

    def init_dataset(self, dataset: LibriTTSRMixDataset) -> LibriTTSRMixDataset:
        dataset = dataset.map(partial(self.lowcut, input_key="audio", cutoff=50))
        return dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            persistent_workers=True,
        )

    def collate_fn(self, batch) -> dict[str, Any]:
        text_1 = []
        text_2 = []
        for sample in batch:
            text_1.append(sample["text_1"])
            text_2.append(sample["text_2"])
        return {"text_1": text_1, "text_2": text_2}

    @staticmethod
    @torch.inference_mode()
    def lowcut(sample, input_key: str, cutoff=50):
        wav, sr = sample[input_key]
        wav = torchaudio.functional.highpass_biquad(wav, sr, cutoff)
        new_sample = sample.copy()
        new_sample[input_key] = (wav, sr)
        return new_sample
