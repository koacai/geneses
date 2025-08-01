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
            LibriTTSRMixDataset(train_cuts), self.cfg.batch_size
        )
        self.valid_dataset = self.setup_dataset_pipeline(
            LibriTTSRMixDataset(valid_cuts), self.cfg.batch_size
        )
        self.test_dataset = self.setup_dataset_pipeline(
            LibriTTSRMixDataset(test_cuts), self.cfg.batch_size
        )

    def setup_dataset_pipeline(
        self, dataset: LibriTTSRMixDataset, batch_size: int
    ) -> LibriTTSRMixDataset:
        dataset = self.init_dataset(dataset).batched(
            batch_size, collation_fn=self.collate_fn
        )
        return dataset

    def init_dataset(self, dataset: LibriTTSRMixDataset) -> LibriTTSRMixDataset:
        dataset = (
            dataset.map(partial(self.lowcut, input_key="audio", cutoff=50))
            .map(
                partial(self.rename_audio, input_key="audio", output_key="clean_stereo")
            )
            .map(partial(self.stereo_to_mono, input_key="audio", output_key="noisy"))
            .map(partial(self.stereo_to_mono, input_key="audio", output_key="clean"))
        )
        return dataset

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

    def collate_fn(self, batch) -> dict[str, Any]:
        text_1 = []
        text_2 = []
        for sample in batch:
            text_1.append(sample["text_1"])
            text_2.append(sample["text_2"])
        return {"text_1": text_1, "text_2": text_2}

    def identity(self, x):
        return x[0]

    @staticmethod
    @torch.inference_mode()
    def lowcut(sample, input_key: str, cutoff=50):
        wav, sr = sample[input_key]
        wav = torchaudio.functional.highpass_biquad(wav, sr, cutoff)
        new_sample = sample.copy()
        new_sample[input_key] = (wav, sr)
        return new_sample

    @staticmethod
    def rename_audio(sample, output_key: str, input_key: str | None = None):
        if input_key is None:
            audio_key = [k for k in sample.keys() if "audio" in k][0]
        else:
            audio_key = input_key
        sample[output_key] = sample[audio_key]
        return sample

    @staticmethod
    def stereo_to_mono(sample, output_key: str, input_key: str):
        wav, sr = sample[input_key]
        assert wav.shape[0] == 2
        sample[output_key] = (wav[0] + wav[1], sr)
        return sample
