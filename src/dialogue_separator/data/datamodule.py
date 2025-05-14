import json
from typing import Any

import torch
import torchaudio
import webdataset as wds
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence


class DialogueSeparatorDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(DialogueSeparatorDataModule, self).__init__()
        self.cfg = cfg

        with open(f"{cfg.stats_path}", "r") as f:
            self.stats = json.load(f)

    def setup(self, stage: str) -> None:
        nodesplitter = wds.split_by_worker if self.cfg.use_ddp else wds.single_node_only
        if stage == "fit":
            self.train_dataset = (
                wds.WebDataset(
                    self.cfg.train.dataset_path,
                    shardshuffle=100,
                    nodesplitter=nodesplitter,
                    repeat=True,
                )
                .decode(wds.torch_audio)
                .batched(self.cfg.train.batch_size, collation_fn=self.collate_fn)
            )

            self.valid_dataset = (
                wds.WebDataset(
                    self.cfg.valid.dataset_path,
                    shardshuffle=False,
                    nodesplitter=nodesplitter,
                    repeat=True,
                )
                .decode(wds.torch_audio)
                .batched(self.cfg.valid.batch_size, collation_fn=self.collate_fn)
            )

    def train_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
        )

    def val_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
        )

    def identity(self, x):
        return x[0]

    def collate_fn(self, batch) -> dict[str, Any]:
        max_duration = self.cfg.max_duration

        wav1_resample = []
        wav2_resample = []
        wav_merged_resample = []
        wav_len = []

        feature_1 = []
        feature_2 = []
        feature_merged = []
        feature_len = []

        for sample in batch:
            dialogue, sr = sample["audio.flac"]

            dialogue_resample = torchaudio.functional.resample(
                dialogue, sr, self.cfg.sample_rate
            )
            wav1_resample_ = dialogue_resample[
                0, : self.cfg.sample_rate * max_duration
            ].numpy()
            wav2_resample_ = dialogue_resample[
                1, : self.cfg.sample_rate * max_duration
            ].numpy()
            wav1_resample.append(wav1_resample_)
            wav2_resample.append(wav2_resample_)
            wav_merged_resample.append(wav1_resample_ + wav2_resample_)

            wav_len.append(wav1_resample_.shape[0])

            feature_1_ = sample["feature_1.pth"]
            feature_1.append(self.normalize_feature(feature_1_.T))
            feature_2_ = sample["feature_2.pth"]
            feature_2.append(self.normalize_feature(feature_2_.T))
            feature_merged_ = sample["feature_merged.pth"]
            feature_merged.append(self.normalize_feature(feature_merged_.T))

            feature_len.append(feature_1_.shape[-1])

        wav1_resample_padded = pad_sequence(
            [torch.tensor(w) for w in wav1_resample], batch_first=True
        )
        wav2_resample_padded = pad_sequence(
            [torch.tensor(w) for w in wav2_resample], batch_first=True
        )
        wav_merged_resample_padded = pad_sequence(
            [torch.tensor(w) for w in wav_merged_resample], batch_first=True
        )
        feature_1_padded = pad_sequence(feature_1, batch_first=True).transpose(1, 2)
        feature_2_padded = pad_sequence(feature_2, batch_first=True).transpose(1, 2)
        feature_merged_padded = pad_sequence(
            feature_merged, batch_first=True
        ).transpose(1, 2)

        output = {
            "wav_1": wav1_resample_padded,
            "wav_2": wav2_resample_padded,
            "wav_merged": wav_merged_resample_padded,
            "feature_1": feature_1_padded,
            "feature_2": feature_2_padded,
            "feature_merged": feature_merged_padded,
            "wav_len": torch.tensor(wav_len),
            "feature_len": torch.tensor(feature_len),
        }

        return output

    def normalize_feature(self, feature: torch.Tensor) -> torch.Tensor:
        return (feature - self.stats["mean"]) / self.stats["std"]
