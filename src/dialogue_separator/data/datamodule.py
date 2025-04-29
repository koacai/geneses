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

        token_1 = []
        token_2 = []
        token_merged = []
        token_len = []

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

            token_1_ = sample["token_1.pth"]
            token_1.append(token_1_.T)
            token_2_ = sample["token_2.pth"]
            token_2.append(token_2_.T)
            token_merged_ = sample["token_merged.pth"]
            token_merged.append(token_merged_.T)

            token_len.append(token_1_.shape[-1])

        wav1_resample_padded = pad_sequence(
            [torch.tensor(w) for w in wav1_resample], batch_first=True
        )
        wav2_resample_padded = pad_sequence(
            [torch.tensor(w) for w in wav2_resample], batch_first=True
        )
        wav_merged_resample_padded = pad_sequence(
            [torch.tensor(w) for w in wav_merged_resample], batch_first=True
        )
        token_1_padded = pad_sequence(token_1, batch_first=True).transpose(1, 2)
        token_2_padded = pad_sequence(token_2, batch_first=True).transpose(1, 2)
        token_merged_padded = pad_sequence(token_merged, batch_first=True).transpose(
            1, 2
        )

        output = {
            "wav_1": wav1_resample_padded,
            "wav_2": wav2_resample_padded,
            "wav_merged": wav_merged_resample_padded,
            "token_1": token_1_padded,
            "token_2": token_2_padded,
            "token_merged": token_merged_padded,
            "wav_len": torch.tensor(wav_len),
            "token_len": torch.tensor(token_len),
        }

        return output
