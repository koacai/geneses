from typing import Any

import torch
import torchaudio
import webdataset as wds
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence


class HuBERTSeparatorDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(HuBERTSeparatorDataModule, self).__init__()
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
            pin_memory=False,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
        )

    def val_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
        )

    def identity(self, x):
        return x[0]

    def collate_fn(self, batch) -> dict[str, Any]:
        max_duration = self.cfg.max_duration

        wav1_22050s = []
        wav2_22050s = []
        wav_merged_22050s = []
        wav_lens = []

        token_1s = []
        token_2s = []
        token_mergeds = []

        for sample in batch:
            dialogue = sample["resampled_audio.pth"]
            sr = 16000

            dialogue_22050 = torchaudio.functional.resample(dialogue, sr, 22050)
            wav1_22050 = dialogue_22050[0, : 22050 * max_duration].numpy()
            wav2_22050 = dialogue_22050[1, : 22050 * max_duration].numpy()
            wav1_22050s.append(wav1_22050)
            wav2_22050s.append(wav2_22050)
            wav_merged_22050s.append(wav1_22050 + wav2_22050)

            wav_lens.append(wav1_22050s[-1].shape[0])

            token_1 = sample["token_1.pth"]
            token_1s.append(token_1)
            token_2 = sample["token_2.pth"]
            token_2s.append(token_2)
            token_merged = sample["token_merged.pth"]
            token_mergeds.append(token_merged)

        wav1_22050s_padded = pad_sequence(
            [torch.tensor(w) for w in wav1_22050s], batch_first=True
        )
        wav2_22050s_padded = pad_sequence(
            [torch.tensor(w) for w in wav2_22050s], batch_first=True
        )
        wav_merged_22050s_padded = pad_sequence(
            [torch.tensor(w) for w in wav_merged_22050s], batch_first=True
        )
        token_1s_padded = pad_sequence(
            [torch.tensor(t) for t in token_1s], batch_first=True
        )
        token_2s_padded = pad_sequence(
            [torch.tensor(t) for t in token_2s], batch_first=True
        )
        token_mergeds_padded = pad_sequence(
            [torch.tensor(t) for t in token_mergeds], batch_first=True
        )

        output = {
            "wav_1": wav1_22050s_padded,
            "wav_2": wav2_22050s_padded,
            "wav_merged": wav_merged_22050s_padded,
            "token_1": token_1s_padded,
            "token_2": token_2s_padded,
            "token_merged": token_mergeds_padded,
            "wav_len": torch.tensor(wav_lens),
        }

        return output
