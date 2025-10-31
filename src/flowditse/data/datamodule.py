from collections import defaultdict
from typing import Any

import torch
import webdataset as wds
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from flowditse.data.util import glob_wds


class GenesesDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(GenesesDataModule, self).__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        nodesplitter = wds.split_by_worker if self.cfg.use_ddp else wds.single_node_only
        if stage == "fit":
            self.train_dataset = (
                wds.WebDataset(
                    glob_wds(self.cfg.train.dataset_dir),
                    shardshuffle=100,
                    nodesplitter=nodesplitter,
                    repeat=True,
                )
                .decode(wds.torch_audio)
                .batched(self.cfg.train.batch_size, collation_fn=self.collate_fn)
            )

            self.valid_dataset = (
                wds.WebDataset(
                    glob_wds(self.cfg.valid.dataset_dir),
                    shardshuffle=False,
                    nodesplitter=nodesplitter,
                    repeat=True,
                )
                .decode(wds.torch_audio)
                .batched(self.cfg.valid.batch_size, collation_fn=self.collate_fn)
            )

        elif stage == "test":
            self.test_dataset = (
                wds.WebDataset(
                    glob_wds(self.cfg.test.dataset_dir),
                    shardshuffle=False,
                    nodesplitter=wds.single_node_only,
                    repeat=True,
                )
                .decode(wds.torch_audio)
                .batched(self.cfg.test.batch_size, collation_fn=self.collate_fn)
            )

    def train_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
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

    def collate_fn(self, batch) -> dict[str, Any]:
        raw_wav_1 = []
        raw_wav_2 = []
        clean_wav = []
        noisy_wav = []
        wav_len = []
        wav_len_1 = []
        wav_len_2 = []
        text_1 = []
        text_2 = []
        ssl_input = defaultdict(list)

        for sample in batch:
            raw_wav_1.append(sample["raw_wav_1.pth"].squeeze())
            raw_wav_2.append(sample["raw_wav_2.pth"].squeeze())
            clean_wav.append(sample["clean_wav.pth"].squeeze())
            noisy_wav.append(sample["noisy_wav.pth"].squeeze())
            wav_len.append(sample["wav_len.pth"][0])
            wav_len_1.append(sample["wav_len_1.pth"][0])
            wav_len_2.append(sample["wav_len_2.pth"][0])
            if "text_1.pickle" in sample:
                text_1.append(sample["text_1.pickle"][0])
            if "text_2.pickle" in sample:
                text_2.append(sample["text_2.pickle"][0])

            for k, v in sample["ssl_input.pickle"].items():
                ssl_input[k].append(v.squeeze())

        output = {
            "raw_wav_1": torch.stack(raw_wav_1),
            "raw_wav_2": torch.stack(raw_wav_2),
            "clean_wav": torch.stack(clean_wav),
            "noisy_wav": pad_sequence(noisy_wav, batch_first=True),
            "wav_len": torch.stack(wav_len),
            "wav_len_1": torch.stack(wav_len_1),
            "wav_len_2": torch.stack(wav_len_2),
            "ssl_input": {
                k: pad_sequence(v, batch_first=True) for k, v in ssl_input.items()
            },
        }
        if len(text_1) != 0:
            output["text_1"] = text_1
        if len(text_2) != 0:
            output["text_2"] = text_2

        return output
