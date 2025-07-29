from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
import webdataset as wds
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor


class AugmentDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(AugmentDataModule, self).__init__()
        self.cfg = cfg
        self.processor = AutoFeatureExtractor.from_pretrained(cfg.ssl_model.name)

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

        elif stage == "test":
            self.test_dataset = (
                wds.WebDataset(
                    self.cfg.test.dataset_path,
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
        max_duration = self.cfg.vae.max_duration

        wav_1 = torch.zeros(len(batch), self.cfg.vae.sample_rate * max_duration)
        wav_2 = torch.zeros(len(batch), self.cfg.vae.sample_rate * max_duration)
        wav_merged = torch.zeros(len(batch), self.cfg.vae.sample_rate * max_duration)

        wav_len = []
        vae_feature_1 = []
        vae_feature_2 = []
        vae_len = []
        text_1 = []
        text_2 = []
        wav_ssl_input = []

        for i, sample in enumerate(batch):
            dialogue, sr = sample["audio.flac"]

            if sr != self.cfg.vae.sample_rate:
                dialogue = torchaudio.functional.resample(
                    dialogue, sr, self.cfg.vae.sample_rate
                )

            dialogue = dialogue[:, : self.cfg.vae.sample_rate * max_duration]
            wav_1[i, : dialogue.shape[-1]] = dialogue[0]
            wav_2[i, : dialogue.shape[-1]] = dialogue[1]
            wav_merged[i, : dialogue.shape[-1]] = dialogue[0] + dialogue[1]

            _wav_len = dialogue.shape[-1]
            wav_len.append(_wav_len)

            vae_feature_1.append(sample["vae_feature_1.pth"])
            vae_feature_2.append(sample["vae_feature_2.pth"])

            _vae_len = (
                sample["vae_feature_1.pth"].shape[-1] * _wav_len // wav_1.shape[-1]
            )
            vae_len.append(_vae_len)

            text_1.append(sample["text_1.txt"])
            text_2.append(sample["text_2.txt"])

            if sr != self.cfg.ssl_model.sample_rate:
                dialogue = torchaudio.functional.resample(
                    dialogue, sr, self.cfg.ssl_model.sample_rate
                )

            _wav_ssl_input = dialogue[0] + dialogue[1]
            _wav_ssl_input = F.pad(_wav_ssl_input, (40, 40), mode="constant", value=0)
            wav_ssl_input.append(_wav_ssl_input)

        vae_feature_1 = pad_sequence(vae_feature_1, batch_first=True)
        vae_feature_2 = pad_sequence(vae_feature_2, batch_first=True)

        ssl_input = self.processor(
            [w.cpu().numpy() for w in wav_ssl_input],
            sampling_rate=self.cfg.ssl_model.sample_rate,
            return_tensors="pt",
        )

        output = {
            "wav_1": wav_1,
            "wav_2": wav_2,
            "wav_merged": wav_merged,
            "wav_len": torch.tensor(wav_len),
            "vae_len": torch.tensor(vae_len),
            "vae_feature_1": vae_feature_1,
            "vae_feature_2": vae_feature_2,
            "ssl_input": ssl_input,
            "text_1": text_1,
            "text_2": text_2,
        }

        return output
