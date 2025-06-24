from typing import Any

import torch
import torchaudio
import webdataset as wds
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor


class DialogueSeparatorDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(DialogueSeparatorDataModule, self).__init__()
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

    def test_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.test_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
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

        for i, sample in enumerate(batch):
            dialogue, sr = sample["audio.flac"]

            dialogue_resample = torchaudio.functional.resample(
                dialogue, sr, self.cfg.vae.sample_rate
            )
            dialogue_resample = dialogue_resample[
                :, : self.cfg.vae.sample_rate * max_duration
            ]
            wav_1[i, : dialogue_resample.shape[-1]] = dialogue_resample[0]
            wav_2[i, : dialogue_resample.shape[-1]] = dialogue_resample[1]
            wav_merged[i, : dialogue_resample.shape[-1]] = (
                dialogue_resample[0] + dialogue_resample[1]
            )

            wav_len.append(dialogue_resample.shape[-1])

            vae_feature_1.append(sample["vae_feature_1.pth"])
            vae_feature_2.append(sample["vae_feature_2.pth"])

        wav_merged_ssl = torchaudio.functional.resample(
            wav_merged, self.cfg.vae.sample_rate, self.cfg.ssl_model.sample_rate
        )

        ssl_input_merged = self.processor(
            [w for w in wav_merged_ssl],
            sampling_rate=self.cfg.ssl_model.sample_rate,
            return_tensors="pt",
        )

        vae_feature_1 = pad_sequence(vae_feature_1, batch_first=True)
        vae_feature_2 = pad_sequence(vae_feature_2, batch_first=True)

        output = {
            "wav_1": wav_1,
            "wav_2": wav_2,
            "wav_merged": wav_merged,
            "wav_len": torch.tensor(wav_len),
            "ssl_input_merged": ssl_input_merged,
            "vae_feature_1": vae_feature_1,
            "vae_feature_2": vae_feature_2,
        }

        return output
