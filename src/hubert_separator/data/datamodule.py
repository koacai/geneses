from typing import Any

import torch
import torchaudio
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor

import webdataset as wds


class HuBERTSeparatorDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(HuBERTSeparatorDataModule, self).__init__()
        self.cfg = cfg
        self.processor = AutoFeatureExtractor.from_pretrained(cfg.hubert_model_name)

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
        sr_ssl = self.processor.sampling_rate
        assert sr_ssl == 16000, "Sampling rate of HuBERT must be 16000"

        max_duration = self.cfg.max_duration

        wav1_16000s = []
        wav2_16000s = []
        wav_merged_16000s = []
        wav1_22050s = []
        wav2_22050s = []
        wav_merged_22050s = []
        wav_lens = []

        for sample in batch:
            dialogue, sr = sample["audio.flac"]

            dialogue_16000 = torchaudio.functional.resample(dialogue, sr, 16000)
            wav1_16000 = dialogue_16000[0, : 16000 * max_duration].numpy()
            wav2_16000 = dialogue_16000[1, : 16000 * max_duration].numpy()
            wav1_16000s.append(wav1_16000)
            wav2_16000s.append(wav2_16000)
            wav_merged_16000s.append(wav1_16000 + wav2_16000)

            dialogue_22050 = torchaudio.functional.resample(dialogue, sr, 22050)
            wav1_22050 = dialogue_22050[0, : 22050 * max_duration].numpy()
            wav2_22050 = dialogue_22050[1, : 22050 * max_duration].numpy()
            wav1_22050s.append(wav1_22050)
            wav2_22050s.append(wav2_22050)
            wav_merged_22050s.append(wav1_22050 + wav2_22050)

            wav_lens.append(wav1_22050s[-1].shape[0])

        ssl_input_1 = self.processor(
            wav1_16000s,
            return_tensors="pt",
            sampling_rate=sr_ssl,
            padding=True,
        )
        ssl_input_2 = self.processor(
            wav2_16000s,
            return_tensors="pt",
            sampling_rate=sr_ssl,
            padding=True,
        )
        ssl_input_merged = self.processor(
            wav_merged_16000s,
            return_tensors="pt",
            sampling_rate=sr_ssl,
            padding=True,
        )
        wav1_22050s_padded = pad_sequence(
            [torch.tensor(w) for w in wav1_22050s], batch_first=True
        )
        wav2_22050s_padded = pad_sequence(
            [torch.tensor(w) for w in wav2_22050s], batch_first=True
        )
        wav_merged_22050s_padded = pad_sequence(
            [torch.tensor(w) for w in wav_merged_22050s], batch_first=True
        )

        output = {
            "wav_1": wav1_22050s_padded,
            "wav_2": wav2_22050s_padded,
            "wav_merged": wav_merged_22050s_padded,
            "ssl_input_1": ssl_input_1,
            "ssl_input_2": ssl_input_2,
            "ssl_input_merged": ssl_input_merged,
            "wav_len": torch.tensor(wav_lens),
        }

        return output
