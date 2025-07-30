from functools import partial

import torch
import torchaudio
import webdataset as wds
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from transformers import AutoFeatureExtractor


class AugmentDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(AugmentDataModule, self).__init__()
        self.cfg = cfg
        self.processor = AutoFeatureExtractor.from_pretrained(cfg.ssl_model.name)

    def setup(self, stage: str | None = None) -> None:
        _ = stage

        self.train_dataset = self.setup_dataset_pipeline(
            wds.WebDataset(
                self.cfg.train.dataset_path,
                shardshuffle=100,
                nodesplitter=lambda x: x,
                repeat=True,
            )
        )
        self.valid_dataset = self.setup_dataset_pipeline(
            wds.WebDataset(
                self.cfg.valid.dataset_path,
                shardshuffle=False,
                nodesplitter=lambda x: x,
                repeat=True,
            )
        )
        self.test_dataset = self.setup_dataset_pipeline(
            wds.WebDataset(
                self.cfg.test.dataset_path,
                shardshuffle=False,
                nodesplitter=lambda x: x,
                repeat=True,
            )
        )

    def setup_dataset_pipeline(self, dataset: wds.WebDataset) -> wds.WebDataset:
        dataset = (
            dataset.decode(wds.autodecode.basichandlers, wds.torch_audio)
            .map(partial(self.lowcut, input_key="audio.flac", cutoff=50))
            .map(partial(self.rename_audio, input_key="audio.flac", output_key="clean"))
            .map(partial(self.rename_audio, input_key="audio.flac", output_key="noisy"))
        )
        return dataset

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

    @staticmethod
    def rename_audio(sample, output_key: str, input_key: str | None = None):
        if input_key is None:
            audio_key = [k for k in sample.keys() if "audio" in k][0]
        else:
            audio_key = input_key
        sample[output_key] = sample[audio_key]
        return sample

    @staticmethod
    @torch.inference_mode()
    def lowcut(sample, input_key: str, cutoff=50):
        wav, sr = sample[input_key]
        wav = torchaudio.functional.highpass_biquad(wav, sr, cutoff)
        new_sample = sample.copy()
        new_sample[input_key] = (wav.view(1, -1), sr)
        return new_sample
