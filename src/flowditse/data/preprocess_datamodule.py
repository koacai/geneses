import random
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
import webdataset as wds
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor

from flowditse.data.functional_degrations import (
    add_non_parametric_noise,
    band_limit,
    clip,
    codec,
    convolve_rir_pra,
    packet_loss,
    random_apply,
)
from flowditse.data.util import glob_wds


class PreprocessDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(PreprocessDataModule, self).__init__()
        self.cfg = cfg
        self.processor = AutoFeatureExtractor.from_pretrained(cfg.ssl_model.name)

    def setup(self, stage: str | None = None) -> None:
        _ = stage

        noise_dataset = (
            wds.WebDataset(
                glob_wds(self.cfg.noise_dir),
                shardshuffle=False,
                nodesplitter=lambda x: x,
                workersplitter=False,
                repeat=True,
                empty_check=True,
            )
            .decode(wds.torch_audio)
            .compose(
                partial(self.random_crop, n_crops=30, seconds=self.cfg.vae.max_duration)
            )
            .shuffle(10)
            .repeat()
        )
        rir_dataset = (
            wds.WebDataset(
                glob_wds(self.cfg.rir_dir),
                shardshuffle=False,
                nodesplitter=lambda x: x,
                workersplitter=False,
                repeat=True,
                empty_check=True,
            )
            .decode(wds.torch_audio)
            .shuffle(10)
            .repeat()
        )

        self.train_dataset = self.setup_dataset_pipeline(
            wds.WebDataset(
                glob_wds(f"{self.cfg.shard_dir}/train"),
                shardshuffle=100,
                nodesplitter=lambda x: x,
                workersplitter=wds.split_by_worker,
                repeat=True,
            ),
            rir_dataset,
            noise_dataset,
            self.cfg.batch_size,
        )
        self.valid_dataset = self.setup_dataset_pipeline(
            wds.WebDataset(
                glob_wds(f"{self.cfg.shard_dir}/valid"),
                shardshuffle=False,
                nodesplitter=lambda x: x,
                workersplitter=wds.split_by_worker,
                repeat=True,
            ),
            rir_dataset,
            noise_dataset,
            self.cfg.batch_size,
        )

    def setup_dataset_pipeline(
        self,
        dataset: wds.WebDataset,
        rir_dataset: wds.WebDataset,
        noise_dataset: wds.WebDataset,
        batch_size: int,
    ) -> wds.WebDataset:
        dataset = self.init_dataset(dataset)
        dataset = self.add_noise(
            dataset,
            rir_dataset,
            noise_dataset,
        )
        dataset = (
            dataset.map(
                partial(
                    self.normalize, input_key="clean_stereo", output_key="clean_stereo"
                )
            )
            .map(partial(self.normalize, input_key="clean", output_key="clean"))
            .map(partial(self.normalize, input_key="noisy", output_key="noisy"))
        )
        dataset = dataset.batched(batch_size, collation_fn=self.collate_fn)
        return dataset

    def init_dataset(self, dataset: wds.WebDataset) -> wds.WebDataset:
        dataset = (
            dataset.decode(wds.autodecode.basichandlers, wds.torch_audio)
            .map(partial(self.rename_audio, input_key="audio.flac", output_key="audio"))
            .map(partial(self.lowcut, input_key="audio", cutoff=50))
            .map(
                partial(
                    self.padding_by_noise,
                    input_key="audio",
                    wav_len_1_key="wav_len_1.cls",
                    wav_len_2_key="wav_len_2.cls",
                    noise_amp=self.cfg.vae.noise_amp,
                )
            )
            .map(
                partial(
                    self.cut_by_duration,
                    input_key="audio",
                    duration=self.cfg.vae.max_duration,
                )
            )
            .map(partial(self.normalize, input_key="audio", output_key="audio"))
            .map(
                partial(self.rename_audio, input_key="audio", output_key="clean_stereo")
            )
            .map(partial(self.stereo_to_mono, input_key="audio", output_key="noisy"))
            .map(partial(self.stereo_to_mono, input_key="audio", output_key="clean"))
        )
        return dataset

    def add_noise(
        self,
        dataset: wds.WebDataset,
        rir_dataset: wds.WebDataset,
        noise_dataset: wds.WebDataset,
    ) -> wds.WebDataset:
        dataset = (
            dataset.compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=add_non_parametric_noise,
                    input_key="noisy",
                    output_key="noisy",
                    noise_ds=iter(noise_dataset),
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=convolve_rir_pra,
                    input_key="clean",
                    direct_key="clean",
                    reverb_key="noisy",
                    rir_ds=iter(rir_dataset),
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=band_limit,
                    candidate_srs=[8000, 16000, 22050, 24000, 44100, 48000],
                    output_key="noisy",
                    input_key="noisy",
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=clip,
                    input_key="noisy",
                    output_key="noisy",
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=codec,
                    codec_effectors=[
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=10),
                        ),
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=8),
                        ),
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=4),
                        ),
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=2),
                        ),
                        torchaudio.io.AudioEffector(
                            format="mp3",
                            codec_config=torchaudio.io.CodecConfig(qscale=1),
                        ),
                    ],
                    input_key="noisy",
                    output_key="noisy",
                )
            )
            .compose(
                partial(
                    random_apply,
                    prob=0.5,
                    transform_fn=packet_loss,
                    input_key="noisy",
                    output_key="noisy",
                )
            )
        )

        return dataset

    def train_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
            drop_last=True,
        )

    def val_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
            drop_last=True,
        )

    @staticmethod
    def padding_by_noise(
        sample, input_key: str, wav_len_1_key: str, wav_len_2_key: str, noise_amp: float
    ):
        wav, sr = sample[input_key]
        assert wav.shape[0] == 2
        wav_len_1 = sample[wav_len_1_key]
        wav_len_2 = sample[wav_len_2_key]
        # Clone the tensor to avoid in-place modification of inference tensor
        wav = wav.clone()
        if wav_len_1 > wav_len_2:
            wav[1, wav_len_2:] = torch.randn(1, wav_len_1 - wav_len_2) * noise_amp
        else:
            wav[0, wav_len_1:] = torch.randn(1, wav_len_2 - wav_len_1) * noise_amp
        new_sample = sample.copy()
        new_sample[input_key] = (wav, sr)
        return new_sample

    @staticmethod
    def cut_by_duration(sample, input_key: str, duration: int):
        wav, sr = sample[input_key]
        assert wav.shape[0] == 2
        new_sample = sample.copy()
        new_sample[input_key] = (wav[:, : sr * duration], sr)
        return new_sample

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

    @staticmethod
    def random_crop(samples, n_crops, seconds, input_key=None):
        for sample in samples:
            if input_key is None:
                audio_key = [k for k in sample.keys() if "audio" in k][0]
            else:
                audio_key = input_key
            wav, sr = sample[audio_key]
            if wav.shape[0] > 1:
                wav = wav[0, None, :]  # if stereo, take only one channel
            wav = wav.view(1, -1)
            duration = wav.size(1) / sr
            n_crops = int(max(min(duration / n_crops, n_crops), 1))
            for _ in range(n_crops):
                start = random.randint(  # noqa: S311
                    0,
                    max(0, wav.size(1) - int(sr * seconds)),
                )
                cropped = wav[
                    :,
                    start : start + round(sr * seconds),
                ].squeeze(0)
                new_sample = sample.copy()
                new_sample[audio_key] = (cropped, sr)
                yield new_sample

    @staticmethod
    def normalize(sample, input_key: str, output_key: str):
        wav, sr = sample[input_key]
        wav = (wav / wav.abs().max() + 1e-7) * 0.9
        new_sample = sample.copy()
        new_sample[output_key] = (wav, sr)
        return new_sample

    def collate_fn(self, batch) -> dict[str, Any]:
        max_duration = self.cfg.vae.max_duration

        raw_wav_1 = torch.zeros(len(batch), self.cfg.vae.sample_rate * max_duration)
        raw_wav_2 = torch.zeros(len(batch), self.cfg.vae.sample_rate * max_duration)
        clean_wav = torch.zeros(len(batch), self.cfg.vae.sample_rate * max_duration)

        noisy_wav = []
        wav_len = []
        wav_len_1 = []
        wav_len_2 = []
        wav_ssl_input = []
        text_1 = []
        text_2 = []

        for i, sample in enumerate(batch):
            _raw, sr = sample["clean_stereo"]
            if sr != self.cfg.vae.sample_rate:
                _raw = torchaudio.functional.resample(
                    _raw, sr, self.cfg.vae.sample_rate
                )
            raw_wav_1[i, : _raw.shape[-1]] = _raw[0]
            raw_wav_2[i, : _raw.shape[-1]] = _raw[1]
            wav_len.append(_raw.shape[-1])
            wav_len_1.append(
                min(
                    int(sample["wav_len_1.cls"] * self.cfg.vae.sample_rate / sr),
                    self.cfg.vae.sample_rate * max_duration,
                )
            )
            wav_len_2.append(
                min(
                    int(sample["wav_len_2.cls"] * self.cfg.vae.sample_rate / sr),
                    self.cfg.vae.sample_rate * max_duration,
                )
            )

            _clean, sr = sample["clean"]
            _noisy, sr = sample["noisy"]
            if sr != self.cfg.vae.sample_rate:
                _clean = torchaudio.functional.resample(
                    _clean, sr, self.cfg.vae.sample_rate
                )
                _noisy = torchaudio.functional.resample(
                    _noisy, sr, self.cfg.vae.sample_rate
                )
            clean_wav[i, : _clean.shape[-1]] = _clean
            noisy_wav.append(_noisy.squeeze(0))

            if sr != self.cfg.ssl_model.sample_rate:
                _noisy = torchaudio.functional.resample(
                    _noisy, sr, self.cfg.ssl_model.sample_rate
                )

            _wav_ssl_input = F.pad(_noisy, (40, 40), mode="constant", value=0)
            wav_ssl_input.append(_wav_ssl_input)

            if "text_1.txt" in sample:
                text_1.append(sample["text_1.txt"])
            if "text_2.txt" in sample:
                text_2.append(sample["text_2.txt"])

        ssl_input = self.processor(
            [w.cpu().numpy() for w in wav_ssl_input],
            sampling_rate=self.cfg.ssl_model.sample_rate,
            return_tensors="pt",
        )

        output = {
            "raw_wav_1": raw_wav_1,
            "raw_wav_2": raw_wav_2,
            "wav_len": torch.tensor(wav_len),
            "wav_len_1": torch.tensor(wav_len_1),
            "wav_len_2": torch.tensor(wav_len_2),
            "clean_wav": clean_wav,
            "noisy_wav": pad_sequence(noisy_wav, batch_first=True),
            "ssl_input": ssl_input,
        }

        if len(text_1) != 0:
            output["text_1"] = text_1
        if len(text_2) != 0:
            output["text_2"] = text_2

        return output
