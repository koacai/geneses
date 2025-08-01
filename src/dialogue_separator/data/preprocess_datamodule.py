import random
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torchaudio
import webdataset as wds
from lhotse import CutSet, MultiCut
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dialogue_separator.data.dataset import LibriTTSRMixDataset
from dialogue_separator.data.functional_degrations import (
    add_non_parametric_noise,
    band_limit,
    clip,
    codec,
    convolve_rir_pra,
    packet_loss,
    random_apply,
)
from dialogue_separator.data.util import glob_wds


class PreprocessDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(PreprocessDataModule, self).__init__()
        self.cfg = cfg

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
            LibriTTSRMixDataset(train_cuts),
            rir_dataset,
            noise_dataset,
            self.cfg.batch_size,
        )
        self.valid_dataset = self.setup_dataset_pipeline(
            LibriTTSRMixDataset(valid_cuts),
            rir_dataset,
            noise_dataset,
            self.cfg.batch_size,
        )
        self.test_dataset = self.setup_dataset_pipeline(
            LibriTTSRMixDataset(test_cuts),
            rir_dataset,
            noise_dataset,
            self.cfg.batch_size,
        )

    def setup_dataset_pipeline(
        self,
        dataset: LibriTTSRMixDataset,
        rir_dataset: wds.WebDataset,
        noise_dataset: wds.WebDataset,
        batch_size: int,
    ) -> LibriTTSRMixDataset:
        dataset = self.init_dataset(dataset)
        for _ in range(self.cfg.noise_pipeline_times):
            dataset = self.add_noise(dataset, rir_dataset, noise_dataset)
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

    def init_dataset(self, dataset: LibriTTSRMixDataset) -> LibriTTSRMixDataset:
        dataset = (
            dataset.map(partial(self.lowcut, input_key="audio", cutoff=50))
            .map(
                partial(
                    self.padding_by_noise,
                    input_key="audio",
                    wav_len_1_key="wav_len_1",
                    wav_len_2_key="wav_len_2",
                    noise_amp=self.cfg.vae.noise_amp,
                )
            )
            .map(
                partial(
                    self.align_duration,
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
        dataset: LibriTTSRMixDataset,
        rir_dataset: wds.WebDataset,
        noise_dataset: wds.WebDataset,
    ) -> LibriTTSRMixDataset:
        dataset = (
            dataset.compose(
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda x: x[0],
            drop_last=True,
            persistent_workers=True,
        )

    @staticmethod
    def padding_by_noise(
        sample, input_key: str, wav_len_1_key: str, wav_len_2_key: str, noise_amp: float
    ):
        wav, sr = sample[input_key]
        assert wav.shape[0] == 2
        wav_len_1 = sample[wav_len_1_key]
        wav_len_2 = sample[wav_len_2_key]
        if wav_len_1 > wav_len_2:
            wav[1, wav_len_2:] = torch.randn(1, wav_len_1 - wav_len_2) * noise_amp
        else:
            wav[0, wav_len_1:] = torch.randn(1, wav_len_2 - wav_len_1) * noise_amp
        new_sample = sample.copy()
        new_sample[input_key] = (wav, sr)
        return new_sample

    @staticmethod
    def align_duration(sample, input_key: str, duration: int):
        wav, sr = sample[input_key]
        assert wav.shape[0] == 2
        new_wav = torch.zeros((wav.shape[0], sr * duration))
        new_wav[:, : wav.shape[1]] = wav
        new_sample = sample.copy()
        new_sample[input_key] = (new_wav, sr)
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
        raw_wav_1 = []
        raw_wav_2 = []
        clean = []
        noisy = []
        text_1 = []
        text_2 = []

        for sample in batch:
            _raw, sr = sample["clean_stereo"]
            if sr != self.cfg.vae.sample_rate:
                _raw = torchaudio.functional.resample(
                    _raw, sr, self.cfg.vae.sample_rate
                )
            raw_wav_1.append(_raw[0])
            raw_wav_2.append(_raw[1])

            _clean, sr = sample["clean"]
            if sr != self.cfg.vae.sample_rate:
                _clean = torchaudio.functional.resample(
                    _clean, sr, self.cfg.vae.sample_rate
                )
            clean.append(_clean.squeeze(0))

            _noisy, sr = sample["noisy"]
            if sr != self.cfg.ssl_model.sample_rate:
                _noisy = torchaudio.functional.resample(
                    _noisy, sr, self.cfg.vae.sample_rate
                )
            noisy.append(_noisy.squeeze(0))

            text_1.append(sample["text_1"])
            text_2.append(sample["text_2"])

        output = {
            "raw_wav_1": torch.stack(raw_wav_1),
            "raw_wav_2": torch.stack(raw_wav_2),
            "clean_wav": torch.stack(clean),
            "noisy_wav": torch.stack(noisy),
            "text_1": text_1,
            "text_2": text_2,
        }

        return output
