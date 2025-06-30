import io
import random
import uuid
from pathlib import Path
from typing import Any

import torch
import torchaudio
import webdataset as wds
from lhotse import CutSet, MultiCut
from lhotse.cut import Cut
from omegaconf import DictConfig
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel


class Preprocessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.dacvae = torch.jit.load(cfg.vae.ckpt_path).to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(cfg.ssl_model.name)
        self.ssl_model = (
            Wav2Vec2BertModel.from_pretrained(
                cfg.ssl_model.name,
            )
            .eval()
            .to(self.device)  # type: ignore
        )

    def write_webdataset(self) -> None:
        shar_dir = Path(self.cfg.shar_dir)
        cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
        recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
        cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

        Path(f"{self.cfg.webdataset_dir}/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.cfg.webdataset_dir}/valid").mkdir(parents=True, exist_ok=True)
        Path(f"{self.cfg.webdataset_dir}/test").mkdir(parents=True, exist_ok=True)
        train_sink = wds.ShardWriter(
            f"{self.cfg.webdataset_dir}/train/data-%06d.tar",
            maxsize=self.cfg.shard_size.train,
        )
        valid_sink = wds.ShardWriter(
            f"{self.cfg.webdataset_dir}/valid/data-%06d.tar",
            maxsize=self.cfg.shard_size.valid,
        )
        test_sink = wds.ShardWriter(
            f"{self.cfg.webdataset_dir}/test/data-%06d.tar",
            maxsize=self.cfg.shard_size.test,
        )

        cuts = cuts.shuffle(random.Random(42))
        for cut in cuts.data:
            sample = self.process_cut(cut)

            assert isinstance(cut, MultiCut)
            assert isinstance(cut.custom, dict)

            if cut.custom["subset"] == "dev-clean":
                valid_sink.write(sample)
            elif cut.custom["subset"] == "test-clean":
                test_sink.write(sample)
            elif cut.custom["subset"] in ["train-clean-100", "train-clean-360"]:
                train_sink.write(sample)

        train_sink.close()
        valid_sink.close()
        test_sink.close()

    def process_cut(self, cut: Cut) -> dict[str, Any]:
        buf = io.BytesIO()
        audio = torch.from_numpy(cut.load_audio())
        torchaudio.save(buf, audio, cut.sampling_rate, format="flac")

        vae_feature_1, vae_feature_2 = self.vae_encode(cut)
        ssl_feature = self.extract_ssl_feature(cut)

        s = {
            "__key__": uuid.uuid1().hex,
            "audio.flac": buf.getvalue(),
            "vae_feature_1.pth": wds.torch_dumps(vae_feature_1.cpu()),
            "vae_feature_2.pth": wds.torch_dumps(vae_feature_2.cpu()),
            "ssl_feature.pth": wds.torch_dumps(ssl_feature.cpu()),
        }

        return s

    def vae_encode(self, cut: Cut) -> tuple[torch.Tensor, torch.Tensor]:
        audio = torch.from_numpy(cut.load_audio())

        if self.cfg.vae.sample_rate != cut.sampling_rate:
            audio = torchaudio.functional.resample(
                audio, cut.sampling_rate, self.cfg.vae.sample_rate
            )

        audio = audio[:, : self.cfg.vae.sample_rate * self.cfg.vae.max_duration]

        wav_input = torch.zeros(
            2,
            1,
            self.cfg.vae.sample_rate * self.cfg.vae.max_duration,
            device=self.device,
        )
        wav_input[0, 0, : audio.shape[-1]] = audio[0]
        wav_input[1, 0, : audio.shape[-1]] = audio[1]

        with torch.no_grad():
            feature, _, _, _ = self.dacvae.encode(wav_input)

        return feature[0], feature[1]

    def extract_ssl_feature(self, cut: Cut) -> torch.Tensor:
        audio = torch.from_numpy(cut.load_audio())

        if self.cfg.ssl_model.sample_rate != cut.sampling_rate:
            audio = torchaudio.functional.resample(
                audio, cut.sampling_rate, self.cfg.ssl_model.sample_rate
            )

        audio = audio[:, : self.cfg.ssl_model.sample_rate * self.cfg.vae.max_duration]

        wav_input = torch.zeros(
            1,
            self.cfg.ssl_model.sample_rate * self.cfg.vae.max_duration,
        )
        wav_input[0, : audio.shape[-1]] = audio[0] + audio[1]

        inputs = self.processor(
            [w.cpu().numpy() for w in wav_input],
            sampling_rate=self.cfg.ssl_model.sample_rate,
            return_tensors="pt",
        )
        inputs["input_features"] = inputs["input_features"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            return self.ssl_model(**inputs, output_hidden_states=True).hidden_states[
                self.cfg.ssl_model.layer
            ]
