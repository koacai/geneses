import pickle
import random
import uuid
from pathlib import Path
from typing import Any

import torch
import torchaudio
import webdataset as wds
from huggingface_hub import hf_hub_download
from lhotse import CutSet
from lhotse.cut import Cut
from omegaconf import DictConfig
from sklearn.cluster import MiniBatchKMeans
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, HubertModel


class Preprocessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.processor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
        self.device = torch.device(cfg.device)
        self.ssl_model = HubertModel.from_pretrained(cfg.model_name).to(self.device)  # type: ignore

        tokenizer_path = hf_hub_download(
            cfg.tokenizer.repo, cfg.tokenizer.filename, token=True
        )
        with open(tokenizer_path, "rb") as f:
            self.tokenizer: MiniBatchKMeans = pickle.load(f)

        self.xvector = EncoderClassifier.from_hparams(
            cfg.xvector.model_name, run_opts={"device": cfg.device}
        )

    def write_webdataset(self) -> None:
        shar_dir = Path(self.cfg.shar_dir)
        cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
        recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
        cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

        Path(f"{self.cfg.webdataset_dir}/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.cfg.webdataset_dir}/valid").mkdir(parents=True, exist_ok=True)
        train_sink = wds.ShardWriter(
            f"{self.cfg.webdataset_dir}/train/data-%06d.tar",
            maxsize=self.cfg.shard_size.train,
        )
        valid_sink = wds.ShardWriter(
            f"{self.cfg.webdataset_dir}/valid/data-%06d.tar",
            maxsize=self.cfg.shard_size.valid,
        )

        cuts = cuts.shuffle(random.Random(42))
        for i, cut in enumerate(cuts.data):
            samples = self.process_cut(cut)
            for sample in samples:
                if i < self.cfg.train_ratio * len(cuts):
                    train_sink.write(sample)
                else:
                    valid_sink.write(sample)

        train_sink.close()
        valid_sink.close()

    def process_cut(self, cut: Cut) -> list[dict[str, Any]]:
        cuts = cut.cut_into_windows(duration=self.cfg.duration)
        res = []
        for c in cuts.data:
            audio, token_1, token_2, token_merged = self.get_audio_tokens(c)
            x_vector_1, x_vector_2 = self.get_xvector(c)

            s = {
                "__key__": uuid.uuid1().hex,
                "resampled_audio.pth": wds.torch_dumps(audio.cpu()),
                "token_1.pth": wds.torch_dumps(token_1.cpu()),
                "token_2.pth": wds.torch_dumps(token_2.cpu()),
                "token_merged.pth": wds.torch_dumps(token_merged.cpu()),
                "x_vector_1.pth": wds.torch_dumps(x_vector_1.cpu()),
                "x_vector_2.pth": wds.torch_dumps(x_vector_2.cpu()),
            }
            res.append(s)

        return res

    def get_audio_tokens(
        self, cut: Cut
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio = torch.from_numpy(cut.load_audio())

        if cut.sampling_rate != self.processor.sampling_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=cut.sampling_rate,
                new_freq=self.processor.sampling_rate,
            )

        audio_1 = audio[0]
        audio_2 = audio[1]
        audio_merged = audio_1 + audio_2

        audio_stack = torch.stack([audio_1, audio_2, audio_merged], dim=0)

        with torch.no_grad():
            input = self.processor(
                audio_stack,
                return_tensors="pt",
                sampling_rate=self.processor.sampling_rate,
            ).input_values
            hidden_states = (
                self.ssl_model(
                    input.to(self.device).squeeze(), output_hidden_states=True
                )
                .hidden_states[self.cfg.layer]
                .squeeze()
            )

        token_1 = self.tokenizer.predict(hidden_states[0].cpu().numpy())
        token_1 = torch.from_numpy(token_1)
        token_2 = self.tokenizer.predict(hidden_states[1].cpu().numpy())
        token_2 = torch.from_numpy(token_2)
        token_merged = self.tokenizer.predict(hidden_states[2].cpu().numpy())
        token_merged = torch.from_numpy(token_merged)

        return audio, token_1, token_2, token_merged

    def get_xvector(self, cut: Cut) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: VADしてからxvector取る必要ある？
        audio = torch.from_numpy(cut.load_audio())
        if cut.sampling_rate != self.cfg.xvector.sr:
            audio = torchaudio.functional.resample(
                audio, orig_freq=cut.sampling_rate, new_freq=self.cfg.xvector.sr
            )

        with torch.no_grad():
            xvector = self.xvector.encode_batch(audio.to(self.device))  # type: ignore

        return xvector.squeeze()[0], xvector.squeeze()[1]
