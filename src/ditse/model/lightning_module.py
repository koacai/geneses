from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torchaudio
import utmosv2
import wandb
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from omegaconf import DictConfig
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio.nisqa import (
    NonIntrusiveSpeechQualityAssessment,
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import SignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from utmosv2._core.create import UTMOSv2Model

from ditse.metrics.lsd import lsd_metric
from ditse.metrics.mcd import mcd_metric
from ditse.metrics.speech_bert_score import (
    SpeechBERTScore,
    speech_bert_score_metric,
)
from ditse.model.components import MMDiT
from ditse.model.dacvae import DACVAE
from ditse.model.ssl_feature_extractor import SSLFeatureExtractor
from ditse.util.util import create_mask


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        ssl_merged = extras.get("ssl_merged", None)
        assert ssl_merged is not None
        vae_1 = x[:, 0, :, :]
        vae_2 = x[:, 1, :, :]
        res_1, res_2 = self.model.forward(ssl_merged, t.unsqueeze(0), vae_1, vae_2)
        return torch.stack([res_1, res_2], dim=1)


class DiTSELightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(DiTSELightningModule, self).__init__()
        self.cfg = cfg

        self.mmdit = MMDiT(**cfg.model.mmdit)
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.dacvae = DACVAE(cfg.model.vae.ckpt_path)
        self.ssl_feature_extractor = SSLFeatureExtractor(
            cfg.model.ssl_model.name,
            cfg.model.ssl_model.layer,
            cfg.model.ssl_model.fine_tuning_mode,
        )

        self.save_hyperparameters(cfg)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = hydra.utils.instantiate(
            self.cfg.model.optimizer, params=self.parameters()
        )
        lr_scheduler = hydra.utils.instantiate(
            self.cfg.model.lr_scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def on_fit_start(self) -> None:
        self.dacvae.to(self.device)

    def on_test_start(self) -> None:
        self.dacvae.to(self.device)

    def calc_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            vae_1 = self.dacvae.encode(batch["raw_wav_1"])
            vae_2 = self.dacvae.encode(batch["raw_wav_2"])
            ssl_merged = self.ssl_feature_extractor.forward(batch["ssl_input"])

        batch_size = ssl_merged.size(0)

        vae_len = [
            vae_1.shape[-1] * wl // batch["raw_wav_1"].shape[-1]
            for wl in batch["wav_len"]
        ]
        mask = create_mask(torch.tensor(vae_len), vae_1)

        t = self.sampling_t(batch_size)
        vae = torch.stack([vae_1, vae_2], dim=1)
        noise = torch.randn_like(vae, device=self.device)
        path_sample = self.path.sample(x_0=noise, x_1=vae, t=t)

        x_t_1 = path_sample.x_t[:, 0, :, :] * mask
        x_t_2 = path_sample.x_t[:, 1, :, :] * mask

        est_dxt_1, est_dxt_2 = self.mmdit.forward(
            ssl_merged, t, x_t_1.permute(0, 2, 1), x_t_2.permute(0, 2, 1)
        )

        loss = self.loss_fn(
            est_dxt_1.permute(0, 2, 1) * mask,
            est_dxt_2.permute(0, 2, 1) * mask,
            path_sample.dx_t[:, 0, :, :] * mask,
            path_sample.dx_t[:, 1, :, :] * mask,
        )

        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        _ = batch_idx

        loss = self.calc_loss(batch)

        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        _ = batch_idx

        loss = self.calc_loss(batch)

        self.log("validation_loss", loss, sync_dist=True)

        wav_sr = self.cfg.model.vae.sample_rate
        if batch_idx < 5 and self.global_rank == 0 and self.local_rank == 0:
            wav_len = batch["wav_len"][0]
            wav_1 = batch["raw_wav_1"][0][:wav_len].cpu().numpy()
            wav_2 = batch["raw_wav_2"][0][:wav_len].cpu().numpy()
            clean_wav = batch["clean_wav"][0][:wav_len].cpu().numpy()
            noisy_wav = batch["noisy_wav"][0][:wav_len].cpu().numpy()

            self.log_audio(wav_1, f"wav_1/{batch_idx}", wav_sr)
            self.log_audio(wav_2, f"wav_2/{batch_idx}", wav_sr)
            self.log_audio(clean_wav, f"clean_wav/{batch_idx}", wav_sr)
            self.log_audio(noisy_wav, f"noisy_wav/{batch_idx}", wav_sr)

            est_feature1, est_feature2 = self.forward(batch)

            with torch.no_grad():
                vae_1 = self.dacvae.encode(batch["raw_wav_1"])
                vae_2 = self.dacvae.encode(batch["raw_wav_2"])
                est_feature1, est_feature2 = self.change_permutation(
                    est_feature1, est_feature2, vae_1, vae_2
                )

                decoded_1 = (
                    self.dacvae.decode(vae_1)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                decoded_2 = (
                    self.dacvae.decode(vae_2)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

                estimated_1 = (
                    self.dacvae.decode(est_feature1)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                estimated_2 = (
                    self.dacvae.decode(est_feature2)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

            self.log_audio(decoded_1, f"decoded_1/{batch_idx}", wav_sr)
            self.log_audio(decoded_2, f"decoded_2/{batch_idx}", wav_sr)
            self.log_audio(estimated_1, f"estimated_1/{batch_idx}", wav_sr)
            self.log_audio(estimated_2, f"estimated_2/{batch_idx}", wav_sr)

        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        with torch.no_grad():
            vae_1 = self.dacvae.encode(batch["raw_wav_1"])
            vae_2 = self.dacvae.encode(batch["raw_wav_2"])

            est_feature1, est_feature2 = self.forward(batch, step_size=0.01)
            est_feature1, est_feature2 = self.change_permutation(
                est_feature1, est_feature2, vae_1, vae_2
            )

            decoded_1_all = self.dacvae.decode(vae_1)
            decoded_2_all = self.dacvae.decode(vae_2)
            estimated_1_all = self.dacvae.decode(est_feature1)
            estimated_2_all = self.dacvae.decode(est_feature2)

        wav_sr = self.cfg.model.vae.sample_rate
        dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=wav_sr, personalized=False)
        nisqa = NonIntrusiveSpeechQualityAssessment(fs=wav_sr)
        utmos = utmosv2.create_model(pretrained=True, device=self.device)
        pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
        estoi = ShortTimeObjectiveIntelligibility(fs=wav_sr, extended=True)
        sdr = SignalDistortionRatio().to(device=self.device)
        speech_bert_score = SpeechBERTScore(self.device)
        speech_bert_score.speech_bert_score.model.eval()

        batch_size = batch["raw_wav_1"].size(0)
        for i in range(batch_size):
            sample_dir = Path("test_output") / f"{batch_idx}" / f"{i}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir = sample_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            wav_len = batch["wav_len"][i]
            wav_len_1 = batch["wav_len_1"][i]
            wav_len_2 = batch["wav_len_2"][i]
            wav_1 = batch["raw_wav_1"][i][:wav_len_1]
            wav_2 = batch["raw_wav_2"][i][:wav_len_2]
            clean_wav = batch["clean_wav"][i][:wav_len]
            noisy_wav = batch["noisy_wav"][i][:wav_len]

            torchaudio.save(sample_dir / "wav_1.wav", wav_1.cpu().unsqueeze(0), wav_sr)
            torchaudio.save(sample_dir / "wav_2.wav", wav_2.cpu().unsqueeze(0), wav_sr)
            torchaudio.save(
                sample_dir / "clean_wav.wav", clean_wav.cpu().unsqueeze(0), wav_sr
            )
            torchaudio.save(
                sample_dir / "noisy_wav.wav", noisy_wav.cpu().unsqueeze(0), wav_sr
            )

            decoded_1 = decoded_1_all[i].squeeze()[:wav_len_1].to(torch.float32)
            decoded_2 = decoded_2_all[i].squeeze()[:wav_len_2].to(torch.float32)
            torchaudio.save(
                sample_dir / "decoded_1.wav", decoded_1.cpu().unsqueeze(0), wav_sr
            )
            torchaudio.save(
                sample_dir / "decoded_2.wav", decoded_2.cpu().unsqueeze(0), wav_sr
            )

            estimated_1 = estimated_1_all[i].squeeze()[:wav_len_1].to(torch.float32)
            estimated_2 = estimated_2_all[i].squeeze()[:wav_len_2].to(torch.float32)

            torchaudio.save(
                sample_dir / "estimated_1.wav", estimated_1.cpu().unsqueeze(0), wav_sr
            )
            torchaudio.save(
                sample_dir / "estimated_2.wav", estimated_2.cpu().unsqueeze(0), wav_sr
            )

            df_without_ref, df_with_ref = self.evaluation_metrics(
                dnsmos,
                nisqa,
                utmos,
                pesq,
                estoi,
                sdr,
                speech_bert_score,
                wav_1,
                wav_2,
                decoded_1,
                decoded_2,
                estimated_1,
                estimated_2,
                wav_sr,
                sample_dir,
            )
            df_without_ref.to_csv(metrics_dir / "without_ref.csv", index=False)
            df_with_ref.to_csv(metrics_dir / "with_ref.csv", index=False)

    @staticmethod
    def evaluation_metrics(
        dnsmos: DeepNoiseSuppressionMeanOpinionScore,
        nisqa: NonIntrusiveSpeechQualityAssessment,
        utmos: UTMOSv2Model,
        pesq: PerceptualEvaluationSpeechQuality,
        estoi: ShortTimeObjectiveIntelligibility,
        sdr: SignalDistortionRatio,
        speech_bert_score: SpeechBERTScore,
        wav_1: torch.Tensor,
        wav_2: torch.Tensor,
        decoded_1: torch.Tensor,
        decoded_2: torch.Tensor,
        estimated_1: torch.Tensor,
        estimated_2: torch.Tensor,
        wav_sr: int,
        sample_dir: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        wav_dict = {
            "wav_1": (wav_1, sample_dir / "wav_1.wav"),
            "wav_2": (wav_2, sample_dir / "wav_2.wav"),
            "decoded_1": (decoded_1, sample_dir / "decoded_1.wav"),
            "decoded_2": (decoded_2, sample_dir / "decoded_2.wav"),
            "estimated_1": (estimated_1, sample_dir / "estimated_1.wav"),
            "estimated_2": (estimated_2, sample_dir / "estimated_2.wav"),
        }

        without_ref = []
        for name, (wav, path) in wav_dict.items():
            _dnsmos = dnsmos(wav)[-1].item()
            _nisqa = nisqa(wav)[0].item()
            with torch.no_grad():
                _utmos = utmos.predict(input_path=path)
            without_ref.append(
                dict(key=name, dnsmos=_dnsmos, nisqa=_nisqa, utmos=_utmos)
            )
        df_without_ref = pd.DataFrame(without_ref)

        wav_pair_dict = {
            "wav_decoded_1": (wav_1, decoded_1),
            "wav_decoded_2": (wav_2, decoded_2),
            "wav_estimated_1": (wav_1, estimated_1),
            "wav_estimated_2": (wav_2, estimated_2),
        }

        with_ref = []
        for name, (ref, inf) in wav_pair_dict.items():
            # NOTE: これをやる必要があるのは、max_durationいっぱいいっぱいの音声のみ（VAEの都合）
            if ref.shape[-1] > inf.shape[-1]:
                ref = ref[: inf.shape[-1]]
            elif inf.shape[-1] > ref.shape[-1]:
                inf = inf[: ref.shape[-1]]

            if wav_sr != 16000:
                ref_resample = torchaudio.functional.resample(ref, wav_sr, 16000)
                inf_resample = torchaudio.functional.resample(inf, wav_sr, 16000)
            else:
                ref_resample = ref
                inf_resample = inf

            _pesq = pesq(ref_resample, inf_resample).item()
            _estoi = estoi(ref, inf).item()
            _sdr = sdr(ref, inf).item()
            _mcd = mcd_metric(ref, inf, wav_sr)
            _lsd = lsd_metric(ref, inf, wav_sr)
            _sbs = speech_bert_score_metric(speech_bert_score, ref, inf, wav_sr)
            with_ref.append(
                dict(
                    key=name,
                    pesq=_pesq,
                    estoi=_estoi,
                    sdr=_sdr,
                    mcd=_mcd,
                    lsd=_lsd,
                    speech_bert_score=_sbs,
                )
            )
        df_with_ref = pd.DataFrame(with_ref)

        return df_without_ref, df_with_ref

    def change_permutation(
        self,
        est_feature1: torch.Tensor,
        est_feature2: torch.Tensor,
        src_feature1: torch.Tensor,
        src_feature2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mse_loss = torch.nn.MSELoss()

        batch_size = est_feature1.size(0)

        new_est_feature1 = torch.zeros_like(est_feature1)
        new_est_feature2 = torch.zeros_like(est_feature2)

        for i in range(batch_size):
            src = torch.stack([src_feature1[i], src_feature2[i]], dim=1)
            est_p1 = torch.stack([est_feature1[i], est_feature2[i]], dim=1)
            est_p2 = torch.stack([est_feature2[i], est_feature1[i]], dim=1)

            loss1 = mse_loss(est_p1, src)
            loss2 = mse_loss(est_p2, src)

            if loss1 < loss2:
                new_est_feature1[i] = est_feature1[i]
                new_est_feature2[i] = est_feature2[i]
            else:
                new_est_feature1[i] = est_feature2[i]
                new_est_feature2[i] = est_feature1[i]

        return new_est_feature1, new_est_feature2

    def sampling_t(
        self, batch_size: int, m: float = 0.0, s: float = 1.0
    ) -> torch.Tensor:
        schema = self.cfg.model.t_sampling_schema

        if schema == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif schema == "logit_normal":
            u = torch.randn((batch_size,), device=self.device) * s + m
            t = torch.sigmoid(u)
        else:
            raise ValueError(f"Unknown t sampling schema: {schema}")

        return t

    def loss_fn(
        self,
        est_dxt1: torch.Tensor,
        est_dxt2: torch.Tensor,
        dxt_1: torch.Tensor,
        dxt_2: torch.Tensor,
    ) -> torch.Tensor:
        mse_loss = torch.nn.MSELoss()

        est = torch.stack([est_dxt1, est_dxt2], dim=1)
        src = torch.stack([dxt_1, dxt_2], dim=1)

        loss = mse_loss(est, src)

        return loss

    def forward(
        self, batch: dict[str, Any], step_size: float = 0.01
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            ssl_merged = self.ssl_feature_extractor.forward(batch["ssl_input"])

        vae_size = (
            ssl_merged.size(0),
            333,  # 20秒の音声のVAEは長さ333（ハードコーディング）
            self.cfg.model.vae.hidden_size,
        )

        noise_1 = torch.randn(vae_size, device=self.device)
        noise_2 = torch.randn(vae_size, device=self.device)
        noise = torch.stack([noise_1, noise_2], dim=1)

        time_grid = torch.tensor([0.0, 1.0])

        solver = ODESolver(velocity_model=WrappedModel(self.mmdit))

        res = solver.sample(
            x_init=noise,
            step_size=step_size,
            time_grid=time_grid,
            ssl_merged=ssl_merged,
        )
        assert isinstance(res, torch.Tensor)

        vae_1 = res[:, 0, :, :].permute(0, 2, 1)
        vae_2 = res[:, 1, :, :].permute(0, 2, 1)

        return vae_1, vae_2

    def log_audio(self, audio: np.ndarray, name: str, sampling_rate: int) -> None:
        if isinstance(self.logger, loggers.WandbLogger):
            wandb.log({name: wandb.Audio(audio, sample_rate=sampling_rate)})
