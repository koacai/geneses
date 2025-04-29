import hydra
import numpy as np
import torch
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from huggingface_hub import hf_hub_download
from lightning.pytorch import LightningModule, loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from moshi.models import loaders
from omegaconf import DictConfig

import wandb
from dialogue_separator.utils.model import sequence_mask

from .flow_predictor import Decoder


class WrappedDecoder(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        mask = extras.get("mask", None)
        assert mask is not None
        x_merged = extras.get("x_merged", None)
        assert x_merged is not None
        return torch.softmax(self.model.forward(x, mask, x_merged, t), dim=-1)


class DialogueSeparatorLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(DialogueSeparatorLightningModule, self).__init__()
        self.cfg = cfg
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(cfg.model.mimi.num_codebooks)

        self.decoder_1 = Decoder(**cfg.model.flow_predictor)
        self.decoder_2 = Decoder(**cfg.model.flow_predictor)

        self.path = MixtureDiscreteProbPath(
            scheduler=PolynomialConvexScheduler(n=cfg.model.scheduler_n)
        )

        self.loss_fn = MixturePathGeneralizedKL(path=self.path)

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
            "monitor": "train_loss",
        }

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        _ = batch_idx
        loss = self.calc_loss(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        _ = batch_idx
        loss = self.calc_loss(batch)

        self.log("validation_loss", loss)

        wav_sr = self.cfg.model.mimi.sr
        if batch_idx < 5 and self.global_rank == 0 and self.local_rank == 0:
            wav_len = batch["wav_len"][0]
            source_1 = batch["wav_1"][0][:wav_len].cpu().numpy()
            source_2 = batch["wav_2"][0][:wav_len].cpu().numpy()
            source_merged = batch["wav_merged"][0][:wav_len].cpu().numpy()

            self.log_audio(source_1, f"source_1/{batch_idx}", wav_sr)
            self.log_audio(source_2, f"source_2/{batch_idx}", wav_sr)
            self.log_audio(source_merged, f"source_merged/{batch_idx}", wav_sr)

            with torch.no_grad():
                decoded_1 = (
                    self.mimi.decode(batch["token_1"])[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                decoded_2 = (
                    self.mimi.decode(batch["token_2"])[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                decoded_merged = (
                    self.mimi.decode(batch["token_merged"])[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

            self.log_audio(decoded_1, f"decoded_1/{batch_idx}", wav_sr)
            self.log_audio(decoded_2, f"decoded_2/{batch_idx}", wav_sr)
            self.log_audio(decoded_merged, f"decoded_merged/{batch_idx}", wav_sr)

            est_src1, est_src2 = self.forward(batch)

            with torch.no_grad():
                estimated_2 = (
                    self.mimi.decode(est_src2)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                estimated_1 = (
                    self.mimi.decode(est_src1)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

            self.log_audio(estimated_1, f"estimated_1/{batch_idx}", wav_sr)
            self.log_audio(estimated_2, f"estimated_2/{batch_idx}", wav_sr)

        return loss

    def calc_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token_1 = batch["token_1"]
        token_2 = batch["token_2"]
        token_merged = batch["token_merged"]

        batch_size = token_merged.size(0)

        lengths = batch["token_len"]
        mask = sequence_mask(lengths, torch.max(lengths)).unsqueeze(1).to(self.device)

        t = torch.rand((batch_size,), device=self.device)
        noise_1 = torch.randint_like(token_1, high=self.cfg.model.vocab_size)
        path_sample1 = self.path.sample(x_0=noise_1, x_1=token_1, t=t)
        noise_2 = torch.randint_like(token_2, high=self.cfg.model.vocab_size)
        path_sample2 = self.path.sample(x_0=noise_2, x_1=token_2, t=t)

        logits_1 = self.decoder_1.forward(path_sample1.x_t, mask, token_merged, t)
        logits_2 = self.decoder_2.forward(path_sample2.x_t, mask, token_merged, t)

        loss = self.loss_fn(
            logits=logits_1,
            x_1=token_1.to(torch.int64),
            x_t=path_sample1.x_t.to(torch.int64),
            t=path_sample1.t,
        ) + self.loss_fn(
            logits=logits_2,
            x_1=token_2.to(torch.int64),
            x_t=path_sample2.x_t.to(torch.int64),
            t=path_sample2.t,
        )

        return loss

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_merged = batch["token_merged"]

        lengths = batch["token_len"]
        mask = sequence_mask(lengths, torch.max(lengths)).unsqueeze(1).to(self.device)

        noise_1 = torch.randint_like(
            token_merged, high=self.cfg.model.vocab_size
        ).long()
        noise_2 = torch.randint_like(
            token_merged, high=self.cfg.model.vocab_size
        ).long()

        nfe = 64
        step_size = 1 / nfe
        n_plots = 9
        epsilon = 1e-3
        linspace_to_plot = torch.linspace(0, 1 - epsilon, n_plots)

        wrapped_model_1 = WrappedDecoder(self.decoder_1)
        wrapped_model_2 = WrappedDecoder(self.decoder_2)

        solver_1 = MixtureDiscreteEulerSolver(
            model=wrapped_model_1,
            path=self.path,
            vocabulary_size=self.cfg.model.vocab_size,
        )
        res_1 = solver_1.sample(
            x_init=noise_1,
            step_size=step_size,
            time_grid=linspace_to_plot,
            mask=mask,
            x_merged=token_merged,
        )
        assert isinstance(res_1, torch.Tensor)

        solver_2 = MixtureDiscreteEulerSolver(
            model=wrapped_model_2,
            path=self.path,
            vocabulary_size=self.cfg.model.vocab_size,
        )
        res_2 = solver_2.sample(
            x_init=noise_2,
            step_size=step_size,
            time_grid=linspace_to_plot,
            mask=mask,
            x_merged=token_merged,
        )
        assert isinstance(res_2, torch.Tensor)

        return res_1, res_2

    def log_audio(self, audio: np.ndarray, name: str, sampling_rate: int) -> None:
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                wandb.log({name: wandb.Audio(audio, sample_rate=sampling_rate)})
