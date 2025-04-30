import hydra
import numpy as np
import torch
import torch.nn.functional as F
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
from dialogue_separator.utils.model import fix_len_compatibility, sequence_mask

from .decoder import Decoder


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

        self.decoder = Decoder(**cfg.model.decoder)

        self.path = MixtureDiscreteProbPath(
            scheduler=PolynomialConvexScheduler(n=cfg.model.scheduler_n)
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
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            mimi = loaders.get_mimi(mimi_weight, device=self.device)
            mimi.set_num_codebooks(self.cfg.model.mimi.num_codebooks)

            wav_len = batch["wav_len"][0]
            source_1 = batch["wav_1"][0][:wav_len].cpu().numpy()
            source_2 = batch["wav_2"][0][:wav_len].cpu().numpy()
            source_merged = batch["wav_merged"][0][:wav_len].cpu().numpy()

            self.log_audio(source_1, f"source_1/{batch_idx}", wav_sr)
            self.log_audio(source_2, f"source_2/{batch_idx}", wav_sr)
            self.log_audio(source_merged, f"source_merged/{batch_idx}", wav_sr)

            with torch.no_grad():
                decoded_1 = (
                    mimi.decode(batch["token_1"])[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                decoded_2 = (
                    mimi.decode(batch["token_2"])[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                decoded_merged = (
                    mimi.decode(batch["token_merged"])[0]
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
                estimated_1 = (
                    mimi.decode(est_src1)[0]
                    .squeeze()[:wav_len]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                estimated_2 = (
                    mimi.decode(est_src2)[0]
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

        orig_len = token_merged.size(-1)
        new_len = fix_len_compatibility(orig_len)
        token_merged = F.pad(token_merged, (0, new_len - orig_len))
        token_1 = F.pad(token_1, (0, new_len - orig_len))
        token_2 = F.pad(token_2, (0, new_len - orig_len))

        lengths = batch["token_len"]
        mask = sequence_mask(lengths, new_len).unsqueeze(1).to(self.device)

        t = torch.rand((batch_size,), device=self.device)
        token_both = torch.stack([token_1, token_2], dim=1)
        noise = torch.randint_like(token_both, high=self.cfg.model.vocab_size)
        path_sample = self.path.sample(x_0=noise, x_1=token_both, t=t)

        logits = self.decoder.forward(path_sample.x_t, mask, token_merged, t)

        loss_fn = MixturePathGeneralizedKL(path=self.path)

        loss = loss_fn(
            logits=logits,
            x_1=token_both.to(torch.int64),
            x_t=path_sample.x_t.to(torch.int64),
            t=path_sample.t,
        )
        return loss

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_merged = batch["token_merged"]

        orig_len = token_merged.size(-1)
        new_len = fix_len_compatibility(orig_len)
        token_merged = F.pad(token_merged, (0, new_len - orig_len))

        lengths = batch["token_len"]
        mask = sequence_mask(lengths, new_len).unsqueeze(1).to(self.device)

        token_both = torch.stack([token_merged, token_merged], dim=1)
        noise = torch.randint_like(token_both, high=self.cfg.model.vocab_size).long()

        nfe = 64
        step_size = 1 / nfe
        n_plots = 9
        epsilon = 1e-3
        linspace_to_plot = torch.linspace(0, 1 - epsilon, n_plots)

        wrapped_model = WrappedDecoder(self.decoder)

        solver = MixtureDiscreteEulerSolver(
            model=wrapped_model,
            path=self.path,
            vocabulary_size=self.cfg.model.vocab_size,
        )
        res = solver.sample(
            x_init=noise,
            step_size=step_size,
            time_grid=linspace_to_plot,
            mask=mask,
            x_merged=token_merged,
        )
        assert isinstance(res, torch.Tensor)

        return res[:, :, :, 0], res[:, :, :, 1]

    def log_audio(self, audio: np.ndarray, name: str, sampling_rate: int) -> None:
        for logger in self.loggers:
            if isinstance(logger, loggers.WandbLogger):
                wandb.log({name: wandb.Audio(audio, sample_rate=sampling_rate)})
