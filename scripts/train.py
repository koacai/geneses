import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from hubert_separator.data.datamodule import HuBERTSeparatorDataModule
from hubert_separator.model.lightning_module import HuBERTSeparatorLightningModule


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(project="hubert-separator")

    L.seed_everything(42)

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=3300,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        precision="16-mixed",
    )

    hubert_separator = HuBERTSeparatorLightningModule(cfg)
    datamodule = HuBERTSeparatorDataModule(cfg.data.datamodule)

    datamodule.setup("fit")

    trainer.fit(
        hubert_separator,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


if __name__ == "__main__":
    main()
