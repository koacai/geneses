import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from dialogue_separator.data.datamodule import DialogueSeparatorDataModule
from dialogue_separator.model.lightning_module import DialogueSeparatorLightningModule


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(project="dialogue-separator")

    L.seed_everything(42)

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=3300,
        logger=wandb_logger,
        check_val_every_n_epoch=100,
        precision="16-mixed",
    )

    dialogue_separator = DialogueSeparatorLightningModule(cfg)
    datamodule = DialogueSeparatorDataModule(cfg.data.datamodule)

    datamodule.setup("fit")

    trainer.fit(
        dialogue_separator,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


if __name__ == "__main__":
    main()
