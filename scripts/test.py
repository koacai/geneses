import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from dialogue_separator.data.datamodule import DialogueSeparatorDataModule
from dialogue_separator.model.lightning_module import DialogueSeparatorLightningModule


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")

    L.seed_everything(42)

    trainer = L.Trainer(limit_test_batches=3)

    dialogue_separator = DialogueSeparatorLightningModule.load_from_checkpoint(
        "dialogue-separator/etn2w208/checkpoints/epoch=54-step=27940.ckpt"
    )
    datamodule = DialogueSeparatorDataModule(cfg.data.datamodule)

    datamodule.setup("test")

    trainer.test(
        dialogue_separator,
        dataloaders=datamodule.test_dataloader(),
    )


if __name__ == "__main__":
    main()
