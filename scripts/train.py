import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from geneses.data.datamodule import GenesesDataModule
from geneses.model.lightning_module import GenesesLightningModule


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    torch.set_float32_matmul_precision("medium")

    L.seed_everything(42)

    trainer = hydra.utils.instantiate(cfg.train.trainer)

    geneses = GenesesLightningModule(cfg)
    datamodule = GenesesDataModule(cfg.data.datamodule)

    datamodule.setup("fit")

    trainer.fit(
        geneses,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


if __name__ == "__main__":
    main()
