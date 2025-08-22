import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from flowditse.data.datamodule import FlowDiTSEDataModule
from flowditse.model.lightning_module import FlowDiTSELightningModule


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    torch.set_float32_matmul_precision("medium")

    L.seed_everything(42)

    trainer = hydra.utils.instantiate(cfg.train.trainer)

    dialogue_separator = FlowDiTSELightningModule(cfg)
    datamodule = FlowDiTSEDataModule(cfg.data.datamodule)

    datamodule.setup("fit")

    trainer.fit(
        dialogue_separator,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


if __name__ == "__main__":
    main()
