import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from litmodels import download_model
from omegaconf import DictConfig

from dialogue_separator.data.datamodule import DialogueSeparatorDataModule
from dialogue_separator.model.lightning_module import DialogueSeparatorLightningModule


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    ckpt_paths = download_model(
        name="koacai/speech/dialogue-separator", download_dir="model_ckpt"
    )
    assert len(ckpt_paths) == 1

    torch.set_float32_matmul_precision("medium")

    L.seed_everything(42)

    trainer = L.Trainer(limit_test_batches=1)

    dialogue_separator = DialogueSeparatorLightningModule.load_from_checkpoint(
        f"model_ckpt/{ckpt_paths[0]}",
    )
    datamodule = DialogueSeparatorDataModule(cfg.data.datamodule)

    datamodule.setup("test")

    trainer.test(
        dialogue_separator,
        dataloaders=datamodule.test_dataloader(),
    )


if __name__ == "__main__":
    main()
