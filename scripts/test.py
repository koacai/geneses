import hydra
import lightning as L
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from dialogue_separator.data.datamodule import DialogueSeparatorDataModule
from dialogue_separator.model.lightning_module import DialogueSeparatorLightningModule


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    # hf_path = "Libri2Mix/MSELoss/epoch=113-step=57912.ckpt"
    hf_path = "Libri2Mix/L1Loss/epoch=112-step=57404.ckpt"
    ckpt_path = hf_hub_download("koacai/dialogue-separator", hf_path)

    torch.set_float32_matmul_precision("medium")

    L.seed_everything(42)

    trainer = L.Trainer(limit_test_batches=3)

    dialogue_separator = DialogueSeparatorLightningModule.load_from_checkpoint(
        ckpt_path
    )
    datamodule = DialogueSeparatorDataModule(cfg.data.datamodule)

    datamodule.setup("test")

    trainer.test(
        dialogue_separator,
        dataloaders=datamodule.test_dataloader(),
    )


if __name__ == "__main__":
    main()
