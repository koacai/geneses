import hydra
import lightning as L
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from flowditse.data.datamodule import GenesesDataModule
from flowditse.model.lightning_module import GenesesLightningModule


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    # ckpt_path = hf_hub_download(
    #     repo_id="koacai/flowditse", filename="only_bg_noise/epoch=8-step=136863.ckpt"
    # )
    ckpt_path = hf_hub_download(
        repo_id="koacai/flowditse", filename="complex_noise/epoch=20-step=151515.ckpt"
    )

    torch.set_float32_matmul_precision("medium")

    L.seed_everything(42)

    trainer = L.Trainer()

    geneses = GenesesLightningModule.load_from_checkpoint(ckpt_path)
    datamodule = GenesesDataModule(cfg.data.datamodule)

    datamodule.setup("test")

    trainer.test(
        geneses,
        dataloaders=datamodule.test_dataloader(),
    )


if __name__ == "__main__":
    main()
