import webdataset as wds
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from transformers import AutoFeatureExtractor


class AugmentDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(AugmentDataModule, self).__init__()
        self.cfg = cfg
        self.processor = AutoFeatureExtractor.from_pretrained(cfg.ssl_model.name)

    def setup(self, stage: str | None = None) -> None:
        _ = stage

        self.train_dataset = wds.WebDataset(
            self.cfg.train.dataset_path,
            shardshuffle=100,
            nodesplitter=lambda x: x,
            repeat=True,
        ).decode(wds.torch_audio)

        self.valid_dataset = wds.WebDataset(
            self.cfg.valid.dataset_path,
            shardshuffle=False,
            nodesplitter=lambda x: x,
            repeat=True,
        ).decode(wds.torch_audio)

        self.test_dataset = wds.WebDataset(
            self.cfg.test.dataset_path,
            shardshuffle=False,
            nodesplitter=wds.single_node_only,
            repeat=True,
        ).decode(wds.torch_audio)

    def train_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.valid_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> wds.WebLoader:
        return wds.WebLoader(
            self.test_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.identity,
            drop_last=True,
            persistent_workers=True,
        )

    def identity(self, x):
        return x[0]
