from hydra import compose, initialize

from dialogue_separator.data.augment_datamodule import AugmentDataModule


def test_augment_datamodule() -> None:
    with initialize(config_path="../../config", version_base=None):
        cfg = compose(config_name="default").data.augment_datamodule
        datamodule = AugmentDataModule(cfg)

    datamodule.setup()
    for batch in datamodule.train_dataloader():
        print(batch)
        print(batch.keys())
        break
