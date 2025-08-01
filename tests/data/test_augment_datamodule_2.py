import itertools

from hydra import compose, initialize

from dialogue_separator.data.augment_datamodule_2 import AugmentDataModule2


def test_augment_datamodule_train_dataloader() -> None:
    with initialize(config_path="../../config", version_base=None):
        cfg = compose(config_name="default").data.augment_datamodule_2
        datamodule = AugmentDataModule2(cfg)

    datamodule.setup()
    for batch in itertools.islice(datamodule.train_dataloader(), 3):
        print(batch)
