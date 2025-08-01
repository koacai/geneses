import itertools

from hydra import compose, initialize

from dialogue_separator.data.preprocess_datamodule import PreprocessDataModule


def test_preprocess_datamodule_train_dataloader() -> None:
    with initialize(config_path="../../config", version_base=None):
        cfg = compose(config_name="default").data.preprocess_datamodule
        datamodule = PreprocessDataModule(cfg)

    datamodule.setup()
    for batch in itertools.islice(datamodule.train_dataloader(), 3):
        print(batch)
