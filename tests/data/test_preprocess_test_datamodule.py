import itertools

from hydra import compose, initialize

from flowditse.data.preprocess_test_datamodule import PreprocessTestDataModule


def test_preprocess_test_datamodule_test_dataloader() -> None:
    with initialize(config_path="../../config", version_base=None):
        cfg = compose(config_name="default").data.preprocess_datamodule
        datamodule = PreprocessTestDataModule(cfg)

    datamodule.setup()
    batch_size = datamodule.cfg.batch_size
    for batch in itertools.islice(datamodule.test_dataloader(), 3):
        assert batch["raw_wav_1"].size(0) == batch_size
        assert batch["raw_wav_2"].size(0) == batch_size
        assert batch["clean_wav"].size(0) == batch_size
        assert batch["noisy_wav"].size(0) == batch_size
        assert batch["wav_len"].size(0) == batch_size
        assert "ssl_input" in batch
