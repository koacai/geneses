import itertools

from hydra import compose, initialize

from flowditse.data.datamodule import FlowDiTSEDataModule


def test_augment_datamodule_test_dataloader() -> None:
    with initialize(config_path="../../config", version_base=None):
        cfg = compose(config_name="default").data.datamodule
        datamodule = FlowDiTSEDataModule(cfg)

    datamodule.setup("test")
    batch_size = datamodule.cfg.test.batch_size
    for batch in itertools.islice(datamodule.test_dataloader(), 3):
        assert batch["raw_wav_1"].size(0) == batch_size
        assert batch["raw_wav_2"].size(0) == batch_size
        assert batch["clean_wav"].size(0) == batch_size
        assert batch["wav_len"].size(0) == batch_size
        assert len(batch["text_1"]) == batch_size
        assert len(batch["text_2"]) == batch_size
        assert "ssl_input" in batch
