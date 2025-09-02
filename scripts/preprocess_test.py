import hydra
from omegaconf import DictConfig

from flowditse.data.preprocess_test_datamodule import PreprocessTestDataModule
from flowditse.data.wds_writer import run_parallel_writing


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    NUM_WRITERS = 1

    datamodule = PreprocessTestDataModule(cfg.data.preprocess_datamodule)
    datamodule.setup()

    print("--- Starting Test Set Processing ---")
    run_parallel_writing(
        dataloader=datamodule.test_dataloader(),
        output_dir=f"{cfg.data.preprocess_datamodule.out_dir}/test",
        num_writers=NUM_WRITERS,
        shard_maxcount=cfg.data.preprocess_datamodule.shard_maxcount.test,
    )

    print("\nAll processing and writing complete.")


if __name__ == "__main__":
    main()
