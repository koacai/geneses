import hydra
from omegaconf import DictConfig

from dialogue_separator.data.augment_datamodule import AugmentDataModule
from dialogue_separator.data.wds_writer import run_parallel_writing


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the parallel preprocessing and writing for train and validation sets."""
    NUM_WRITERS = 32

    datamodule = AugmentDataModule(cfg.data.augment_datamodule)
    datamodule.setup()

    print("\n--- Starting Validation Set Processing ---")
    run_parallel_writing(
        dataloader=datamodule.val_dataloader(),
        output_dir=f"{cfg.data.augment_datamodule.out_dir}/valid",
        num_writers=NUM_WRITERS,
        shard_maxcount=cfg.data.augment_datamodule.valid.shard_maxcount,
    )
    print("\n--- Starting Test Set Processing ---")
    run_parallel_writing(
        dataloader=datamodule.test_dataloader(),
        output_dir=f"{cfg.data.augment_datamodule.out_dir}/test",
        num_writers=NUM_WRITERS,
        shard_maxcount=cfg.data.augment_datamodule.test.shard_maxcount,
    )
    print("--- Starting Training Set Processing ---")
    run_parallel_writing(
        dataloader=datamodule.train_dataloader(),
        output_dir=f"{cfg.data.augment_datamodule.out_dir}/train",
        num_writers=NUM_WRITERS,
        shard_maxcount=cfg.data.augment_datamodule.train.shard_maxcount,
    )

    print("\nAll processing and writing complete.")


if __name__ == "__main__":
    main()
