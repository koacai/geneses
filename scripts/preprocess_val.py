import hydra
from omegaconf import DictConfig

from geneses.data.preprocess_datamodule import PreprocessDataModule
from geneses.data.wds_writer import run_parallel_writing


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the parallel preprocessing and writing for train and validation sets."""
    NUM_WRITERS = 1

    # Pass split_idx and array_num from top-level config to preprocess_datamodule
    split_idx = None
    if "split_idx" in cfg:
        split_idx = cfg.split_idx
        cfg.data.preprocess_datamodule.split_idx = cfg.split_idx
    if "array_num" in cfg:
        cfg.data.preprocess_datamodule.array_num = cfg.array_num

    datamodule = PreprocessDataModule(cfg.data.preprocess_datamodule)
    datamodule.setup()

    # Determine output directory based on whether parallel processing is enabled
    base_out_dir = cfg.data.preprocess_datamodule.out_dir
    if split_idx is not None:
        # Each job writes to its own subdirectory
        valid_out_dir = f"{base_out_dir}/valid/job_{split_idx:03d}"
    else:
        valid_out_dir = f"{base_out_dir}/valid"

    print("\n--- Starting Validation Set Processing ---")
    run_parallel_writing(
        dataloader=datamodule.val_dataloader(),
        output_dir=valid_out_dir,
        num_writers=NUM_WRITERS,
        shard_maxcount=cfg.data.preprocess_datamodule.shard_maxcount.valid,
    )

    print("\nAll processing and writing complete.")


if __name__ == "__main__":
    main()
