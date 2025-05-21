import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    preprocessor = hydra.utils.instantiate(cfg.data.preprocess)
    preprocessor.write_webdataset()


if __name__ == "__main__":
    main()
