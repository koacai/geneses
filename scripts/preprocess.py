import hydra
from omegaconf import DictConfig

from dialogue_separator.data.preprocessor import Preprocessor


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    preprocessor = Preprocessor(cfg.data.preprocess)
    preprocessor.write_webdataset()


if __name__ == "__main__":
    main()
