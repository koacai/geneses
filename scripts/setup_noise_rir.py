import io
import uuid
from pathlib import Path

import hydra
import soundfile as sf
import torch
import webdataset as wds
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_cfgs = [cfg.data.test_rir_dataset, cfg.data.test_noise_dataset]

    for dataset_cfg in dataset_cfgs:
        corpus = hydra.utils.instantiate(dataset_cfg.corpus)

        Path(f"{dataset_cfg.shard_dir}").mkdir(parents=True, exist_ok=True)
        sink = wds.ShardWriter(
            f"{dataset_cfg.shard_dir}/data-%06d.tar",
            maxcount=dataset_cfg.shard_maxcount,
        )

        for cut in tqdm(corpus.get_cuts()):
            buf = io.BytesIO()
            audio = torch.from_numpy(cut.load_audio())
            sf.write(buf, audio.T.numpy(), cut.sampling_rate, format="FLAC")

            sample = {"__key__": uuid.uuid1().hex, "audio.flac": buf.getvalue()}
            sink.write(sample)

        sink.close()


if __name__ == "__main__":
    main()
