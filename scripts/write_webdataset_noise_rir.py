import io
import uuid
from pathlib import Path

import hydra
import torch
import torchaudio
import webdataset as wds
from lhotse import CutSet
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_cfgs = [cfg.data.test_noise_dataset, cfg.data.test_rir_dataset]

    for dataset_cfg in dataset_cfgs:
        shar_dir = Path(dataset_cfg.shar_dir)

        cut_paths = sorted(list(map(str, shar_dir.glob("**/cuts.*.jsonl.gz"))))
        recording_paths = sorted(list(map(str, shar_dir.glob("**/recording.*.tar"))))

        cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

        Path(f"{dataset_cfg.shard_dir}").mkdir(parents=True, exist_ok=True)
        sink = wds.ShardWriter(
            f"{dataset_cfg.shard_dir}/data-%06d.tar",
            maxcount=dataset_cfg.shard_maxcount,
        )

        for cut in tqdm(cuts.data):
            buf = io.BytesIO()
            audio = torch.from_numpy(cut.load_audio())
            torchaudio.save(buf, audio, cut.sampling_rate, format="flac")

            sample = {"__key__": uuid.uuid1().hex, "audio.flac": buf.getvalue()}
            sink.write(sample)

        sink.close()


if __name__ == "__main__":
    main()
