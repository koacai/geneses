import io
import uuid
from pathlib import Path

import hydra
import torch
import torchaudio
import webdataset as wds
from lhotse import MultiCut
from lhotse_dataset import LibriTTSR
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    libritts_r = LibriTTSR()
    libritts_r.write_shar(Path(cfg.data.dataset.libritts_r_shar_dir))

    libritts_r_mix = hydra.utils.instantiate(cfg.data.dataset.libritts_r_mix)

    Path(f"{cfg.data.dataset.shard_dir}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{cfg.data.dataset.shard_dir}/valid").mkdir(parents=True, exist_ok=True)
    Path(f"{cfg.data.dataset.shard_dir}/test").mkdir(parents=True, exist_ok=True)
    train_sink = wds.ShardWriter(
        f"{cfg.data.dataset.shard_dir}/train/data-%06d.tar",
        maxcount=cfg.data.dataset.shard_maxcount.train,
    )
    valid_sink = wds.ShardWriter(
        f"{cfg.data.dataset.shard_dir}/valid/data-%06d.tar",
        maxcount=cfg.data.dataset.shard_maxcount.valid,
    )
    test_sink = wds.ShardWriter(
        f"{cfg.data.dataset.shard_dir}/test/data-%06d.tar",
        maxcount=cfg.data.dataset.shard_maxcount.test,
    )

    for cut in tqdm(libritts_r_mix.get_cuts()):
        assert isinstance(cut, MultiCut)

        buf = io.BytesIO()
        audio = torch.from_numpy(cut.load_audio())
        torchaudio.save(buf, audio, cut.sampling_rate, format="flac")

        assert isinstance(cut.custom, dict)
        assert len(cut.supervisions) == 2
        assert cut.supervisions[0].custom is not None
        assert cut.supervisions[1].custom is not None

        sample = {
            "__key__": uuid.uuid1().hex,
            "audio.flac": buf.getvalue(),
            "text_1.txt": cut.supervisions[0].text,
            "text_2.txt": cut.supervisions[1].text,
            "wav_len_1.cls": cut.supervisions[0].custom["wav_len"],
            "wav_len_2.cls": cut.supervisions[1].custom["wav_len"],
        }
        if cut.custom["subset"] == "dev_clean":
            valid_sink.write(sample)
        elif cut.custom["subset"] == "test_clean":
            test_sink.write(sample)
        elif cut.custom["subset"] in ["train_clean_100", "train_clean_360"]:
            train_sink.write(sample)

    train_sink.close()
    valid_sink.close()
    test_sink.close()


if __name__ == "__main__":
    main()
