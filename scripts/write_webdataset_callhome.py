import io
import uuid
from pathlib import Path

import hydra
import torch
import torchaudio
import webdataset as wds
from lhotse import CutSet, MultiCut
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    shar_dir = Path(cfg.data.dataset.shar_dir)

    cut_paths = sorted(list(map(str, shar_dir.glob("**/cuts.*.jsonl.gz"))))
    recording_paths = sorted(list(map(str, shar_dir.glob("**/recording.*.tar"))))

    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    Path(f"{cfg.data.dataset.shard_dir}/test").mkdir(parents=True, exist_ok=True)
    test_sink = wds.ShardWriter(
        f"{cfg.data.dataset.shard_dir}/test/data-%06d.tar",
        maxcount=cfg.data.dataset.shard_maxcount.test,
    )

    for cut in tqdm(cuts.data):
        assert isinstance(cut, MultiCut)

        buf = io.BytesIO()
        audio = torch.from_numpy(cut.load_audio())
        torchaudio.save(buf, audio, cut.sampling_rate, format="flac")

        sample = {
            "__key__": uuid.uuid1().hex,
            "audio.flac": buf.getvalue(),
            "wav_len_1.cls": cut.num_samples,
            "wav_len_2.cls": cut.num_samples,
        }
        test_sink.write(sample)

    test_sink.close()


if __name__ == "__main__":
    main()
