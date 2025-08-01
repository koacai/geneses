from pathlib import Path

from lhotse import CutSet

from dialogue_separator.data.dataset import LibriTTSRMixDataset


def test_libritts_r_mix_dataset() -> None:
    shar_dir = Path("/groups/gag51394/users/asai/shar/libritts_r_mix_clean/")
    cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
    recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
    cuts = CutSet.from_shar({"cuts": cut_paths, "recording": recording_paths})

    dataset = LibriTTSRMixDataset(cuts)
    for data in dataset:
        print(data)
        break
