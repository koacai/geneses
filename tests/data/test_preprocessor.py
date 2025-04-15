import itertools
from pathlib import Path

import pytest
import torch
import webdataset as wds
from hydra import compose, initialize
from lhotse import CutSet

from hubert_separator.data.preprocessor import Preprocessor


class TestPreprocessor:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default").data.preprocess
            self.preprocessor = Preprocessor(cfg)
            shar_dir = Path(cfg.shar_dir)
            cut_paths = sorted(map(str, shar_dir.glob("cuts.*.jsonl.gz")))
            recording_paths = sorted(map(str, shar_dir.glob("recording.*.tar")))
            self.cuts = CutSet.from_shar(
                {"cuts": cut_paths, "recording": recording_paths}
            )

    def test_process_cut(self, init) -> None:
        _ = init

        for cut in itertools.islice(self.cuts.data, 3):
            res = self.preprocessor.process_cut(cut)
            assert isinstance(res, list)
            assert isinstance(res[0], dict)
            token_1 = wds.torch_loads(res[0]["token_1.pth"])
            assert isinstance(token_1, torch.Tensor)
            token_2 = wds.torch_loads(res[0]["token_2.pth"])
            assert isinstance(token_2, torch.Tensor)
            token_merged = wds.torch_loads(res[0]["token_merged.pth"])
            assert isinstance(token_merged, torch.Tensor)

            assert token_1.shape == token_2.shape == token_merged.shape
