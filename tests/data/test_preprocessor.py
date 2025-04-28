import itertools
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from lhotse import CutSet

from dialogue_separator.data.preprocessor import Preprocessor


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

    def test_get_audio_tokens(self, init) -> None:
        _ = init

        for cut in itertools.islice(self.cuts.data, 3):
            audio, token_1, token_2, token_merged = self.preprocessor.get_audio_tokens(
                cut
            )
            assert isinstance(audio, torch.Tensor)
            assert isinstance(token_1, torch.Tensor)
            assert isinstance(token_2, torch.Tensor)
            assert isinstance(token_merged, torch.Tensor)

            assert token_1.shape == token_2.shape == token_merged.shape

    def test_get_xvector(self, init) -> None:
        _ = init

        for cut in itertools.islice(self.cuts.data, 3):
            xvector1, xvector2 = self.preprocessor.get_xvector(cut)
            assert isinstance(xvector1, torch.Tensor)
            assert isinstance(xvector2, torch.Tensor)
