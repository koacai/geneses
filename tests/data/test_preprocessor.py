import itertools
from pathlib import Path

import pytest
from hydra import compose, initialize
from lhotse import CutSet

from dialogue_separator.data.preprocessor import Preprocessor


class TestPreprocessor:
    @pytest.fixture
    def init(self) -> None:
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="default").data.preprocess.cfg
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
            assert isinstance(res, dict)
