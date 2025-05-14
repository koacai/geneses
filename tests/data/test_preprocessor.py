import itertools
from pathlib import Path

import pytest
from hydra import compose, initialize
from lhotse import CutSet

from dialogue_separator.data.preprocessor import Preprocessor


# @pytest.mark.skip("ローカルで実行するためのテスト")
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

    def test_get_mimi_feature(self, init) -> None:
        _ = init

        for cut in self.cuts[0].cut_into_windows(duration=30):
            feature_1, feature_2, feature_merged = self.preprocessor.get_mimi_feature(
                cut
            )
            assert feature_1.size(0) == 512
            assert feature_2.size(0) == 512
            assert feature_merged.size(0) == 512
