from typing import Any, Generator

import torch
from lhotse import CutSet
from webdataset import FluidInterface
from webdataset.pipeline import DataPipeline


class LibriTTSRMixDataset(DataPipeline, FluidInterface):
    def __init__(self, cuts: CutSet) -> None:
        super(LibriTTSRMixDataset, self).__init__()
        self.cuts = cuts

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        for cut in self.cuts.data:
            wav = torch.from_numpy(cut.load_audio())
            assert len(cut.supervisions) == 2
            assert cut.supervisions[0].custom is not None
            assert cut.supervisions[1].custom is not None

            yield {
                "audio": (wav, cut.sampling_rate),
                "text_1": cut.supervisions[0].text,
                "text_2": cut.supervisions[1].text,
                "wav_len_1": cut.supervisions[0].custom["wav_len"],
                "wav_len_2": cut.supervisions[1].custom["wav_len"],
            }
