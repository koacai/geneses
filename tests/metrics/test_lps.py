import torch
from dotenv import load_dotenv

from ditse.metrics.lps import LevenshteinPhonemeSimilarity, lps_metrics


def test_lps() -> None:
    load_dotenv()
    device = torch.device("cuda")
    lps = LevenshteinPhonemeSimilarity(device)
    wav1 = torch.randn(16000)
    wav2 = torch.randn(16000)
    m = lps_metrics(lps, wav1, wav2, 16000)
    print(m)
