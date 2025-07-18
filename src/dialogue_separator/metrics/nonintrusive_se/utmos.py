import torch


def calc_utmos(audio: torch.Tensor, sr: int, use_gpu: bool) -> float:
    utmos_tag = "utmos22_strong"
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", utmos_tag, trust_repo=True)

    device = torch.device("cuda" if use_gpu else "cpu")
    model.device = device  # type: ignore

    score = model(audio.unsqueeze(0).to(device=model.device), sr)  # type: ignore
    return score.cpu().item()
