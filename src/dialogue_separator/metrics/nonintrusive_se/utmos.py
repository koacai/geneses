import torch


def calc_utmos() -> None:
    utmos_tag = "utmos22_strong"
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", utmos_tag, trust_repo=True)
    print(model)
