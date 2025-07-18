from pathlib import Path

from dialogue_separator.util.util import download_file

PRIMARY_MODEL_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
P808_MODEL_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/refs/heads/master/DNSMOS/DNSMOS/model_v8.onnx"
FS = 16000


def calc_dnsmos() -> None:
    model_dir = Path("DNSMOS")
    model_dir.mkdir(exist_ok=True)

    primary_model_path = model_dir / "sig_bak_ovr.onnx"
    if not primary_model_path.exists():
        download_file(PRIMARY_MODEL_URL, primary_model_path)

    p808_model_path = model_dir / "model_v8.onnx"
    if not p808_model_path.exists():
        download_file(P808_MODEL_URL, p808_model_path)
