from pathlib import Path

import numpy as np
import torch
import torchaudio

from dialogue_separator.util.util import download_file

PRIMARY_MODEL_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
P808_MODEL_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/refs/heads/master/DNSMOS/DNSMOS/model_v8.onnx"
SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


def poly1d(coefficients, use_numpy=False):
    if use_numpy:
        return np.poly1d(coefficients)
    coefficients = tuple(reversed(coefficients))

    def func(p):
        return sum(coef * p**i for i, coef in enumerate(coefficients))

    return func


class DNSMOS_local:
    # ported from
    # https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
    def __init__(
        self, primary_model_path: Path, p808_model_path: Path, use_gpu: bool = False
    ) -> None:
        self.use_gpu = use_gpu
        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            raise RuntimeError("Please install onnxruntime manually and retry!")

        prvd = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
        if primary_model_path is not None:
            self.onnx_sess = ort.InferenceSession(primary_model_path, providers=[prvd])
            self.p808_onnx_sess = ort.InferenceSession(
                p808_model_path, providers=[prvd]
            )

    def audio_melspec(
        self,
        audio: torch.Tensor,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        sr: int = 16000,
        to_db: bool = True,
    ) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        mel_spec = mel_transform(audio)

        if to_db:
            mel_spec = torchaudio.transforms.AmplitudeToDB(stype="power")(mel_spec)
            mel_spec = (mel_spec + 40) / 40

        mel_spec = mel_spec.squeeze(0).transpose(0, 1)

        return mel_spec

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        flag = True
        if is_personalized_MOS:
            p_ovr = poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046], flag)
            p_sig = poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726], flag)
            p_bak = poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132], flag)
        else:
            p_ovr = poly1d([-0.06766283, 1.11546468, 0.04602535], flag)
            p_sig = poly1d([-0.08397278, 1.22083953, 0.0052439], flag)
            p_bak = poly1d([-0.13166888, 1.60915514, -0.39604546], flag)

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(
        self, aud: torch.Tensor, input_fs: int, is_personalized_MOS: bool = False
    ) -> dict[str, torch.Tensor]:
        if input_fs != SAMPLING_RATE:
            audio = torchaudio.functional.resample(
                aud, orig_freq=input_fs, new_freq=SAMPLING_RATE
            )
        else:
            audio = aud

        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            audio = torch.cat([audio, audio])

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = audio_seg.cpu().numpy().astype("float32")[np.newaxis, :]
            p808_input_features = (
                self.audio_melspec(audio=audio_seg[:-160])
                .cpu()
                .numpy()
                .astype("float32")[np.newaxis, :, :]
            )
            p808_mos = self.p808_onnx_sess.run(  # type: ignore
                None, {"input_1": p808_input_features}
            )[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(  # type: ignore
                None, {"input_1": input_features}
            )[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        to_array = torch.stack
        return {
            "OVRL_raw": to_array(predicted_mos_ovr_seg_raw).mean(),
            "SIG_raw": to_array(predicted_mos_sig_seg_raw).mean(),
            "BAK_raw": to_array(predicted_mos_bak_seg_raw).mean(),
            "OVRL": to_array(predicted_mos_ovr_seg).mean(),
            "SIG": to_array(predicted_mos_sig_seg).mean(),
            "BAK": to_array(predicted_mos_bak_seg).mean(),
            "P808_MOS": to_array(predicted_p808_mos).mean(),
        }


def calc_dnsmos(use_gpu: bool) -> None:
    model_dir = Path("DNSMOS")
    model_dir.mkdir(exist_ok=True)

    primary_model_path = model_dir / "sig_bak_ovr.onnx"
    if not primary_model_path.exists():
        download_file(PRIMARY_MODEL_URL, primary_model_path)

    p808_model_path = model_dir / "model_v8.onnx"
    if not p808_model_path.exists():
        download_file(P808_MODEL_URL, p808_model_path)

    model = DNSMOS_local(primary_model_path, primary_model_path, use_gpu=use_gpu)
    print(model)
