import math
from pathlib import Path

import librosa
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
            from onnx2torch import convert
        except ModuleNotFoundError:
            raise RuntimeError("Please install onnx2torch manually and retry!")

        if primary_model_path is not None:
            self.primary_model = convert(primary_model_path).eval()
            self.p808_model = convert(p808_model_path).eval()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=321, hop_length=160, pad_mode="constant"
        )

        self.to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80.0)
        if use_gpu:
            if primary_model_path is not None:
                self.primary_model = self.primary_model.cuda()
                self.p808_model = self.p808_model.cuda()
            self.spectrogram = self.spectrogram.cuda()

    def audio_melspec(
        self,
        audio: torch.Tensor,
        n_mels: int = 120,
        frame_size: int = 320,
        sr: int = 16000,
        to_db: bool = True,
    ) -> torch.Tensor:
        specgram = self.spectrogram(audio)
        fb = torch.as_tensor(
            librosa.filters.mel(sr=sr, n_fft=frame_size + 1, n_mels=n_mels).T,
            dtype=audio.dtype,
            device=audio.device,
        )
        mel_spec = torch.matmul(specgram.transpose(-1, -2), fb).transpose(-1, -2)
        if to_db:
            self.to_db.db_multiplier = math.log10(
                max(self.to_db.amin, torch.max(mel_spec).item())
            )
            mel_spec = (self.to_db(mel_spec) + 40) / 40

        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        flag = False
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
        device = "cuda" if self.use_gpu else "cpu"
        aud = aud.to(device=device)

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

            input_features = audio_seg.float()[None, :]
            p808_input_features = self.audio_melspec(audio=audio_seg[:-160]).float()[
                None, :, :
            ]
            p808_mos = self.p808_model(p808_input_features)
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.primary_model(input_features)[
                0
            ]

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


def calc_dnsmos(audio: torch.Tensor, sr: int, use_gpu: bool) -> float:
    model_dir = Path("DNSMOS")
    model_dir.mkdir(exist_ok=True)

    primary_model_path = model_dir / "sig_bak_ovr.onnx"
    if not primary_model_path.exists():
        download_file(PRIMARY_MODEL_URL, primary_model_path)

    p808_model_path = model_dir / "model_v8.onnx"
    if not p808_model_path.exists():
        download_file(P808_MODEL_URL, p808_model_path)

    model = DNSMOS_local(primary_model_path, p808_model_path, use_gpu=use_gpu)
    score = model(audio, input_fs=sr)
    return score["OVRL"].item()
