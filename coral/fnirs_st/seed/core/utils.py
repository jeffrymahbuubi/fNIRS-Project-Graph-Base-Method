import os
import random
import shutil
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_experiment_dir(
    experiment_name: str, base_dir: str = "experiments", overwrite: bool = False
) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    exp_dir = os.path.join(base_dir, date_str, experiment_name)
    if os.path.exists(exp_dir) and overwrite:
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def compute_statistical_features(
    data: np.ndarray, channels_first: bool = True
) -> Dict[str, np.ndarray]:
    x = data.astype(np.float64, copy=False)
    if not channels_first:
        x = x.T
    C, N = x.shape
    eps = 1e-15
    mean = x.mean(axis=1)
    centered = x - mean[:, None]
    var = (centered ** 2).mean(axis=1)
    m3 = (centered ** 3).mean(axis=1)
    m4 = (centered ** 4).mean(axis=1)
    var_pos = var > eps
    skewness = np.full(C, np.nan, dtype=np.float64)
    kurtosis = np.full(C, np.nan, dtype=np.float64)
    skewness[var_pos] = m3[var_pos] / np.power(var[var_pos], 1.5)
    kurtosis[var_pos] = m4[var_pos] / np.power(var[var_pos], 2.0)
    return {
        "mean": mean, "min": x.min(axis=1), "max": x.max(axis=1),
        "skewness": skewness, "kurtosis": kurtosis, "variance": var,
    }


def pearson_correlation_matrix(data: np.ndarray, channels_first: bool = True) -> np.ndarray:
    x = data.astype(np.float64, copy=False)
    if not channels_first:
        x = x.T
    eps = 1e-15
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    ss = np.sqrt((xc ** 2).sum(axis=1))
    ss[ss < eps] = np.inf
    R = np.clip((xc @ xc.T) / (ss[:, None] * ss[None, :]), -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    return R


def _hann_window(M: int) -> np.ndarray:
    if M <= 1:
        return np.ones(M, dtype=np.float64)
    n = np.arange(M, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / M)


def coherence_matrix(
    data: np.ndarray,
    fs: float = 1.0,
    coherence_ratio: str = "1/3",
    channels_first: bool = True,
    return_spectrum: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x = data.astype(np.float64, copy=False)
    if not channels_first:
        x = x.T
    C, N = x.shape
    ratio_map = {"1/3": 1 / 3, "1/2": 1 / 2, "2/3": 2 / 3}
    seg_length = int(N * ratio_map[coherence_ratio])
    step = max(1, int(seg_length * 0.5))
    if seg_length > N:
        seg_length = N
        step = N
    w = _hann_window(seg_length)
    F_bins = seg_length // 2 + 1
    Sxx = np.zeros((C, F_bins), dtype=np.complex128)
    Sxy = np.zeros((C, C, F_bins), dtype=np.complex128)
    n_segments = 0
    for start in range(0, N - seg_length + 1, step):
        seg = x[:, start: start + seg_length]
        seg = seg - seg.mean(axis=1, keepdims=True)
        X = np.fft.rfft(seg * w, n=seg_length, axis=1)
        Sxx += X * np.conj(X)
        for i in range(C):
            Sxy[:, i, :] += X * np.conj(X[i, :])
        n_segments += 1
    if n_segments == 0:
        raise ValueError("Not enough samples for a single coherence segment.")
    Sxx /= n_segments
    Sxy /= n_segments
    eps = 1e-30
    Sxx_real = np.maximum(Sxx.real, eps)
    denom = Sxx_real[:, None, :] * Sxx_real[None, :, :]
    coh_spec = np.clip((np.abs(Sxy) ** 2) / denom, 0.0, 1.0)
    f = np.fft.rfftfreq(seg_length, d=1.0 / fs)
    valid = slice(1, -1) if F_bins >= 3 else slice(None)
    coh_mean = coh_spec[..., valid].mean(axis=-1)
    np.fill_diagonal(coh_mean, 1.0)
    return coh_mean, f, coh_spec if return_spectrum else None
