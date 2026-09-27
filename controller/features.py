"""
Windowing and feature extraction for multi-channel surface EMG.

Two feature families are computed for every analysis window and every channel:

* TD  - the classic time-domain set used in myoelectric control (Hudgins et al., 1993, plus RMS):
        mean absolute value (MAV), root mean square (RMS), waveform length (WL),
        zero crossings (ZC) and slope sign changes (SSC).
* MFCC - mel-frequency cepstral coefficients of the window, borrowed from speech processing:
        power spectrum -> triangular mel filter bank -> log -> DCT.

At 1 kHz the whole EMG band (20-500 Hz) sits below the ~1 kHz knee of the mel scale, so the mel
filters here are close to linearly spaced; what MFCC contributes on EMG is mainly a compact,
decorrelated description of the spectral envelope.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from python_speech_features import mfcc

FS = 1000                      # Hz, MYO bracelet as recorded in the UCI dataset (time column in ms)
GESTURES = {1: "rest", 2: "fist", 3: "wrist flexion", 4: "wrist extension",
            5: "radial deviation", 6: "ulnar deviation"}   # class 7 (extended palm): only 2 subjects, dropped


def load_uci(root: Path) -> list:
    """Return one dict per labelled gesture segment: subject, series, gesture, signal (n, 8)."""
    segments = []
    for f in sorted(Path(root).glob("*/*_raw_data_*.txt")):
        subject, series = int(f.parent.name), int(f.name.split("_")[0])
        table = pd.read_csv(f, sep="\t").dropna()
        labels = table["class"].to_numpy(int)
        signal = table[[f"channel{i}" for i in range(1, 9)]].to_numpy(float)
        change = np.flatnonzero(np.diff(labels) != 0) + 1
        for start, end in zip(np.r_[0, change], np.r_[change, len(labels)]):
            if labels[start] in GESTURES:
                segments.append(dict(subject=subject, series=series, gesture=int(labels[start]),
                                     signal=signal[start:end]))
    return segments


def windows(signal: np.ndarray, length: int, step: int, trim: int):
    """Slide a window over a segment, skipping `trim` samples at both ends (gesture onset/offset)."""
    x = signal[trim:len(signal) - trim]
    for start in range(0, len(x) - length + 1, step):
        yield x[start:start + length]


def td_features(w: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """(length, channels) -> 5 * channels: MAV, RMS, WL, ZC, SSC per channel."""
    d = np.diff(w, axis=0)
    mav = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum((w[:-1] * w[1:] < 0) & (np.abs(d) >= threshold), axis=0)
    ssc = np.sum((d[:-1] * d[1:] < 0) & ((np.abs(d[:-1]) >= threshold) | (np.abs(d[1:]) >= threshold)), axis=0)
    return np.concatenate([mav, rms, wl, zc, ssc])


def mfcc_features(w: np.ndarray, n_ceps: int = 13, n_filters: int = 20) -> np.ndarray:
    """(length, channels) -> n_ceps * channels: MFCCs of the whole window, per channel.

    The window is treated as one analysis frame (Hamming), 20 mel filters over 20-500 Hz,
    coefficient 0 replaced by log frame energy.
    """
    length = len(w)
    nfft = int(2 ** np.ceil(np.log2(length)))
    out = []
    for ch in range(w.shape[1]):
        c = mfcc(w[:, ch], samplerate=FS, winlen=length / FS, winstep=length / FS, numcep=n_ceps,
                 nfilt=n_filters, nfft=nfft, lowfreq=20, highfreq=FS / 2, preemph=0.0, ceplifter=0,
                 winfunc=np.hamming)
        out.append(c[0])
    return np.concatenate(out)


def build_table(segments: list, win_ms: int = 200, step_ms: int = 100, trim_ms: int = 200):
    """Feature matrices for every window, with labels and subject/series ids."""
    length, step, trim = (int(v * FS / 1000) for v in (win_ms, step_ms, trim_ms))
    td, mf, y, subj, series = [], [], [], [], []
    for s in segments:
        for w in windows(s["signal"], length, step, trim):
            td.append(td_features(w))
            mf.append(mfcc_features(w))
            y.append(s["gesture"])
            subj.append(s["subject"])
            series.append(s["series"])
    return (np.array(td), np.array(mf), np.array(y), np.array(subj), np.array(series))
