"""
Frequency-domain feature extraction. The ESP32 sends time-domain summary features
(rms, peak, std) per window; this module is for deeper offline/server-side analysis
if you also stream raw samples, or for computing an FFT spectrum for the dashboard
from a buffered signal.

If you extend the firmware to publish raw sample arrays, feed them here to get the
FFT spectrum and spectral features (dominant frequency, spectral centroid, band energy).
"""

import numpy as np


def compute_fft_spectrum(signal, sample_rate):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    magnitude = np.abs(np.fft.rfft(signal)) / len(signal)
    return freqs, magnitude


def spectral_features(signal, sample_rate):
    freqs, mag = compute_fft_spectrum(signal, sample_rate)
    total_energy = np.sum(mag) + 1e-12
    dominant_freq = float(freqs[int(np.argmax(mag))])
    spectral_centroid = float(np.sum(freqs * mag) / total_energy)
    return {
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "total_spectral_energy": float(total_energy),
    }


def time_features(signal):
    signal = np.asarray(signal, dtype=float)
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    std = float(np.std(signal))
    kurt = float(((signal - signal.mean()) ** 4).mean() / (signal.std() ** 4 + 1e-12))
    return {"rms": rms, "peak": peak, "std": std, "kurtosis": kurt}
