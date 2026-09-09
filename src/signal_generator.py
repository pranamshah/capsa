"""
Stage 1 — PQD signal generation.

Generates the six power-quality-disturbance classes as 1-D voltage signals using the
standard parametric equations (IEEE 1159 style). Saves them to data/signals.npz.

This replaces the paper's private MATLAB/Simulink IEEE-bus dataset with equivalent
software-generated signals — the accepted way to reproduce such studies.
"""

import numpy as np
import config as C


def _t():
    n = int(C.FS * C.CYCLES / C.F0)
    return np.linspace(0, C.CYCLES / C.F0, n, endpoint=False)


def _noise(x):
    if C.NOISE_SNR_DB is None:
        return x
    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (C.NOISE_SNR_DB / 10))
    return x + np.random.normal(0, np.sqrt(p_noise), size=x.shape)


def gen_normal(t):
    return np.sin(2 * np.pi * C.F0 * t)


def gen_sag(t):
    x = np.sin(2 * np.pi * C.F0 * t)
    a = np.random.uniform(0.1, 0.9)                 # remaining amplitude (0.1-0.9 pu)
    s, e = sorted(np.random.uniform(0.1, 0.9, 2) * len(t))
    m = np.ones_like(t); m[int(s):int(e)] = a
    return x * m


def gen_swell(t):
    x = np.sin(2 * np.pi * C.F0 * t)
    a = np.random.uniform(1.1, 1.8)                 # swell amplitude (1.1-1.8 pu)
    s, e = sorted(np.random.uniform(0.1, 0.9, 2) * len(t))
    m = np.ones_like(t); m[int(s):int(e)] = a
    return x * m


def gen_interruption(t):
    x = np.sin(2 * np.pi * C.F0 * t)
    a = np.random.uniform(0.0, 0.1)                 # near-zero during interruption
    s, e = sorted(np.random.uniform(0.1, 0.9, 2) * len(t))
    m = np.ones_like(t); m[int(s):int(e)] = a
    return x * m


def gen_harmonics(t):
    x = np.sin(2 * np.pi * C.F0 * t)
    x += np.random.uniform(0.05, 0.15) * np.sin(2 * np.pi * 3 * C.F0 * t)
    x += np.random.uniform(0.05, 0.15) * np.sin(2 * np.pi * 5 * C.F0 * t)
    x += np.random.uniform(0.02, 0.10) * np.sin(2 * np.pi * 7 * C.F0 * t)
    return x


def gen_transient(t):
    x = np.sin(2 * np.pi * C.F0 * t)
    pos = np.random.uniform(0.2, 0.8) * len(t)
    idx = int(pos)
    width = max(2, int(0.002 * C.FS))               # ~2 ms transient
    amp = np.random.uniform(0.5, 2.0)
    decay = np.exp(-np.linspace(0, 5, width))
    end = min(len(t), idx + width)
    x[idx:end] += amp * decay[: end - idx] * np.sin(2 * np.pi * 900 * t[idx:end])
    return x


GEN = {
    "normal": gen_normal, "sag": gen_sag, "swell": gen_swell,
    "interruption": gen_interruption, "harmonics": gen_harmonics, "transient": gen_transient,
}


def build():
    t = _t()
    X, y = [], []
    for label, cls in enumerate(C.CLASSES):
        for _ in range(C.SAMPLES_PER_CLASS):
            sig = _noise(GEN[cls](t.copy()))
            X.append(sig.astype(np.float32))
            y.append(label)
    X = np.array(X); y = np.array(y)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm], t


if __name__ == "__main__":
    np.random.seed(C.RANDOM_STATE)
    X, y, t = build()
    out = f"{C.DATA_DIR}/signals.npz"
    np.savez_compressed(out, X=X, y=y, t=t, classes=C.CLASSES)
    print(f"Generated {len(X)} signals ({C.SAMPLES_PER_CLASS}/class) -> {out}")
    print(f"Signal length = {X.shape[1]} samples")
