"""
Stage 2 — CWT scalograms.

Each 1-D PQD signal is transformed with the Continuous Wavelet Transform (Morlet
wavelet) into a 2-D time-frequency scalogram, saved as a 224x224x3 RGB image (matching
the ResNet/VGG input size), exactly as the paper describes.
"""

import numpy as np
import pywt
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import config as C


def signal_to_scalogram(sig):
    scales = np.arange(1, C.CWT_MAX_SCALE + 1)
    coeffs, _ = pywt.cwt(sig, scales, C.WAVELET, sampling_period=1.0 / C.FS)
    mag = np.abs(coeffs)
    # normalise to 0-1
    mag = (mag - mag.min()) / (np.ptp(mag) + 1e-12)
    # apply a colormap (jet) -> RGB, like the paper's colored scalograms
    rgb = (cm.jet(mag)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((C.SCALO_SIZE, C.SCALO_SIZE))
    return np.asarray(img, dtype=np.uint8)


def build(save_images=True, max_images_to_save=60):
    d = np.load(f"{C.DATA_DIR}/signals.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    scalos = np.zeros((len(X), C.SCALO_SIZE, C.SCALO_SIZE, 3), dtype=np.uint8)
    for i, sig in enumerate(X):
        scalos[i] = signal_to_scalogram(sig)
        if save_images and i < max_images_to_save:
            Image.fromarray(scalos[i]).save(
                f"{C.SCALO_DIR}/{C.CLASSES[y[i]]}_{i}.png")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(X)} scalograms")
    out = f"{C.DATA_DIR}/scalograms.npz"
    np.savez_compressed(out, scalos=scalos, y=y)
    print(f"Saved {len(scalos)} scalograms -> {out}")
    print(f"Sample images (up to {max_images_to_save}) in {C.SCALO_DIR}/")


if __name__ == "__main__":
    build()
