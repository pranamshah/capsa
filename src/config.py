"""
Central configuration — all parameters in one place.
Values match the paper (Peruman & Ayyar, Sci. Rep. 2025) where applicable.
Lower SAMPLES_PER_CLASS / set USE_VGG=False for a fast first run on CPU.
"""

import os

# ---- paths ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SCALO_DIR = os.path.join(DATA_DIR, "scalograms")
OUT_DIR = os.path.join(ROOT, "outputs")
for d in (DATA_DIR, SCALO_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

# ---- classes (the six PQD types in the paper) ----
CLASSES = ["normal", "sag", "swell", "harmonics", "transient", "interruption"]

# ---- signal generation ----
SAMPLES_PER_CLASS = 2000     # paper: 2000/class (12000 total). Use 100 for a quick test.
FS = 10000                   # sampling frequency (Hz), paper: 10 kHz
F0 = 50.0                    # fundamental frequency (Hz)
CYCLES = 10                  # number of fundamental cycles per signal window
NOISE_SNR_DB = 30            # additive white gaussian noise level

# ---- CWT scalogram ----
WAVELET = "morl"             # Morlet wavelet (paper's choice)
SCALO_SIZE = 224             # 224x224 to match ResNet/VGG input
CWT_MAX_SCALE = 128          # number of scales in the CWT

# ---- deep feature extraction ----
USE_RESNET = True
USE_VGG = True               # set False for a faster first run

# ---- NCA ----
NCA_COMPONENTS = 300         # paper reduces to 300 discriminative features
NCA_MAX_ITER = 50

# ---- SVM ----
SVM_C = 10.0                 # paper: C = 10
SVM_GAMMA = 0.01             # paper: gamma = 0.01
SVM_KERNEL = "rbf"
DO_GRID_SEARCH = False       # True to re-tune C/gamma via cross-validation

# ---- split ----
TEST_SIZE = 0.5              # paper: 50/50 train/test
RANDOM_STATE = 42
