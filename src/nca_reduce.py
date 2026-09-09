"""
Stage 4 — NCA feature selection / dimensionality reduction.

Neighborhood Component Analysis reduces the concatenated deep-feature vector to the
NCA_COMPONENTS (300) most discriminative dimensions, as in the paper, improving class
separability before the SVM.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import config as C


def build():
    d = np.load(f"{C.DATA_DIR}/features.npz")
    F, y = d["F"], d["y"]

    F = StandardScaler().fit_transform(F)

    n_comp = min(C.NCA_COMPONENTS, F.shape[1], F.shape[0] - 1)
    print(f"Fitting NCA: {F.shape[1]} -> {n_comp} dimensions "
          f"(this can take a few minutes)...")
    nca = NeighborhoodComponentsAnalysis(
        n_components=n_comp, max_iter=C.NCA_MAX_ITER,
        random_state=C.RANDOM_STATE, verbose=1)
    Fr = nca.fit_transform(F, y)

    out = f"{C.DATA_DIR}/features_nca.npz"
    np.savez_compressed(out, F=Fr, y=y)
    print(f"Reduced features {Fr.shape} -> {out}")


if __name__ == "__main__":
    build()
