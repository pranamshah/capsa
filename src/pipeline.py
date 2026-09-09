"""
Runs the full hybrid pipeline end to end:
signal generation -> CWT scalograms -> ResNet/VGG features -> NCA -> SVM -> evaluation.

    python src/pipeline.py
"""

import numpy as np
import config as C
import signal_generator, cwt_scalogram, feature_extractor, nca_reduce, svm_classifier


def main():
    np.random.seed(C.RANDOM_STATE)

    print("\n[1/5] Generating PQD signals...")
    X, y, t = signal_generator.build()
    np.savez_compressed(f"{C.DATA_DIR}/signals.npz", X=X, y=y, t=t, classes=C.CLASSES)

    print("\n[2/5] Building CWT scalograms...")
    cwt_scalogram.build()

    print("\n[3/5] Extracting deep features (ResNet-50 / VGG-16)...")
    feature_extractor.build()

    print("\n[4/5] NCA feature reduction...")
    nca_reduce.build()

    print("\n[5/5] SVM classification + evaluation...")
    svm_classifier.run()

    print("\nDone. See outputs/ for the confusion matrix and metrics.")


if __name__ == "__main__":
    main()
