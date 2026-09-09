"""
Stage 5 — SVM classification + evaluation.

An RBF-kernel SVM (C=10, gamma=0.01, one-vs-one) classifies the NCA-reduced features
into the six PQD classes, matching the paper. Reports accuracy, precision, recall, and
saves a confusion-matrix image to outputs/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
import config as C


def _load():
    # prefer NCA-reduced features; fall back to raw features
    try:
        d = np.load(f"{C.DATA_DIR}/features_nca.npz")
    except FileNotFoundError:
        d = np.load(f"{C.DATA_DIR}/features.npz")
    return d["F"], d["y"]


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(C.CLASSES))); ax.set_yticks(range(len(C.CLASSES)))
    ax.set_xticklabels(C.CLASSES, rotation=45, ha="right"); ax.set_yticklabels(C.CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(len(C.CLASSES)):
        for j in range(len(C.CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run():
    X, y = _load()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y)

    if C.DO_GRID_SEARCH:
        print("Grid-searching SVM hyperparameters...")
        grid = GridSearchCV(
            SVC(kernel=C.SVM_KERNEL, decision_function_shape="ovo"),
            {"C": [1, 10, 100], "gamma": [0.001, 0.01, 0.1]},
            cv=3, n_jobs=-1)
        grid.fit(Xtr, ytr)
        clf = grid.best_estimator_
        print("Best params:", grid.best_params_)
    else:
        clf = SVC(kernel=C.SVM_KERNEL, C=C.SVM_C, gamma=C.SVM_GAMMA,
                  decision_function_shape="ovo")
        clf.fit(Xtr, ytr)

    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    prec = precision_score(yte, yp, average="macro", zero_division=0)
    rec = recall_score(yte, yp, average="macro", zero_division=0)

    print("\n=== Results (test set) ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%\n")
    print(classification_report(yte, yp, target_names=C.CLASSES, zero_division=0))

    cm = confusion_matrix(yte, yp)
    plot_confusion(cm, f"{C.OUT_DIR}/confusion_matrix.png")
    with open(f"{C.OUT_DIR}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\nPrecision: {prec*100:.2f}%\nRecall: {rec*100:.2f}%\n\n")
        f.write(classification_report(yte, yp, target_names=C.CLASSES, zero_division=0))
    print(f"Saved confusion matrix + metrics to {C.OUT_DIR}/")


if __name__ == "__main__":
    run()
