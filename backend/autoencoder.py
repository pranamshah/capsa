"""
Dense Autoencoder for vibration anomaly detection via reconstruction error.
Train on NORMAL feature windows only; flag windows whose reconstruction error
exceeds a learned threshold. This is the deep-learning layer of the pipeline.

Usage:
    python autoencoder.py --train    # fit on current normal data, save model
    python autoencoder.py --score    # score recent windows, write flags to DB
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from db import fetch_recent, update_flags

MODEL_PATH = "vib_autoencoder.h5"
THRESHOLD_PATH = "vib_ae_threshold.txt"
SCALER_PATH = "vib_scaler.npz"


def build_autoencoder(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(8, activation="relu")(inputs)
    x = layers.Dense(3, activation="relu")(x)          # bottleneck
    x = layers.Dense(8, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def get_matrix(rows):
    return np.array([[r["rms"], r["peak"], r["std"]] for r in rows], dtype=float)


def train():
    rows = fetch_recent(n=5000)
    if len(rows) < 30:
        print(f"Not enough data ({len(rows)} rows). Collect more NORMAL data first.")
        return

    X = get_matrix(rows)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xn = (X - mean) / std

    model = build_autoencoder(X.shape[1])
    model.fit(Xn, Xn, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

    recon = model.predict(Xn, verbose=0)
    errors = np.mean(np.square(Xn - recon), axis=1)
    threshold = float(errors.mean() + 3 * errors.std())

    model.save(MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))
    np.savez(SCALER_PATH, mean=mean, std=std)
    print(f"Saved model, threshold={threshold:.5f}")


def score():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        threshold = float(f.read().strip())
    scaler = np.load(SCALER_PATH)
    mean, std = scaler["mean"], scaler["std"]

    rows = fetch_recent(n=300)
    if not rows:
        print("No data to score.")
        return

    X = get_matrix(rows)
    Xn = (X - mean) / std
    recon = model.predict(Xn, verbose=0)
    errors = np.mean(np.square(Xn - recon), axis=1)
    flags = (errors > threshold).astype(int)

    for row, flag in zip(rows, flags):
        update_flags(row["id"], autoencoder_flag=int(flag))

    print(f"Autoencoder scored {len(rows)} windows, {flags.sum()} anomalies.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    elif args.score:
        score()
    else:
        print("Pass --train or --score")
