"""
Stage 3 — Deep feature extraction.

Feeds the CWT scalogram images through ResNet-50 (2048-d) and VGG-16 (4096-d),
both pretrained on ImageNet, and concatenates their features (6144-d) per image,
exactly as the paper does.
"""

import numpy as np
import config as C


def _load_models():
    from tensorflow.keras.applications import ResNet50, VGG16
    from tensorflow.keras.models import Model
    models = {}
    if C.USE_RESNET:
        base = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                        input_shape=(C.SCALO_SIZE, C.SCALO_SIZE, 3))
        models["resnet"] = Model(base.input, base.output)  # 2048-d
    if C.USE_VGG:
        base = VGG16(weights="imagenet", include_top=False, pooling="avg",
                     input_shape=(C.SCALO_SIZE, C.SCALO_SIZE, 3))
        models["vgg"] = Model(base.input, base.output)     # 512-d (avg-pooled)
    return models


def build():
    from tensorflow.keras.applications.resnet50 import preprocess_input as pre_res
    from tensorflow.keras.applications.vgg16 import preprocess_input as pre_vgg

    d = np.load(f"{C.DATA_DIR}/scalograms.npz")
    scalos, y = d["scalos"], d["y"]
    models = _load_models()

    feats = []
    if "resnet" in models:
        x = pre_res(scalos.astype("float32").copy())
        print("Extracting ResNet-50 features...")
        feats.append(models["resnet"].predict(x, batch_size=32, verbose=1))
    if "vgg" in models:
        x = pre_vgg(scalos.astype("float32").copy())
        print("Extracting VGG-16 features...")
        feats.append(models["vgg"].predict(x, batch_size=32, verbose=1))

    F = np.concatenate(feats, axis=1)
    out = f"{C.DATA_DIR}/features.npz"
    np.savez_compressed(out, F=F, y=y)
    print(f"Feature matrix {F.shape} -> {out}")


if __name__ == "__main__":
    build()
