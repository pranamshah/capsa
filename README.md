# Power Quality Disturbance Classification using Hybrid Deep Learning
### CWT Scalograms + ResNet-50 + VGG-16 + NCA + SVM

A faithful software implementation of the method in:

> **P. Mahendra Peruman & K. Ayyar**, "Power quality disturbance identification using
> hybrid deep learning in renewable energy systems," *Scientific Reports* (Nature),
> vol. 15, art. 44592, 2025. DOI: 10.1038/s41598-025-28291-0
> Open-access PDF: https://www.nature.com/articles/s41598-025-28291-0.pdf

This is a **software-only** project for the subject *Computer-Aided Power Signal Analysis*.
It analyses electrical power-signal disturbances (voltage sag, swell, harmonics, transient,
interruption, normal) and classifies them with the paper's exact hybrid pipeline.

---

## 1. The exact pipeline (matches the paper stage-for-stage)

```
 1. Signal generation      six PQD classes as 1-D voltage signals (parametric equations)
        |
 2. CWT scalograms         each signal -> Morlet CWT -> 224x224 RGB time-frequency image
        |
 3. Deep feature extraction ResNet-50 (2048-d) + VGG-16 (4096-d) on the scalograms,
        |                    features concatenated -> 6144-d vector
 4. NCA feature selection   reduce 6144 -> 300 most discriminative features
        |
 5. SVM classification      RBF kernel, C=10, gamma=0.01, one-vs-one multi-class
        |
    accuracy / precision / recall + confusion matrix
```

Paper's headline result: 98.54% accuracy (IEEE 9-bus). Your run will differ because the
signals are generated in software rather than from the authors' private Simulink dataset
(see the honesty note at the bottom).

## 2. Files

| File | Stage | What it does |
|---|---|---|
| `src/signal_generator.py` | 1 | Generates the six PQD classes as 1-D signals with noise |
| `src/cwt_scalogram.py` | 2 | Continuous Wavelet Transform → 224×224 scalogram images |
| `src/feature_extractor.py` | 3 | ResNet-50 + VGG-16 deep features (ImageNet weights) |
| `src/nca_reduce.py` | 4 | NCA dimensionality reduction (6144 → 300) |
| `src/svm_classifier.py` | 5 | RBF-SVM with grid search + evaluation |
| `src/pipeline.py` | all | Runs the whole thing end to end |
| `src/config.py` | — | All parameters in one place (matches the paper's settings) |
| `requirements.txt` | — | Python dependencies |

## 3. How to run (hand this to Claude Code)

```bash
pip install -r requirements.txt

# Option A: run the full pipeline in one command
python src/pipeline.py

# Option B: run stage by stage (useful for the report / understanding)
python src/signal_generator.py     # -> data/signals.npz
python src/cwt_scalogram.py        # -> data/scalograms/*.png  + data/scalograms.npz
python src/feature_extractor.py    # -> data/features.npz
python src/nca_reduce.py           # -> data/features_nca.npz
python src/svm_classifier.py       # -> outputs/ (confusion matrix, metrics)
```

Outputs (confusion matrix image, per-class metrics, accuracy/precision/recall) are written
to `outputs/`.

## 4. Parameters (from the paper — set in `src/config.py`)

- Classes: sag, swell, harmonics, transient, interruption, normal
- Samples per class: 2000 (paper uses 2000/class, 12000 total) — lower it in config for a fast first run
- Sampling: 10 kHz, fundamental 50 Hz
- Wavelet: Morlet (CWT)
- Scalogram size: 224 × 224 × 3
- CNNs: ResNet-50 (2048-d) + VGG-16 (4096-d), ImageNet weights, features concatenated
- NCA: reduce to 300 features
- SVM: RBF kernel, C = 10, gamma = 0.01, one-vs-one
- Split: 50% train / 50% test (as in the paper)

## 5. Suggested tuning for a first run
Deep features on 12000 images need a GPU to be quick. For a fast CPU test, in
`src/config.py` set `SAMPLES_PER_CLASS = 100` and `USE_VGG = False` first; once it works
end to end, scale back up toward the paper's numbers.

## 6. Honesty note (for your report / viva)
The paper's disturbance signals come from modified IEEE 9-bus/13-bus MATLAB/Simulink
models, which are not publicly released. This project generates equivalent PQD signals in
software using the standard parametric equations for each disturbance class — the accepted
way to reproduce such studies. Everything downstream (CWT scalograms → ResNet-50 + VGG-16 →
NCA → RBF-SVM) follows the paper's method exactly. State this plainly: the *method* is the
paper's; the *signal source* is software-generated because their dataset is private.
