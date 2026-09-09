"""
Web dashboard for the Hybrid Deep-Learning PQD Classification project.

Presents the whole project as a website: what it does, the pipeline, live
signal/scalogram previews, a one-click pipeline runner, and the results
(confusion matrix + metrics).

Run (accessible on your private/LAN IP):
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Then open  http://<your-private-ip>:8501  from any device on the same network.
"""

import io
import os
import sys
import time

import numpy as np
import streamlit as st

# make the flat modules in src/ importable (they do `import config`, etc.)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config as C  # noqa: E402
import signal_generator as sg  # noqa: E402

st.set_page_config(page_title="Hybrid PQD Classifier", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------- sidebar ----
st.sidebar.title("⚡ PQD Classifier")
st.sidebar.caption("Hybrid deep learning: CWT + ResNet-50 + VGG-16 + NCA + SVM")

st.sidebar.subheader("Run settings")
spc = st.sidebar.slider("Samples per class", 20, 2000, 100, step=20,
                        help="Paper uses 2000/class. Lower = much faster on CPU.")
use_vgg = st.sidebar.checkbox("Use VGG-16 (slower)", value=False,
                              help="ResNet-50 is always used. Adding VGG doubles feature time.")
grid = st.sidebar.checkbox("Grid-search SVM", value=False)

# apply overrides to the shared config module so every stage sees them
C.SAMPLES_PER_CLASS = spc
C.USE_VGG = use_vgg
C.DO_GRID_SEARCH = grid

page = st.sidebar.radio(
    "Section",
    ["Overview", "Signal explorer", "Scalograms", "Run pipeline", "Results"],
)

# --------------------------------------------------------------- helpers -----
@st.cache_data(show_spinner=False)
def make_signal(cls: str, seed: int = 0):
    np.random.seed(seed)
    t = sg._t()
    fn = getattr(sg, f"gen_{cls}")
    return t, sg._noise(fn(t))


def scalogram_image(sig):
    import cwt_scalogram as cw
    return cw.signal_to_scalogram(sig)


# -------------------------------------------------------------- Overview -----
if page == "Overview":
    st.title("Power Quality Disturbance Classification")
    st.subheader("Hybrid Deep Learning — CWT Scalograms + ResNet-50 + VGG-16 + NCA + SVM")

    st.markdown(
        """
This is a **software-only** reproduction of the method in
**Peruman & Ayyar, *Scientific Reports* (Nature), 2025** — classifying six kinds of
electrical **power-quality disturbances** from voltage signals.

**What it does, end to end:**
"""
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.info("**1. Signals**\n\nGenerate 6 PQD classes as 1-D voltage waveforms")
    c2.info("**2. CWT**\n\nMorlet wavelet → 224×224 scalogram images")
    c3.info("**3. Deep features**\n\nResNet-50 + VGG-16 → 6144-d vector")
    c4.info("**4. NCA**\n\nReduce 6144 → 300 best features")
    c5.info("**5. SVM**\n\nRBF-SVM classifies into 6 classes")

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        st.markdown("### The six classes")
        st.markdown(
            "- **normal** — clean 50 Hz sine\n"
            "- **sag** — temporary voltage drop\n"
            "- **swell** — temporary voltage rise\n"
            "- **harmonics** — distorted waveform\n"
            "- **transient** — brief spike/oscillation\n"
            "- **interruption** — voltage nearly zero"
        )
    with colB:
        st.markdown("### Key parameters (from the paper)")
        st.markdown(
            f"- Sampling: **{C.FS/1000:.0f} kHz**, fundamental **{C.F0:.0f} Hz**\n"
            f"- Scalogram: **{C.SCALO_SIZE}×{C.SCALO_SIZE}**, wavelet **{C.WAVELET}**\n"
            f"- NCA: reduce to **{C.NCA_COMPONENTS}** features\n"
            f"- SVM: RBF, **C={C.SVM_C}**, **γ={C.SVM_GAMMA}**\n"
            f"- Split: **{int((1-C.TEST_SIZE)*100)}/{int(C.TEST_SIZE*100)}** train/test"
        )

    st.divider()
    st.markdown(
        "> **Honesty note for your report/viva:** the paper's signals come from a private "
        "IEEE-bus Simulink dataset. Here the *method* is identical; the *signals* are "
        "generated in software using standard parametric PQD equations — the accepted way "
        "to reproduce such studies."
    )
    st.success("Use the sidebar to explore signals, view scalograms, run the pipeline, and see results.")

# ------------------------------------------------------- Signal explorer -----
elif page == "Signal explorer":
    st.title("Signal explorer")
    st.caption("The raw 1-D voltage waveforms that feed the pipeline (Stage 1).")

    cls = st.selectbox("Disturbance class", C.CLASSES)
    seed = st.number_input("Random seed (changes the noise realisation)", 0, 9999, 0)
    t, sig = make_signal(cls, int(seed))

    st.line_chart({"voltage (p.u.)": sig})
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(sig))
    c2.metric("Duration", f"{t[-1]*1000:.1f} ms")
    c3.metric("RMS", f"{np.sqrt(np.mean(sig**2)):.3f}")

    st.markdown("#### All six classes at a glance")
    cols = st.columns(3)
    for i, k in enumerate(C.CLASSES):
        _, s = make_signal(k, int(seed))
        with cols[i % 3]:
            st.caption(k)
            st.line_chart({k: s}, height=140)

# ------------------------------------------------------------ Scalograms -----
elif page == "Scalograms":
    st.title("CWT scalograms")
    st.caption("Stage 2 — each signal becomes a time-frequency image (this is what the CNNs see).")

    seed = st.number_input("Random seed", 0, 9999, 0)
    if st.button("Generate scalograms for all six classes", type="primary"):
        cols = st.columns(3)
        for i, k in enumerate(C.CLASSES):
            _, sig = make_signal(k, int(seed))
            with st.spinner(f"CWT for {k}..."):
                img = scalogram_image(sig)
            with cols[i % 3]:
                st.image(img, caption=k, use_container_width=True)
    else:
        st.info("Click the button above to compute Morlet-CWT scalograms live.")

# ---------------------------------------------------------- Run pipeline -----
elif page == "Run pipeline":
    st.title("Run the full pipeline")
    st.caption("Signal generation → CWT → deep features → NCA → SVM → evaluation.")

    st.warning(
        f"Current run size: **{C.SAMPLES_PER_CLASS} samples/class** "
        f"({C.SAMPLES_PER_CLASS*len(C.CLASSES)} total), VGG **{'on' if C.USE_VGG else 'off'}**. "
        "Adjust in the sidebar. Deep-feature extraction needs TensorFlow and is the slow part."
    )

    if st.button("▶ Run pipeline now", type="primary"):
        try:
            import cwt_scalogram, feature_extractor, nca_reduce, svm_classifier
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not import a stage module: {e}")
            st.stop()

        status = st.status("Running pipeline...", expanded=True)
        try:
            np.random.seed(C.RANDOM_STATE)

            status.write("**[1/5]** Generating PQD signals...")
            X, y, t = sg.build()
            np.savez_compressed(f"{C.DATA_DIR}/signals.npz", X=X, y=y, t=t, classes=C.CLASSES)

            status.write("**[2/5]** Building CWT scalograms...")
            cwt_scalogram.build()

            status.write("**[3/5]** Extracting deep features (ResNet-50 / VGG-16)...")
            feature_extractor.build()

            status.write("**[4/5]** NCA feature reduction...")
            nca_reduce.build()

            status.write("**[5/5]** SVM classification + evaluation...")
            svm_classifier.run()

            status.update(label="Pipeline complete ✔", state="complete")
            st.success("Done! Open the **Results** section to see the confusion matrix and metrics.")
        except ModuleNotFoundError as e:
            status.update(label="Missing dependency", state="error")
            st.error(f"A dependency is missing: `{e.name}`. Install it with "
                     f"`pip install -r requirements.txt` (TensorFlow is required for Stage 3).")
        except Exception as e:  # noqa: BLE001
            status.update(label="Pipeline failed", state="error")
            st.exception(e)
    else:
        st.info("Set your run size in the sidebar, then click **Run pipeline now**.")

# --------------------------------------------------------------- Results -----
elif page == "Results":
    st.title("Results")
    cm_path = os.path.join(C.OUT_DIR, "confusion_matrix.png")
    metrics_path = os.path.join(C.OUT_DIR, "metrics.txt")

    if not os.path.exists(metrics_path) and not os.path.exists(cm_path):
        st.info("No results yet. Go to **Run pipeline** and run it first.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            if os.path.exists(cm_path):
                st.subheader("Confusion matrix")
                st.image(cm_path, use_container_width=True)
            else:
                st.info("No confusion matrix image found.")
        with col2:
            if os.path.exists(metrics_path):
                st.subheader("Metrics")
                with open(metrics_path) as f:
                    text = f.read()
                for line in text.splitlines():
                    if line.startswith("Accuracy"):
                        st.metric("Accuracy", line.split(":", 1)[1].strip())
                    elif line.startswith("Precision"):
                        st.metric("Precision", line.split(":", 1)[1].strip())
                    elif line.startswith("Recall"):
                        st.metric("Recall", line.split(":", 1)[1].strip())
                st.code(text, language="text")
        st.caption(f"Files read from: {C.OUT_DIR}")
