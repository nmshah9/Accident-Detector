"""
Accident Detection from CCTV Footage — Streamlit App
Predicts whether a CCTV frame shows an Accident or Non Accident scene,
using the MobileNetV2 transfer-learning model (Model 2 from the
end-to-end notebook, 92% test accuracy).
"""

import streamlit as st
from PIL import Image
import numpy as np

from src.model_utils import load_model, predict, make_gradcam_heatmap, overlay_heatmap

st.set_page_config(
    page_title="Accident Detection — CCTV Frame Classifier",
    page_icon="🚦",
    layout="centered",
)
# ============================================================
# LOAD BANNER
# ============================================================
from PIL import Image
banner = Image.open("banner.png")

# ============================================================
# DISPLAY BANNER
# ============================================================

st.image(banner, use_container_width=True)

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .result-accident {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 0.5rem;
        padding: 1.2rem;
        text-align: center;
    }
    .result-nonaccident {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 0.5rem;
        padding: 1.2rem;
        text-align: center;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .result-accident .result-label { color: #b91c1c; }
    .result-nonaccident .result-label { color: #15803d; }
    .confidence-text {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- header
st.markdown('<div class="main-header">🚦 Accident Detection from CCTV Footage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a CCTV frame to classify it as <b>Accident</b> or '
    '<b>Non Accident</b>, using a MobileNetV2 transfer-learning model (92% test accuracy).</div>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ About this model", expanded=False):
    st.markdown("""
This app uses **Model 2** from the accompanying analysis notebook — MobileNetV2
pretrained on ImageNet, fine-tuned as a classification head on ~790 CCTV frames.

| Metric | Value |
|---|---|
| Test Accuracy | 92.0% |
| Accident Recall | 91.5% |
| Architecture | MobileNetV2 (frozen backbone) + Dense head |

Three approaches were compared in the notebook: a CNN trained from scratch (53% test
accuracy — collapsed to majority-class prediction), this transfer-learning model (92%),
and transfer learning with data augmentation (73%, under-converged in that run). This
plain transfer-learning model was the clear winner and is what powers this app.

**Note:** This is a research/demo model trained on a few hundred images, not a validated
safety system — don't rely on it as the sole basis for real accident alerting.
""")

# ---------------------------------------------------------------- model loading
@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
    model_load_error = None
except Exception as e:
    model = None
    model_load_error = str(e)

if model_load_error:
    st.error(f"Could not load the model: {model_load_error}")
    st.stop()

# ---------------------------------------------------------------- upload & predict
uploaded_file = st.file_uploader(
    "Upload a CCTV frame (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
)

show_heatmap = st.checkbox("Show Grad-CAM heatmap (which region drove the prediction)", value=True)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Uploaded frame", use_container_width=True)

    with st.spinner("Analyzing frame..."):
        label, confidence, raw_score = predict(model, pil_image)

        heatmap_img = None
        if show_heatmap:
            try:
                heatmap = make_gradcam_heatmap(model, pil_image)
                heatmap_img = overlay_heatmap(pil_image, heatmap)
            except Exception as e:
                st.warning(f"Grad-CAM heatmap unavailable for this image ({e}).")

    with col2:
        if heatmap_img is not None:
            st.image(heatmap_img, caption="Grad-CAM: model attention", use_container_width=True)
        else:
            st.empty()

    st.markdown("---")

    css_class = "result-accident" if label == "Accident" else "result-nonaccident"
    icon = "🚨" if label == "Accident" else "✅"
    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-label">{icon} {label}</div>
        <div class="confidence-text">Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence / 100)

    with st.expander("Raw model output"):
        st.write(f"Sigmoid output (P(Non Accident)): `{raw_score:.4f}`")
        st.write(f"Predicted class: `{label}`")
        st.write(f"Confidence: `{confidence:.2f}%`")

else:
    st.info("👆 Upload a CCTV frame to get a prediction.")

st.markdown("---")
st.caption("Built by Nirav Shah · Model: MobileNetV2 transfer learning · Dataset: Accident Detection from CCTV Footage (Kaggle)")
