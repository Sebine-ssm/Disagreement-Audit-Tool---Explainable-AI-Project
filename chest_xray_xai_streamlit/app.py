import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
from continuous_analysis import render_continuous_analysis

st.set_page_config(
    page_title="Chest X-ray XAI Disagreement Audit",
    page_icon="🩻",
    layout="wide",
)

# Presentation-only styling. All analyses below remain unchanged.
st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 1.6rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] .block-container {padding-top: 1.4rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 12px;
    padding: .85rem 1rem;
}
div[data-testid="stTabs"] button[role="tab"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🩻 XAI Disagreement Audit")
    st.caption("ResNet-50 · RSNA chest X-rays")
    st.divider()
    st.markdown("**Explore the research**")
    st.caption(
        "Use the four tabs in the main panel to explore disagreement, "
        "continuous relationships, statistical evidence, and the case study."
    )
    st.markdown("**How to read the results**")
    st.caption(
        "The study metrics at the top are fixed full-sample results. "
        "Filters below them update the exploratory analyses, not the "
        "original permutation test or published summary findings."
    )
    st.divider()
    st.caption("Academic research demonstration · Not for clinical use")

st.title("When Stable Explanations Disagree")
st.markdown("#### A Stability–Localization Disagreement Audit for Chest X-ray Classification")
st.markdown(
    "Can an explanation remain similar after an image corruption while becoming "
    "**less aligned with the radiologist-annotated opacity region**? "
    "This dashboard investigates that question using a ResNet-50 classifier "
    "and three explanation methods: **Integrated Gradients**, "
    "**Occlusion Sensitivity**, and **Score-CAM**."
)

DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "stability_localization_public.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Verify the original, fixed disagreement definition.
baseline_disagreement = (
    (df["explanation_similarity"] >= 0.80)
    & (df["alignment_change"] <= -0.05)
)
assert (
    baseline_disagreement == df["disagreement"].astype(bool)
).all(), "Baseline disagreement labels do not match!"

st.markdown("### Study at a glance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Evaluations", f"{len(df):,}")
m2.metric("Unique chest X-rays", f"{df['sop_uid'].nunique():,}")
m3.metric("Disagreement events", f"{int(baseline_disagreement.sum()):,}")
m4.metric("Disagreement rate", f"{baseline_disagreement.mean():.2%}")
st.caption(
    "Fixed full-sample summary · Explanation similarity ≥ 0.80 and "
    "localization alignment decrease ≥ 0.05. Multiple evaluations come "
    "from each image; these are not independent patient observations."
)

with st.expander("Study design and metric definitions", expanded=False):
    st.markdown(
        "**Study design:** Opacity-positive chest X-rays from the RSNA "
        "Pneumonia Detection Challenge were evaluated using three XAI methods "
        "across six image-corruption conditions. Each record represents an "
        "image–method–corruption evaluation.\n\n"
        "**Explanation similarity:** Spearman correlation between the original "
        "and corrupted explanation maps.\n\n"
        "**Localization alignment:** Spatial alignment of an explanation with "
        "the annotated opacity region. **Alignment change** is corrupted "
        "alignment minus original alignment; negative values indicate loss.\n\n"
