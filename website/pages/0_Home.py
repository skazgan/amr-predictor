"""
AMR Predictor — Home page content.
Navigation/set_page_config is handled by the parent Home.py entrypoint.
"""
import json
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go

ROOT    = Path(__file__).parent.parent.parent
ART_DIR = ROOT / "artifacts"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css

inject_mobile_css()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 3rem 2rem; border-radius: 12px; margin-bottom: 2rem; }
    .hero h1 { color: #e94560; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; }
    .hero p  { color: #a8b2d8; font-size: 1.2rem; }
    .metric-card { background: #1e1e2e; border: 1px solid #2d2d44;
                   border-radius: 10px; padding: 1.5rem; text-align: center; }
    .metric-card .value { font-size: 2.5rem; font-weight: 700; color: #e94560; }
    .metric-card .label { color: #a8b2d8; font-size: 0.9rem; margin-top: 0.3rem; }
    .step-card { background: #1e1e2e; border-left: 4px solid #e94560;
                 border-radius: 6px; padding: 1rem 1.5rem; margin-bottom: 1rem; }
    .step-card h4 { color: #cdd6f4; margin: 0 0 0.3rem; }
    .step-card p  { color: #a8b2d8; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧬 AMR Predictor</h1>
  <p>Predicting antibiotic resistance in <em>Klebsiella pneumoniae</em>
     from whole-genome sequences using machine learning.</p>
  <p style="color:#6272a4; font-size:0.95rem; margin-top:1rem;">
    A bioinformatics + ML project — from raw DNA to clinical resistance predictions.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Key metrics ───────────────────────────────────────────────────────────────
summary = json.loads((ART_DIR / "summary.json").read_text())
stats   = json.loads((ART_DIR / "dataset_stats.json").read_text())

best_auc   = max(r["test_auc"]   for r in summary)
mean_auc   = sum(r["test_auc"]   for r in summary) / len(summary)
total_gen  = sum(r["n_resistant"] + r["n_susceptible"] for r in stats)
n_antibiotics = len(summary)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{n_antibiotics}</div>
        <div class="label">Antibiotics modelled</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{total_gen:,}</div>
        <div class="label">Labeled genomes used</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{mean_auc:.2f}</div>
        <div class="label">Mean ROC-AUC</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{best_auc:.2f}</div>
        <div class="label">Best ROC-AUC (meropenem)</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Performance overview mini-chart ───────────────────────────────────────────
st.subheader("Model performance across all antibiotics")

antibiotics = [r["antibiotic"] for r in sorted(summary, key=lambda x: x["test_auc"])]
aucs        = [r["test_auc"]   for r in sorted(summary, key=lambda x: x["test_auc"])]
colors      = ["#e94560" if a >= 0.80 else "#ffb86c" if a >= 0.75 else "#8be9fd"
               for a in aucs]

fig = go.Figure(go.Bar(
    x=aucs, y=antibiotics, orientation="h",
    marker_color=colors,
    text=[f"{a:.3f}" for a in aucs], textposition="outside",
))
fig.add_vline(x=0.5, line_dash="dot", line_color="#6272a4",
              annotation_text="Random baseline (0.50)")
fig.add_vline(x=0.8, line_dash="dash", line_color="#50fa7b",
              annotation_text="Good threshold (0.80)")
fig.update_layout(
    xaxis=dict(range=[0.4, 1.0], title="ROC-AUC (test set)"),
    height=320, margin=dict(l=10, r=80, t=20, b=40),
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    font_color="#cdd6f4",
    xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
)
st.plotly_chart(fig, use_container_width=True)

# ── Project pipeline steps ────────────────────────────────────────────────────
st.subheader("How the project works — step by step")

steps = [
    ("1 — The Biology",        "Bacteria develop resistance to antibiotics through DNA mutations and gene acquisition. We treat resistance prediction as a classification problem."),
    ("2 — The Data",           "26,000+ K. pneumoniae genomes from BV-BRC (74 countries, 2000–2024). Each genome has Resistant / Susceptible labels for up to 10 antibiotics."),
    ("3 — Feature Extraction", "Two feature types: (a) k-mer counts — all 6-letter DNA substrings; (b) resistance gene presence/absence from BV-BRC specialty genes database."),
    ("4 — Model Training",     "XGBoost + Random Forest soft-voting ensemble with isotonic calibration, trained with 5-fold cross-validation. One model per antibiotic (10 total)."),
    ("5 — Prediction",         "Three predictors: Live (BV-BRC genome ID), Offline (gene toggles), FASTA Upload (raw assembly). All output calibrated resistance probabilities + PDF report."),
    ("6 — Explainability",     "SHAP values show which resistance genes and k-mer patterns drove each prediction. Co-resistance network reveals which drugs fail together."),
    ("7–13 — Epidemiology",    "Gene emergence curves, MDR trends over 24 years, country-level resistance maps, 5-year resistance forecasts, and MLST lineage tracking."),
]

for title, desc in steps:
    st.markdown(f"""<div class="step-card">
        <h4>{title}</h4><p>{desc}</p></div>""", unsafe_allow_html=True)

st.markdown("""
---
<p style="color:#6272a4; font-size:0.85rem;">
  Navigate the pages in the sidebar to explore each step in detail.
  Use <strong>Live Predictor</strong> to test a new genome.
</p>
""", unsafe_allow_html=True)
