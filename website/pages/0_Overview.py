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
from utils import inject_mobile_css, page_info_expander

inject_mobile_css()
page_info_expander("""
**AMR** — Antimicrobial Resistance. When bacteria develop the ability to survive antibiotic treatment that would normally kill them.

**WGS** — Whole Genome Sequencing. Reading the complete DNA code of a bacterial isolate (≈5 million letters for *K. pneumoniae*).

**ML / Machine Learning** — Computer algorithms that learn patterns from labelled examples. Here: the model learns which genes predict resistance, then applies that to new genomes.

**AUC** — Area Under the ROC Curve. A single number summarising model accuracy: 1.0 = perfect, 0.5 = no better than a coin flip. AUC 0.9+ is considered excellent.

**R / S** — Resistant / Susceptible. The binary clinical label assigned to each genome by laboratory phenotypic testing.

**ESKAPE pathogens** — *E. faecium*, *S. aureus*, *K. pneumoniae*, *A. baumannii*, *P. aeruginosa*, *Enterobacter* spp. — the six WHO critical-priority organisms driving hospital infections worldwide.
""")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #EEF2FF 0%, #F0F9FF 50%, #F5F3FF 100%);
        padding: 3rem 2rem; border-radius: 16px; margin-bottom: 2rem;
        border: 1px solid #E0E7FF;
    }
    .hero h1 { color: #4338CA; font-size: 2.8rem; font-weight: 800; margin-bottom: 0.5rem; }
    .hero p  { color: #475569; font-size: 1.15rem; line-height: 1.6; }
    .metric-card {
        background: #FFFFFF; border: 1px solid #E0E7FF;
        border-radius: 12px; padding: 1.5rem; text-align: center;
        box-shadow: 0 1px 3px rgba(99,102,241,0.08);
    }
    .metric-card .value { font-size: 2.5rem; font-weight: 700; color: #6366F1; }
    .metric-card .label { color: #64748B; font-size: 0.9rem; margin-top: 0.3rem; }
    .org-card {
        background: #FFFFFF; border-radius: 12px; padding: 1.2rem;
        border: 1px solid #E2E8F0; transition: box-shadow 0.2s;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06); height: 200px;
    }
    .step-card {
        background: #FAFAFA; border-left: 4px solid #6366F1;
        border-radius: 6px; padding: 1rem 1.5rem; margin-bottom: 0.8rem;
        border-top: 1px solid #F1F5F9; border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
    }
    .step-card h4 { color: #1E293B; margin: 0 0 0.3rem; }
    .step-card p  { color: #475569; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧬 AMR Predictor</h1>
  <p>Predicting antibiotic resistance across four critical pathogens —
     <em>K. pneumoniae</em>, <em>E. coli</em>, <em>S. aureus</em> &amp; <em>A. baumannii</em> —
     from whole-genome sequences using machine learning.</p>
  <p style="color:#94A3B8; font-size:0.9rem; margin-top:0.8rem;">
    A bioinformatics + ML project — from raw DNA to clinical resistance predictions.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Key metrics ───────────────────────────────────────────────────────────────
summary    = json.loads((ART_DIR / "summary.json").read_text())
stats      = json.loads((ART_DIR / "dataset_stats.json").read_text())
multi_path = ART_DIR / "multi_org_summary.json"
multi_sum  = json.loads(multi_path.read_text()) if multi_path.exists() else []

all_aucs  = [r["test_auc"] for r in summary] + [r["test_auc"] for r in multi_sum]
mean_auc  = sum(all_aucs) / len(all_aucs)
best_auc  = max(all_aucs)
n_models  = len(summary) + len(multi_sum)
total_gen = sum(r["n_resistant"] + r["n_susceptible"] for r in stats)
total_gen += sum(r["n_total"] for r in multi_sum)

col1, col2, col3, col4 = st.columns(4)
for col, value, label in [
    (col1, str(n_models),           "ML models · 4 organisms"),
    (col2, f"{total_gen:,}",        "Labeled genomes used"),
    (col3, f"{mean_auc:.2f}",       "Mean ROC-AUC"),
    (col4, f"{best_auc:.2f}",       "Best ROC-AUC"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{value}</div>
            <div class="label">{label}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Organism cards with navigation ────────────────────────────────────────────
st.subheader("🦠 Select a pathogen to predict resistance")



ORG_CARDS = [
    {
        "emoji": "🔴", "name": "Klebsiella pneumoniae",
        "tag": "Critical priority · WHO",
        "desc": "10 antibiotics · k-mer + gene features · AUC 0.76–0.89",
        "models": len(summary),
        "color": "#EF4444", "bg": "#FFF5F5", "border": "#FCA5A5",
        "page": "pages/5_Live_Predictor.py",
        "btn": "→ Live Predictor",
    },
    {
        "emoji": "🟠", "name": "Escherichia coli",
        "tag": "Critical priority · WHO",
        "desc": "10 antibiotics · gene features · AUC 0.961–0.990",
        "models": sum(1 for r in multi_sum if r["organism"] == "escherichia_coli"),
        "color": "#F97316", "bg": "#FFF7ED", "border": "#FED7AA",
        "page": "pages/17_Multi_Organism.py",
        "btn": "→ Multi-Organism",
    },
    {
        "emoji": "🟡", "name": "Staphylococcus aureus",
        "tag": "High priority · WHO",
        "desc": "8 antibiotics · gene features · AUC 0.952–0.994",
        "models": sum(1 for r in multi_sum if r["organism"] == "staphylococcus_aureus"),
        "color": "#D97706", "bg": "#FFFBEB", "border": "#FCD34D",
        "page": "pages/17_Multi_Organism.py",
        "btn": "→ Multi-Organism",
    },
    {
        "emoji": "🔵", "name": "Acinetobacter baumannii",
        "tag": "Critical priority · WHO",
        "desc": "8 antibiotics · gene features · AUC 0.930–0.989",
        "models": sum(1 for r in multi_sum if r["organism"] == "acinetobacter_baumannii"),
        "color": "#3B82F6", "bg": "#EFF6FF", "border": "#BFDBFE",
        "page": "pages/17_Multi_Organism.py",
        "btn": "→ Multi-Organism",
    },
]

cols = st.columns(4)
for i, card in enumerate(ORG_CARDS):
    with cols[i]:
        st.markdown(f"""
<div style='background:{card["bg"]}; border:1.5px solid {card["border"]};
     border-radius:12px; padding:1rem 1.1rem;
     box-shadow:0 2px 6px rgba(0,0,0,0.06); min-height:190px;'>
  <div style='font-size:2rem; margin-bottom:4px;'>{card["emoji"]}</div>
  <div style='font-weight:700; color:#1E293B; font-size:0.92rem;
       line-height:1.3; margin-bottom:4px;'><em>{card["name"]}</em></div>
  <div style='font-size:0.72rem; color:{card["color"]}; font-weight:600;
       margin-bottom:6px;'>{card["tag"]}</div>
  <div style='font-size:0.78rem; color:#475569; margin-bottom:10px;
       line-height:1.4;'>{card["desc"]}</div>
</div>""", unsafe_allow_html=True)
        st.page_link(card["page"], label=card["btn"], use_container_width=True)

st.divider()

# ── KP Performance mini-chart ─────────────────────────────────────────────────
st.subheader("📊 K. pneumoniae model performance (10 antibiotics)")

antibiotics = [r["antibiotic"] for r in sorted(summary, key=lambda x: x["test_auc"])]
aucs        = [r["test_auc"]   for r in sorted(summary, key=lambda x: x["test_auc"])]
bar_colors  = ["#6366F1" if a >= 0.85 else "#818CF8" if a >= 0.80 else "#A5B4FC"
               for a in aucs]

fig = go.Figure(go.Bar(
    x=aucs, y=antibiotics, orientation="h",
    marker_color=bar_colors,
    text=[f"{a:.3f}" for a in aucs], textposition="outside",
    textfont=dict(color="#1E293B"),
))
fig.add_vline(x=0.8, line_dash="dash", line_color="#10B981",
              annotation_text="Good threshold (0.80)",
              annotation_font_color="#10B981")
fig.update_layout(
    xaxis=dict(range=[0.4, 1.0], title="ROC-AUC (test set)", gridcolor="#E2E8F0"),
    yaxis=dict(gridcolor="#E2E8F0"),
    height=320, margin=dict(l=10, r=90, t=10, b=40),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font_color="#1E293B",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Project pipeline steps ────────────────────────────────────────────────────
st.subheader("⚙️ How the project works")

steps = [
    ("1 — The Biology",        "Bacteria develop resistance through DNA mutations and gene acquisition. We frame it as a binary classification problem per antibiotic."),
    ("2 — The Data",           "26,000+ K. pneumoniae + 130,000+ multi-organism genomes from BV-BRC (74 countries, 2000–2024), each with Resistant / Susceptible labels."),
    ("3 — Feature Extraction", "Two feature types: k-mer frequency counts (6-letter DNA substrings) and resistance gene presence/absence flags from CARD/NDARO databases."),
    ("4 — Model Training",     "Calibrated XGBoost ensembles per organism × antibiotic. 36 models total, AUC 0.76–1.00. Gene features alone are near-deterministic for many antibiotics."),
    ("5 — Prediction",         "Five predictors: Live (BV-BRC ID), Offline (gene toggles), FASTA Upload (raw assembly), Multi-Organism (4 pathogens), Strain Comparison (side-by-side). All output probabilities + PDF report."),
    ("6 — Explainability",     "SHAP values reveal which genes and k-mer patterns drive each prediction. Co-resistance network shows which drug pairs fail together."),
    ("6 — Explainability",     "SHAP values reveal which genes and k-mer patterns drive each prediction. Co-resistance network shows which drug pairs fail together."),
    ("7–13 — Epidemiology",    "Gene emergence curves, MDR trends over 24 years, global resistance maps, 5-year forecasts, and MLST lineage-specific resistance profiles."),
    ("14–16 — Clinical Tools", "Antibiogram heatmap (4 organisms × all antibiotics), treatment recommendation engine with tier-based drug guidance, and MLST outbreak detection via PCA clustering."),
]

for title, desc in steps:
    st.markdown(f"""<div class="step-card">
        <h4>{title}</h4><p>{desc}</p></div>""", unsafe_allow_html=True)

st.markdown("""
<p style="color:#94A3B8; font-size:0.85rem; margin-top:1rem;">
  Use the sidebar to explore each section, or click an organism card above to go straight to the predictor.
</p>
""", unsafe_allow_html=True)
