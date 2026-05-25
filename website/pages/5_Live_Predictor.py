"""
Page 5 — Live Predictor: enter a BV-BRC genome ID → resistance profile
"""
import sys
import pickle
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

ROOT    = Path(__file__).parent.parent.parent
ART_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
st.set_page_config(page_title="Live Predictor", page_icon="🔮", layout="wide")
inject_mobile_css()
st.title("🔮 Live Predictor")
st.markdown("*Enter a BV-BRC genome ID to get a full resistance profile across all 10 antibiotics.*")
st.divider()

ANTIBIOTICS = [
    "ciprofloxacin", "meropenem", "gentamicin", "tetracycline",
    "trimethoprim/sulfamethoxazole", "cefepime",
    "amikacin", "imipenem", "piperacillin/tazobactam", "levofloxacin",
]
SHORT = {
    "ciprofloxacin": "Cipro", "meropenem": "Mero",
    "gentamicin": "Gent", "tetracycline": "Tet",
    "trimethoprim/sulfamethoxazole": "TMP/SMX", "cefepime": "Cef",
    "amikacin": "Amik", "imipenem": "Imi",
    "piperacillin/tazobactam": "Pip/Taz", "levofloxacin": "Levo",
}
DRUG_CLASS = {
    "ciprofloxacin":                 "Fluoroquinolone",
    "meropenem":                     "Carbapenem",
    "gentamicin":                    "Aminoglycoside",
    "tetracycline":                  "Tetracycline",
    "trimethoprim/sulfamethoxazole": "Folate inhibitor",
    "cefepime":                      "Cephalosporin",
    "amikacin":                      "Aminoglycoside",
    "imipenem":                      "Carbapenem",
    "piperacillin/tazobactam":       "Beta-lactam/inhibitor",
    "levofloxacin":                  "Fluoroquinolone",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for ab in ANTIBIOTICS:
        safe = ab.replace("/","_").replace(" ","_")
        p = MODEL_DIR / f"{safe}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                models[ab] = pickle.load(f)
    return models

@st.cache_resource(show_spinner=False)
def load_kmer_matrix():
    X = pd.read_csv(PROC_DIR / "X.csv", index_col=0)
    X.index = X.index.astype(str)
    return X

@st.cache_resource(show_spinner=False)
def load_gene_matrix():
    g = pd.read_csv(PROC_DIR / "gene_matrix.csv", index_col=0)
    g.index = g.index.astype(str)
    g = g.drop(columns=["__label__"], errors="ignore")
    return g

def fetch_gene_vector_api(genome_id: str) -> dict:
    """Fetch resistance gene presence from BV-BRC API."""
    url = (f"https://www.bv-brc.org/api/genome_amr/"
           f"?eq(genome_id,{genome_id})&select(resistance_mechanism,gene)&limit(500)")
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        data = r.json()
        return {d.get("gene","") for d in data if d.get("gene")}
    except Exception:
        return set()

def predict_genome(genome_id: str, models: dict,
                   X_kmer: pd.DataFrame, X_gene: pd.DataFrame,
                   threshold: float = 70.0):
    results = []
    for ab, bundle in models.items():
        features = bundle["features"]
        kmer_cols = [c for c in features if not c.startswith("gene__")]
        gene_cols  = [c for c in features if c.startswith("gene__")]
        gene_bare  = [c.replace("gene__","") for c in gene_cols]

        # k-mer row
        if genome_id in X_kmer.index:
            kmer_row = X_kmer.loc[genome_id, kmer_cols] if kmer_cols else pd.Series(dtype=float)
        else:
            kmer_row = pd.Series(0.0, index=kmer_cols)

        # gene row
        if genome_id in X_gene.index:
            gene_row = X_gene.loc[genome_id, [c for c in gene_bare if c in X_gene.columns]]
            gene_row = gene_row.reindex(gene_bare, fill_value=0).rename(lambda x: "gene__"+x)
        else:
            gene_row = pd.Series(0, index=gene_cols)

        x = pd.concat([kmer_row, gene_row]).to_frame().T

        prob_r = bundle["model"].predict_proba(x)[0][1]
        confidence = max(prob_r, 1 - prob_r) * 100

        if confidence < threshold:
            verdict = "Uncertain"
        elif prob_r > 0.5:
            verdict = "Resistant"
        else:
            verdict = "Susceptible"

        results.append({
            "antibiotic":   ab,
            "drug_class":   DRUG_CLASS[ab],
            "verdict":      verdict,
            "prob_r":       round(prob_r * 100, 1),
            "prob_s":       round((1 - prob_r) * 100, 1),
            "confidence":   round(confidence, 1),
            "model_auc":    bundle.get("test_auc", 0),
        })
    return pd.DataFrame(results)

# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_info = st.columns([2, 3])

with col_input:
    st.subheader("Input")
    genome_id = st.text_input(
        "BV-BRC Genome ID",
        placeholder="e.g. 573.12783",
        help="Find genome IDs at bv-brc.org — use the Genome search and filter by Klebsiella pneumoniae.",
    )
    threshold = st.slider(
        "Confidence threshold (%)",
        min_value=50, max_value=95, value=70, step=5,
        help="Predictions below this confidence are shown as Uncertain.",
    )

    example_ids = {
        "Multi-drug resistant strain": "573.12783",
        "Susceptible reference strain": "573.65923",
        "Carbapenem-resistant (KPC)":  "573.52406",
    }
    st.markdown("**Try an example:**")
    for label, gid in example_ids.items():
        if st.button(label, use_container_width=True):
            genome_id = gid

    run = st.button("🔮 Predict Resistance Profile", type="primary",
                    use_container_width=True, disabled=not genome_id)

with col_info:
    st.subheader("How the predictor works")
    st.markdown("""
    1. **k-mers** — your genome's 6-mer counts are looked up from the pre-computed matrix
    2. **Resistance genes** — gene presence/absence is looked up from the gene annotation matrix
    3. **6 models run in parallel** — one per antibiotic, each producing a probability
    4. **Confidence threshold** — predictions below your chosen threshold are flagged as Uncertain
    """)
    st.info("For genomes in our training set, features are pre-computed (fast). "
            "For new genomes, k-mers would need to be computed from the FASTA file.")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if run and genome_id:
    genome_id = genome_id.strip()
    with st.spinner(f"Running predictions for genome {genome_id} …"):
        models = load_models()
        X_kmer = load_kmer_matrix()
        X_gene = load_gene_matrix()

        if not models:
            st.error("No trained models found. Please run `generate_artifacts.py` first.")
            st.stop()

        df_results = predict_genome(genome_id, models, X_kmer, X_gene, threshold)

    # ── Profile table ─────────────────────────────────────────────────────────
    st.subheader(f"Resistance profile — Genome {genome_id}")

    verdict_color = {"Resistant": "🔴", "Susceptible": "🟢", "Uncertain": "🟡"}

    col_table, col_gauge = st.columns([3, 2])

    with col_table:
        for _, row in df_results.sort_values("antibiotic").iterrows():
            icon = verdict_color[row["verdict"]]
            bar_color = ("#e94560" if row["verdict"] == "Resistant"
                         else "#50fa7b" if row["verdict"] == "Susceptible"
                         else "#ffb86c")
            st.markdown(f"""
<div style="background:#FFFFFF; border-radius:8px; padding:0.8rem 1rem;
            margin-bottom:0.5rem; border-left: 4px solid {bar_color};">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <strong style="color:#1E293B;">{row['antibiotic']}</strong>
      <span style="color:#64748B; font-size:0.8rem; margin-left:0.5rem;">
        ({row['drug_class']})
      </span>
    </div>
    <div style="text-align:right;">
      <strong style="color:{bar_color};">{icon} {row['verdict']}</strong>
      <span style="color:#4A5568; font-size:0.85rem; margin-left:0.8rem;">
        {row['confidence']:.0f}% confidence
      </span>
    </div>
  </div>
  <div style="margin-top:0.4rem; background:#EEF2FF; border-radius:4px; height:6px;">
    <div style="width:{row['prob_r']}%; background:{bar_color};
                border-radius:4px; height:6px;"></div>
  </div>
  <div style="display:flex; justify-content:space-between;
              color:#64748B; font-size:0.75rem; margin-top:0.2rem;">
    <span>P(Susceptible) = {row['prob_s']:.1f}%</span>
    <span>P(Resistant) = {row['prob_r']:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

    with col_gauge:
        n_r = (df_results["verdict"] == "Resistant").sum()
        n_s = (df_results["verdict"] == "Susceptible").sum()
        n_u = (df_results["verdict"] == "Uncertain").sum()

        fig_pie = go.Figure(go.Pie(
            labels=["Resistant", "Susceptible", "Uncertain"],
            values=[n_r, n_s, n_u],
            hole=0.6,
            marker_colors=["#e94560", "#50fa7b", "#ffb86c"],
            textinfo="label+value",
        ))
        fig_pie.update_layout(
            title="Summary",
            height=280, margin=dict(t=40, b=10),
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="#1E293B", showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Model AUC reference
        st.markdown("**Model quality (test AUC):**")
        for _, row in df_results.iterrows():
            auc = row["model_auc"]
            st.markdown(
                f"<small style='color:#4A5568;'>{row['antibiotic']}: "
                f"<strong style='color:#50fa7b;'>{auc:.3f}</strong></small>",
                unsafe_allow_html=True
            )

    if n_u > 0:
        st.warning(
            f"⚠️ {n_u} prediction(s) marked Uncertain (confidence < {threshold}%). "
            "These strains may have unusual genomic profiles not well-represented in training data. "
            "Lab confirmation is advised."
        )

    # ── Download results ──────────────────────────────────────────────────────
    csv = df_results[["antibiotic","drug_class","verdict","prob_r","prob_s","confidence"]].to_csv(index=False)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv,
            file_name=f"amr_profile_{genome_id}.csv",
            mime="text/csv",
        )
    with col_dl2:
        try:
            from pdf_report import generate_pdf
            _SHORT = {
                "ciprofloxacin": "Cipro", "meropenem": "Mero",
                "gentamicin": "Gent", "tetracycline": "Tet",
                "trimethoprim/sulfamethoxazole": "TMP/SMX", "cefepime": "Cef",
                "amikacin": "Amik", "imipenem": "Imi",
                "piperacillin/tazobactam": "Pip/Taz", "levofloxacin": "Levo",
            }
            pred_list = df_results[["antibiotic","drug_class","verdict","prob_r"]].to_dict("records")
            for p in pred_list:
                p["short"] = _SHORT.get(p["antibiotic"], p["antibiotic"][:8])
            pdf_bytes = generate_pdf(pred_list, genome_id=genome_id, source="BV-BRC")
            st.download_button(
                "📄 Download PDF report",
                data=pdf_bytes,
                file_name=f"amr_report_{genome_id}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.caption(f"PDF unavailable: {e}")
