"""
Page 14 — Offline Predictor: predict resistance by toggling gene presence.

No genome ID needed. No internet required. The user manually marks which
resistance genes are present in their strain and gets an instant prediction.

This works because our models use ~256 k-mers + ~1600 gene flags as features.
When a user provides the gene flags directly, we set k-mer features to zero
(their absence is informative) and use the gene-only signal.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT      = Path(__file__).parent.parent.parent
ART_DIR   = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
st.set_page_config(page_title="Offline Predictor", page_icon="⚡", layout="wide")
inject_mobile_css()
st.title("⚡ Offline Predictor")
st.markdown("*No genome ID required. Toggle which resistance genes are present → instant resistance profile.*")
st.info("💡 This predictor works completely offline — just check the genes your lab or sequencing report identified, and the model predicts resistance to all 10 antibiotics instantly.")
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
    "ciprofloxacin": "Fluoroquinolone", "meropenem": "Carbapenem",
    "gentamicin": "Aminoglycoside", "tetracycline": "Tetracycline",
    "trimethoprim/sulfamethoxazole": "Folate inhibitor", "cefepime": "Cephalosporin",
    "amikacin": "Aminoglycoside", "imipenem": "Carbapenem",
    "piperacillin/tazobactam": "Beta-lactam/inhibitor", "levofloxacin": "Fluoroquinolone",
}
RESIST_COLOR = "#e94560"
SUSCEPT_COLOR = "#50fa7b"
UNCERTAIN_COLOR = "#ffb86c"

# Gene categories for organised display
GENE_CATEGORIES = {
    "Beta-lactamases (destroy penicillins & cephalosporins)": [
        "CTX-M", "TEM", "SHV", "OXA", "KPC", "NDM", "VIM", "IMP",
    ],
    "Quinolone resistance (ciprofloxacin / levofloxacin)": [
        "gyrA", "parC", "qnrA", "qnrB", "qnrS", "aac(6')-Ib-cr",
    ],
    "Aminoglycoside resistance (gentamicin / amikacin)": [
        "aac(3)", "aac(6')", "ant(2'')", "aph(3')", "armA", "rmtB",
    ],
    "Efflux pumps (broad spectrum)": [
        "AcrAB-TolC", "MdtABC-TolC", "OqxAB", "KpnEF",
    ],
    "Sulfonamide / TMP resistance": [
        "dhps", "dhfr", "sul1", "sul2", "dfrA",
    ],
    "Carbapenem resistance": [
        "blaKPC", "blaNDM", "blaOXA-48", "blaVIM", "blaIMP",
    ],
}


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for ab in ANTIBIOTICS:
        safe = ab.replace("/", "_").replace(" ", "_")
        p = MODEL_DIR / f"{safe}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                models[ab] = pickle.load(f)
    return models


@st.cache_data(show_spinner=False)
def get_gene_list():
    """Get all genes from the gene matrix columns."""
    gene_path = PROC_DIR / "gene_matrix.csv"
    if not gene_path.exists():
        return []
    # Just read the header
    df = pd.read_csv(gene_path, index_col=0, nrows=0)
    genes = [c for c in df.columns if c != "__label__"]
    return sorted(genes)


def predict_from_genes(gene_presence: dict, models: dict, threshold: float = 0.65) -> list:
    """
    Given a dict of {gene_name: 0/1}, build a feature vector and predict.
    K-mer features are set to 0 (conservative — no k-mer signal available).
    """
    results = []
    for ab, bundle in models.items():
        model    = bundle["model"]
        features = bundle["features"]

        # Build feature vector: 0 for k-mers, gene presence for gene__ features
        x = np.zeros(len(features))
        for i, feat in enumerate(features):
            if feat.startswith("gene__"):
                gene_name = feat[len("gene__"):]
                x[i] = gene_presence.get(gene_name, 0)

        X_input = pd.DataFrame([x], columns=features)

        prob_r = float(model.predict_proba(X_input)[0, 1])
        if prob_r >= threshold:
            verdict = "Resistant"
        elif prob_r <= (1 - threshold):
            verdict = "Susceptible"
        else:
            verdict = "Uncertain"

        results.append({
            "antibiotic": ab,
            "short":      SHORT[ab],
            "drug_class": DRUG_CLASS[ab],
            "prob_r":     prob_r,
            "verdict":    verdict,
        })
    return sorted(results, key=lambda x: -x["prob_r"])


# Load
with st.spinner("Loading models..."):
    models = load_models()
    gene_list = get_gene_list()

if not models:
    st.error("No models found. Run `python src/train_multi.py` first.")
    st.stop()

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ How this predictor works"):
    st.markdown(f"""
Our models were trained on **{len(gene_list):,} resistance genes** (binary presence/absence)
plus k-mer frequency features from whole genome sequences.

In this offline mode:
- You provide **gene presence** directly (from PCR, sequencing report, or CARD database)
- K-mer features are set to zero (conservative — only gene signal is used)
- The model outputs a resistance probability for each of {len(models)} antibiotics

**When to use this:**
- You have gene data from CARD database / ResFinder / AMRFinder
- You don't have a BV-BRC genome ID
- You're working offline or in a demo setting

**Confidence threshold:** {65}% — predictions below this are flagged as "Uncertain".
""")

st.divider()

# ── Gene selection ─────────────────────────────────────────────────────────────
st.header("1. Select resistance genes present in your strain")

tab_quick, tab_full = st.tabs(["⚡ Quick — common genes", "🔍 Full gene list"])

selected_genes = set()

with tab_quick:
    st.markdown("Check all genes identified in your strain. Common categories:")
    for category, gene_patterns in GENE_CATEGORIES.items():
        st.markdown(f"**{category}**")
        cols = st.columns(4)
        for i, gene_pat in enumerate(gene_patterns):
            # Find all matching genes from the full list
            matches = [g for g in gene_list if gene_pat.lower() in g.lower()]
            label = f"{gene_pat}" + (f" ({len(matches)} variants)" if len(matches) > 1 else "")
            if cols[i % 4].checkbox(label, key=f"quick_{gene_pat}"):
                selected_genes.update(matches)

with tab_full:
    st.markdown("Search and toggle any gene from our complete database:")
    search = st.text_input("Search genes:", placeholder="e.g. CTX-M, gyrA, KPC...")
    if search:
        filtered = [g for g in gene_list if search.lower() in g.lower()]
        st.caption(f"{len(filtered)} genes matching '{search}'")
        cols2 = st.columns(2)
        for i, gene in enumerate(filtered[:40]):
            if cols2[i % 2].checkbox(gene[:60], key=f"full_{gene}"):
                selected_genes.add(gene)
        if len(filtered) > 40:
            st.caption(f"Showing first 40 of {len(filtered)} matches. Refine your search.")
    else:
        st.info("Type a gene name above to search the full database of {:,} genes.".format(len(gene_list)))

# ── Preset strain profiles ─────────────────────────────────────────────────────
st.divider()
st.header("2. Or load a preset strain profile")

PRESETS = {
    "ST258 KPC carbapenem-resistant": {
        "desc": "Classic ST258 — dominant worldwide carbapenem-resistant K. pneumoniae",
        "genes": ["CTX-M", "KPC", "SHV", "aac(6')-Ib-cr", "AcrAB-TolC", "OqxAB"],
    },
    "Community ESBL (susceptible to carbapenems)": {
        "desc": "CTX-M ESBL producing strain, carbapenem-susceptible",
        "genes": ["CTX-M", "TEM", "sul1", "dhfr", "gyrA"],
    },
    "Aminoglycoside-resistant (gentamicin failed)": {
        "desc": "High-level aminoglycoside resistance via methylase",
        "genes": ["armA", "aac(6')", "aac(3)", "AcrAB-TolC"],
    },
    "Pan-susceptible baseline": {
        "desc": "No known resistance genes — susceptible to most antibiotics",
        "genes": [],
    },
}

preset_choice = st.selectbox(
    "Load a preset:",
    options=["— none —"] + list(PRESETS.keys()),
)
if preset_choice != "— none —":
    preset = PRESETS[preset_choice]
    st.caption(f"_{preset['desc']}_")
    for gene_pat in preset["genes"]:
        matches = [g for g in gene_list if gene_pat.lower() in g.lower()]
        selected_genes.update(matches)
    if preset["genes"]:
        st.success(f"Loaded: {', '.join(preset['genes'])}")

# ── Predict ────────────────────────────────────────────────────────────────────
st.divider()
st.header("3. Resistance prediction")

threshold = st.slider("Confidence threshold:", 0.50, 0.90, 0.65, 0.05,
                      help="Predictions below this confidence are marked Uncertain")

gene_presence = {g: 1 for g in selected_genes}
n_selected = len(selected_genes)

if n_selected == 0:
    st.info("👆 Select at least one gene above (or load a preset) to see predictions.")
else:
    st.caption(f"**{n_selected} genes selected.** Running prediction across {len(models)} antibiotics...")
    preds = predict_from_genes(gene_presence, models, threshold=threshold)

    # Summary donut
    n_r = sum(1 for p in preds if p["verdict"] == "Resistant")
    n_s = sum(1 for p in preds if p["verdict"] == "Susceptible")
    n_u = sum(1 for p in preds if p["verdict"] == "Uncertain")

    col_donut, col_cards = st.columns([1, 2])
    with col_donut:
        fig_donut = go.Figure(go.Pie(
            labels=["Resistant", "Uncertain", "Susceptible"],
            values=[n_r, n_u, n_s],
            hole=0.6,
            marker_colors=[RESIST_COLOR, UNCERTAIN_COLOR, SUSCEPT_COLOR],
            textinfo="label+value",
        ))
        fig_donut.update_layout(
            height=260, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#1e1e2e", font_color="#cdd6f4",
            showlegend=False,
            annotations=[dict(text=f"{n_r}/{len(preds)}<br>Resistant",
                              x=0.5, y=0.5, font_size=14,
                              font_color="#e94560", showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_cards:
        if n_r == len(preds):
            st.error(f"⚠️ **Pan-resistant pattern** — resistant to all {len(preds)} tested antibiotics.")
        elif n_r >= 3:
            st.warning(f"🔴 **MDR pattern** — resistant to {n_r} antibiotics (MDR threshold = 3+).")
        elif n_r == 0:
            st.success("✅ **Susceptible profile** — no resistance predicted.")
        else:
            st.info(f"🟡 **Partial resistance** — {n_r} antibiotic(s) affected.")

    # Per-antibiotic result cards
    for pred in preds:
        ab       = pred["antibiotic"]
        verdict  = pred["verdict"]
        prob_r   = pred["prob_r"]
        color = (RESIST_COLOR if verdict == "Resistant" else
                 SUSCEPT_COLOR if verdict == "Susceptible" else
                 UNCERTAIN_COLOR)
        emoji = "🔴" if verdict == "Resistant" else "🟢" if verdict == "Susceptible" else "🟡"

        bar_r = int(prob_r * 100)
        bar_s = 100 - bar_r
        st.markdown(f"""
<div style='background:#181825; border-left:4px solid {color};
     padding:0.5rem 1rem; border-radius:6px; margin-bottom:6px;
     display:flex; justify-content:space-between; align-items:center;'>
  <div>
    <b style='color:#cdd6f4; font-size:0.95rem;'>{emoji} {pred["short"]}</b>
    <span style='color:#6272a4; font-size:0.8rem; margin-left:8px;'>{pred["drug_class"]}</span>
  </div>
  <div style='text-align:right; min-width:200px;'>
    <span style='color:{color}; font-weight:bold;'>{verdict}</span>
    <div style='background:#2d2d44; border-radius:4px; height:6px; margin-top:4px; width:180px;'>
      <div style='background:{color}; width:{bar_r}%; height:6px; border-radius:4px;'></div>
    </div>
    <small style='color:#6272a4;'>P(R) = {prob_r:.1%}</small>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # Selected genes display
    with st.expander(f"📋 Genes used in prediction ({n_selected} selected)"):
        gene_cols = st.columns(3)
        for i, g in enumerate(sorted(selected_genes)):
            gene_cols[i % 3].markdown(f"✅ `{g[:50]}`")

    # Download
    result_df = pd.DataFrame([{
        "antibiotic": p["antibiotic"],
        "drug_class": p["drug_class"],
        "verdict": p["verdict"],
        "prob_resistant": round(p["prob_r"], 4),
        "genes_used": n_selected,
    } for p in preds])

    csv = result_df.to_csv(index=False)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv,
            file_name="amr_gene_prediction.csv",
            mime="text/csv",
        )
    with col_dl2:
        try:
            from pdf_report import generate_pdf
            pdf_bytes = generate_pdf(preds, genome_id="Offline (gene toggle)", source="Manual gene selection")
            st.download_button(
                "📄 Download PDF report",
                data=pdf_bytes,
                file_name="amr_gene_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.caption(f"PDF unavailable: {e}")

st.divider()

# ── Gene reference ─────────────────────────────────────────────────────────────
st.header("4. Gene reference guide")

st.markdown("""
| Gene / family | Mechanism | Drugs affected |
|---|---|---|
| **CTX-M, SHV, TEM** (ESBL) | Hydrolyse beta-lactam ring | Cefepime, piperacillin/tazobactam |
| **KPC, NDM, VIM, IMP, OXA-48** | Carbapenemase — destroy carbapenems | Meropenem, imipenem (last resort) |
| **gyrA, parC** | Gyrase mutation — blocks quinolone binding | Ciprofloxacin, levofloxacin |
| **qnrA/B/S** | Protects gyrase from quinolone inhibition | Ciprofloxacin, levofloxacin |
| **aac(6'), aac(3'), armA** | Aminoglycoside-modifying enzymes | Gentamicin, amikacin |
| **sul1/2, dhfr** | Bypass sulfonamide / folate pathway | TMP/SMX |
| **AcrAB-TolC, OqxAB** | Efflux pumps — pump drug out of cell | Multiple (broad spectrum) |

**Tip:** Run your genome through [CARD RGI](https://card.mcmaster.ca/analyze/rgi) or
[AMRFinderPlus](https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/AMRFinder/)
to get a gene list, then paste the results into the selector above.
""")
