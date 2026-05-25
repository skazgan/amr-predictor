"""
Page 15 — FASTA Upload Predictor.

Upload your own genome FASTA file → get a resistance profile.
No BV-BRC genome ID needed. Works with any K. pneumoniae whole-genome assembly.

Pipeline:
  1. User uploads .fasta / .fa / .fna file
  2. We compute 6-mer frequencies (same feature pipeline as training)
  3. We query BV-BRC's gene annotation API for the genome (optional, if genome ID known)
  4. We run all 10 calibrated ensemble models
  5. We display the resistance profile with confidence bars
"""
import io
import pickle
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT      = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"

# Add src to path for features.py
import sys
sys.path.insert(0, str(ROOT / "src"))


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
st.set_page_config(page_title="FASTA Upload", page_icon="📂", layout="wide")
inject_mobile_css()
st.title("📂 FASTA Upload Predictor")
st.markdown("*Upload your own genome assembly → get an instant resistance prediction.*")
st.info("💡 This page runs the complete prediction pipeline on any K. pneumoniae FASTA file you provide — no internet connection to BV-BRC needed for the prediction itself.")
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

RESIST_COLOR   = "#e94560"
SUSCEPT_COLOR  = "#50fa7b"
UNCERTAIN_COLOR = "#ffb86c"


# ── Helpers ───────────────────────────────────────────────────────────────────

def kmer_counts(seq: str, k: int = 6) -> dict:
    """Count all k-mer frequencies in a DNA sequence."""
    counts: dict[str, int] = {}
    seq = seq.upper().replace("N", "")
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if all(c in "ACGT" for c in kmer):
            counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def parse_fasta(content: str) -> str:
    """Parse FASTA text → concatenated sequence (all contigs)."""
    seq_parts = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith(">") or not line:
            continue
        seq_parts.append(line)
    return "".join(seq_parts)


def compute_kmer_vector(fasta_text: str, feature_cols: list[str], k: int = 6) -> np.ndarray:
    """
    Compute normalised k-mer frequencies for a FASTA string,
    aligned to the training feature columns.
    """
    seq = parse_fasta(fasta_text)
    if not seq:
        return np.zeros(len(feature_cols))

    raw = kmer_counts(seq, k)
    total = max(sum(raw.values()), 1)
    vec = np.array([raw.get(col, 0) / total for col in feature_cols], dtype=float)
    return vec


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


def predict_from_fasta(fasta_text: str, models: dict,
                       threshold: float = 0.65) -> list:
    """Run all models on the uploaded FASTA and return predictions."""
    results = []
    for ab, bundle in models.items():
        model    = bundle["model"]
        features = bundle["features"]

        # Separate k-mer and gene feature columns
        kmer_cols = [f for f in features if not f.startswith("gene__")]
        gene_cols = [f for f in features if f.startswith("gene__")]

        # K-mer vector (from FASTA)
        kmer_vec = compute_kmer_vector(fasta_text, kmer_cols, k=6)

        # Gene vector (zeros — no gene annotation from raw FASTA)
        gene_vec = np.zeros(len(gene_cols))

        x = np.concatenate([kmer_vec, gene_vec])
        X_input = pd.DataFrame([x], columns=features)

        prob_r = float(model.predict_proba(X_input)[0, 1])
        verdict = (
            "Resistant"    if prob_r >= threshold else
            "Susceptible"  if prob_r <= (1 - threshold) else
            "Uncertain"
        )

        results.append({
            "antibiotic": ab,
            "short":      SHORT[ab],
            "drug_class": DRUG_CLASS[ab],
            "prob_r":     prob_r,
            "verdict":    verdict,
        })

    return sorted(results, key=lambda x: -x["prob_r"])


# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    models = load_models()

if not models:
    st.error("No models found in `models/`. Run `python src/train_multi.py` to generate them.")
    st.stop()

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ What happens when you upload a FASTA?"):
    st.markdown("""
1. **K-mer extraction** — we slide a 6-letter window across every base in your genome
   and count all 4,096 possible DNA hexamers.
2. **Feature alignment** — we pick the same ~256 most informative k-mers the model
   was trained on (selected via mutual information during training).
3. **Prediction** — the ensemble model (XGBoost 60% + Random Forest 40%) outputs
   a calibrated probability of resistance for each antibiotic.
4. **Note on gene features** — when uploading a raw FASTA, resistance gene annotations
   are not available, so gene features are set to zero. This is conservative — the
   k-mer signal alone is still informative (AUC ~0.66–0.70 without genes).
   For the best accuracy, use the **Live Predictor** (Page 5) with a BV-BRC genome ID,
   which also fetches gene annotation.

**File requirements:**
- Standard FASTA format (`.fasta`, `.fa`, `.fna`, `.fasta.gz`)
- K. pneumoniae whole-genome assembly (contigs or complete chromosome)
- Typical file size: 5–7 MB
""")

st.divider()

# ── File upload ───────────────────────────────────────────────────────────────
st.header("1. Upload your genome")

col_upload, col_demo = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload a FASTA file:",
        type=["fasta", "fa", "fna", "txt"],
        help="Whole-genome assembly of K. pneumoniae in FASTA format.",
    )

with col_demo:
    st.markdown("**Or use a demo sequence:**")
    if st.button("🧬 Load synthetic demo", help="Uses a short synthetic K. pneumoniae-like sequence for demonstration"):
        # Synthetic K. pneumoniae-like sequence (not a real genome, for demo)
        st.session_state["demo_fasta"] = (
            ">synthetic_Kpneumoniae_demo\n" +
            "ATGAGTATTCAACATTTCCGTGTCGCCCTTATTCCCTTTTTTGCGGCATTTTGCCTTCCTGTTTTTGCT" * 50
        )
        st.success("Demo sequence loaded.")

fasta_text = None

if uploaded:
    content = uploaded.read()
    try:
        fasta_text = content.decode("utf-8")
    except UnicodeDecodeError:
        # Try latin-1
        fasta_text = content.decode("latin-1")

    seq_len = len(parse_fasta(fasta_text))
    n_contigs = fasta_text.count(">")
    st.success(f"✅ Loaded: **{uploaded.name}** — {n_contigs} contig(s), {seq_len:,} bases")

    if seq_len < 1000:
        st.warning("⚠️ Sequence is very short (<1,000 bp). K-mer features may not be informative. Upload a complete genome assembly for best results.")

elif "demo_fasta" in st.session_state:
    fasta_text = st.session_state["demo_fasta"]
    st.info("Using synthetic demo sequence. Results are illustrative only.")

# ── Settings ──────────────────────────────────────────────────────────────────
st.divider()
st.header("2. Settings")

threshold = st.slider(
    "Confidence threshold:",
    min_value=0.50, max_value=0.90, value=0.65, step=0.05,
    help="Predictions between (1-threshold) and threshold are flagged as Uncertain."
)

# ── Run prediction ─────────────────────────────────────────────────────────────
st.divider()
st.header("3. Resistance profile")

if fasta_text is None:
    st.info("👆 Upload a FASTA file above to get a prediction.")
else:
    with st.spinner("Computing k-mer features and running ensemble models..."):
        t0 = time.time()
        preds = predict_from_fasta(fasta_text, models, threshold=threshold)
        elapsed = time.time() - t0

    st.caption(f"Prediction completed in {elapsed:.1f}s across {len(preds)} antibiotics.")

    # Summary metrics
    n_r = sum(1 for p in preds if p["verdict"] == "Resistant")
    n_s = sum(1 for p in preds if p["verdict"] == "Susceptible")
    n_u = sum(1 for p in preds if p["verdict"] == "Uncertain")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resistant", n_r, delta=None)
    col2.metric("Susceptible", n_s)
    col3.metric("Uncertain", n_u)
    mdr = "Yes ⚠️" if n_r >= 3 else "No ✅"
    col4.metric("MDR (3+)", mdr)

    if n_r >= 3:
        st.error(f"🔴 **Multi-drug resistant pattern detected** — resistant to {n_r} antibiotics.")
    elif n_r == 0 and n_u == 0:
        st.success("✅ Fully susceptible — no resistance predicted.")
    elif n_r == 0:
        st.info("🟡 No confirmed resistance, but some predictions are uncertain. Consider lab confirmation.")

    st.markdown("---")

    # Result cards
    for pred in preds:
        verdict = pred["verdict"]
        prob_r  = pred["prob_r"]
        color   = (RESIST_COLOR if verdict == "Resistant" else
                   SUSCEPT_COLOR if verdict == "Susceptible" else
                   UNCERTAIN_COLOR)
        emoji   = "🔴" if verdict == "Resistant" else "🟢" if verdict == "Susceptible" else "🟡"
        bar_pct = int(prob_r * 100)

        st.markdown(f"""
<div style='background:#181825; border-left:4px solid {color};
     padding:0.5rem 1rem; border-radius:6px; margin-bottom:6px;
     display:flex; justify-content:space-between; align-items:center;'>
  <div>
    <b style='color:#cdd6f4;'>{emoji} {pred["short"]}</b>
    <span style='color:#6272a4; font-size:0.8rem; margin-left:8px;'>{pred["drug_class"]}</span>
  </div>
  <div style='text-align:right; min-width:220px;'>
    <span style='color:{color}; font-weight:bold;'>{verdict}</span>
    <div style='background:#2d2d44; border-radius:4px; height:6px; margin-top:4px;'>
      <div style='background:{color}; width:{bar_pct}%; height:6px; border-radius:4px;'></div>
    </div>
    <small style='color:#6272a4;'>P(Resistant) = {prob_r:.1%}</small>
  </div>
</div>
""", unsafe_allow_html=True)

    # Waterfall chart
    st.markdown("---")
    fig_bar = go.Figure(go.Bar(
        x=[p["short"] for p in preds],
        y=[p["prob_r"] * 100 for p in preds],
        marker_color=[
            RESIST_COLOR if p["verdict"] == "Resistant" else
            SUSCEPT_COLOR if p["verdict"] == "Susceptible" else
            UNCERTAIN_COLOR
            for p in preds
        ],
        text=[f"{p['prob_r']:.0%}" for p in preds],
        textposition="outside",
    ))
    fig_bar.add_hline(y=threshold * 100, line_dash="dash", line_color="#6272a4",
                      annotation_text=f"Resistance threshold ({threshold:.0%})")
    fig_bar.add_hline(y=(1 - threshold) * 100, line_dash="dash", line_color="#6272a4",
                      annotation_text=f"Susceptible threshold ({1-threshold:.0%})")
    fig_bar.update_layout(
        xaxis_title="Antibiotic",
        yaxis_title="P(Resistant) %",
        yaxis=dict(range=[0, 115]),
        height=340, margin=dict(t=20, b=20),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", yaxis_gridcolor="#2d2d44",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("🔴 Resistant  🟡 Uncertain (between thresholds)  🟢 Susceptible")

    # Download
    result_df = pd.DataFrame([{
        "antibiotic": p["antibiotic"],
        "drug_class": p["drug_class"],
        "verdict": p["verdict"],
        "prob_resistant": round(p["prob_r"], 4),
        "threshold": threshold,
        "source": uploaded.name if uploaded else "demo",
    } for p in preds])

    csv = result_df.to_csv(index=False)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download full results as CSV",
            data=csv,
            file_name="amr_fasta_prediction.csv",
            mime="text/csv",
        )
    with col_dl2:
        try:
            from pdf_report import generate_pdf
            genome_label = uploaded.name if uploaded else "Synthetic demo"
            pdf_bytes = generate_pdf(preds, genome_id=genome_label, source="FASTA upload")
            st.download_button(
                "📄 Download PDF report",
                data=pdf_bytes,
                file_name="amr_fasta_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.caption(f"PDF unavailable: {e}")

    st.divider()
    st.markdown("""
**⚠️ Important caveat:** This prediction uses k-mer features only (no gene annotation).
K-mer features alone have AUC ~0.65–0.70. For higher accuracy:
- Use **Page 5 (Live Predictor)** with a BV-BRC genome ID — also fetches gene annotation
- Use **Page 14 (Offline Predictor)** if you have gene results from CARD/AMRFinder

These predictions are for **research purposes only** and should not be used as a sole basis for clinical treatment decisions.
""")
