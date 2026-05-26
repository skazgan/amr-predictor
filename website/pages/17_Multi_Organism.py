"""
Page 17 — Multi-Organism Predictor

Predict antibiotic resistance across 4 organisms:
  - Klebsiella pneumoniae (existing models, full feature set)
  - Escherichia coli      (gene-based models)
  - Staphylococcus aureus (gene-based models)
  - Acinetobacter baumannii (gene-based models)
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT      = Path(__file__).parent.parent.parent
ART_DIR   = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css

inject_mobile_css()
st.title("🦠 Multi-Organism AMR Predictor")
st.markdown("*Predict antibiotic resistance across four clinically important pathogens.*")
st.divider()

# ── Organism config ───────────────────────────────────────────────────────────
ORGANISM_META = {
    "klebsiella_pneumoniae": {
        "display":  "Klebsiella pneumoniae",
        "emoji":    "🔴",
        "desc":     "Leading cause of hospital-acquired infections. ST258 is the dominant MDR lineage worldwide.",
        "gram":     "Gram-negative",
        "concern":  "Critical priority (WHO)",
        "color":    "#e94560",
        "drug_class": {
            "ciprofloxacin": "Fluoroquinolone", "meropenem": "Carbapenem",
            "gentamicin": "Aminoglycoside", "tetracycline": "Tetracycline",
            "trimethoprim/sulfamethoxazole": "Folate inhibitor", "cefepime": "Cephalosporin",
            "amikacin": "Aminoglycoside", "imipenem": "Carbapenem",
            "piperacillin/tazobactam": "Beta-lactam/inh.", "levofloxacin": "Fluoroquinolone",
        },
        "short": {
            "ciprofloxacin": "Cipro", "meropenem": "Mero", "gentamicin": "Gent",
            "tetracycline": "Tet", "trimethoprim/sulfamethoxazole": "TMP/SMX",
            "cefepime": "Cef", "amikacin": "Amik", "imipenem": "Imi",
            "piperacillin/tazobactam": "Pip/Taz", "levofloxacin": "Levo",
        },
    },
    "escherichia_coli": {
        "display":  "Escherichia coli",
        "emoji":    "🟠",
        "desc":     "Most common cause of urinary tract infections and a major ESBL/carbapenemase reservoir.",
        "gram":     "Gram-negative",
        "concern":  "Critical priority (WHO)",
        "color":    "#F97316",
        "drug_class": {
            "ciprofloxacin": "Fluoroquinolone", "meropenem": "Carbapenem",
            "gentamicin": "Aminoglycoside", "tetracycline": "Tetracycline",
            "trimethoprim/sulfamethoxazole": "Folate inhibitor", "cefepime": "Cephalosporin",
            "amikacin": "Aminoglycoside", "ampicillin": "Penicillin",
            "ceftriaxone": "Cephalosporin", "piperacillin/tazobactam": "Beta-lactam/inh.",
        },
        "short": {
            "ciprofloxacin": "Cipro", "meropenem": "Mero", "gentamicin": "Gent",
            "tetracycline": "Tet", "trimethoprim/sulfamethoxazole": "TMP/SMX",
            "cefepime": "Cef", "amikacin": "Amik", "ampicillin": "Amp",
            "ceftriaxone": "Ceftri", "piperacillin/tazobactam": "Pip/Taz",
        },
    },
    "staphylococcus_aureus": {
        "display":  "Staphylococcus aureus",
        "emoji":    "🟡",
        "desc":     "MRSA (methicillin-resistant S. aureus) is a leading cause of skin, bloodstream, and surgical infections.",
        "gram":     "Gram-positive",
        "concern":  "High priority (WHO)",
        "color":    "#D97706",
        "drug_class": {
            "oxacillin": "Beta-lactam (MRSA marker)", "vancomycin": "Glycopeptide",
            "tetracycline": "Tetracycline", "trimethoprim/sulfamethoxazole": "Folate inhibitor",
            "clindamycin": "Lincosamide", "erythromycin": "Macrolide",
            "ciprofloxacin": "Fluoroquinolone", "gentamicin": "Aminoglycoside",
        },
        "short": {
            "oxacillin": "Oxa", "vancomycin": "Vanc", "tetracycline": "Tet",
            "trimethoprim/sulfamethoxazole": "TMP/SMX", "clindamycin": "Clinda",
            "erythromycin": "Ery", "ciprofloxacin": "Cipro", "gentamicin": "Gent",
        },
    },
    "acinetobacter_baumannii": {
        "display":  "Acinetobacter baumannii",
        "emoji":    "🔵",
        "desc":     "Extremely drug-resistant nosocomial pathogen. Carbapenem resistance makes infections nearly untreatable.",
        "gram":     "Gram-negative",
        "concern":  "Critical priority (WHO)",
        "color":    "#3B82F6",
        "drug_class": {
            "meropenem": "Carbapenem", "imipenem": "Carbapenem",
            "ciprofloxacin": "Fluoroquinolone", "gentamicin": "Aminoglycoside",
            "amikacin": "Aminoglycoside", "colistin": "Polymyxin (last resort)",
            "tetracycline": "Tetracycline", "trimethoprim/sulfamethoxazole": "Folate inhibitor",
        },
        "short": {
            "meropenem": "Mero", "imipenem": "Imi", "ciprofloxacin": "Cipro",
            "gentamicin": "Gent", "amikacin": "Amik", "colistin": "Colist",
            "tetracycline": "Tet", "trimethoprim/sulfamethoxazole": "TMP/SMX",
        },
    },
}

RESIST_COLOR   = "#e94560"
SUSCEPT_COLOR  = "#16A34A"
UNCERTAIN_COLOR = "#D97706"


@st.cache_resource(show_spinner=False)
def load_org_models(org_name: str) -> dict:
    """Load all available models for an organism."""
    models = {}
    org_model_dir = MODEL_DIR / org_name
    if not org_model_dir.exists():
        return models
    for p in org_model_dir.glob("*.pkl"):
        ab = p.stem.replace("_", "/", 1) if "trimethoprim" not in p.stem else p.stem.replace("_", " ", 1)
        # Reconstruct proper antibiotic name from filename
        ab_name = p.stem.replace("_", "/")
        # Fix common patterns
        ab_name = ab_name.replace("trimethoprim/sulfamethoxazole", "trimethoprim/sulfamethoxazole")
        ab_name = ab_name.replace("piperacillin/tazobactam", "piperacillin/tazobactam")
        try:
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            models[bundle.get("antibiotic", p.stem)] = bundle
        except Exception:
            pass
    return models


def predict_from_genes_multi(gene_presence: dict, models: dict,
                              threshold: float = 0.65) -> list:
    results = []
    for ab, bundle in models.items():
        model    = bundle["model"]
        features = bundle["features"]

        x = np.zeros(len(features))
        for i, feat in enumerate(features):
            gene_name = feat.replace("gene__", "", 1)
            x[i] = gene_presence.get(gene_name, 0)

        X_input = pd.DataFrame([x], columns=features)
        prob_r  = float(model.predict_proba(X_input)[0, 1])

        verdict = ("Resistant"   if prob_r >= threshold else
                   "Susceptible" if prob_r <= 1 - threshold else
                   "Uncertain")
        results.append({
            "antibiotic": ab,
            "prob_r":     prob_r,
            "verdict":    verdict,
        })
    return sorted(results, key=lambda x: -x["prob_r"])


# ── Section 1: Organism cards ─────────────────────────────────────────────────
st.header("1. The four pathogens")

cols = st.columns(4)
for i, (org_key, meta) in enumerate(ORGANISM_META.items()):
    models_available = (MODEL_DIR / org_key).exists() and any((MODEL_DIR / org_key).glob("*.pkl"))
    status = "✅ Models ready" if models_available else "⏳ Training pending"
    with cols[i]:
        status_color = "#16A34A" if "ready" in status else "#D97706"
        st.markdown(f"""
<div style='background:#FFFFFF; border:1.5px solid {meta["color"]}40;
     border-top:4px solid {meta["color"]};
     padding:0.9rem 1rem; border-radius:10px; min-height:185px;
     box-shadow:0 2px 6px rgba(0,0,0,0.06);'>
  <div style='font-size:1.8rem; margin-bottom:4px;'>{meta["emoji"]}</div>
  <b style='color:#1E293B; font-size:0.92rem; font-style:italic;'>{meta["display"]}</b><br>
  <span style='color:{meta["color"]}; font-size:0.72rem; font-weight:600;'>{meta["concern"]}</span><br>
  <small style='color:#475569; font-size:0.76rem; line-height:1.4;'>{meta["desc"][:80]}…</small><br>
  <small style='color:{status_color}; font-weight:600;'>{status}</small>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Section 2: Why these organisms? ──────────────────────────────────────────
st.header("2. Why these four pathogens?")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
These four species represent the **ESKAPE pathogens** most amenable to genomic AMR prediction:

| Organism | Clinical role | Key resistance |
|---|---|---|
| *K. pneumoniae* | Hospital pneumonia, UTI, bacteraemia | KPC, NDM carbapenemases |
| *E. coli* | UTI, sepsis, neonatal meningitis | ESBL (CTX-M), PMQR |
| *S. aureus* | Skin, surgical site, endocarditis | MRSA (mecA), VRSA |
| *A. baumannii* | ICU pneumonia, wound infections | OXA carbapenemases |

Together they account for **>60% of hospital-acquired infections** globally.
""")

with col2:
    st.markdown("""
**Why genomic prediction works differently per organism:**

- *K. pneumoniae* & *E. coli* — resistance spreads via **plasmids** (horizontal gene transfer):
  gene-based models are most powerful
- *S. aureus* — MRSA resistance is chromosomal (**mecA** gene):
  single-gene presence is nearly diagnostic
- *A. baumannii* — resistance accumulates via **integrons** (gene cassette arrays):
  co-resistance patterns are extreme; nearly pan-resistant strains exist

**Feature note:** New organism models use gene presence/absence only
(k-mer features require downloading FASTA sequences per genome — coming in a future update).
Gene-only models achieve AUC 0.97–1.00 for most antibiotics — resistance genes are near-perfect predictors.
""")

st.divider()

# ── Section 3: Organism selector & predictor ─────────────────────────────────
st.header("3. Predict resistance by organism")

org_options = {meta["display"]: key for key, meta in ORGANISM_META.items()}
selected_display = st.selectbox(
    "Select organism:",
    options=list(org_options.keys()),
    format_func=lambda x: f"{ORGANISM_META[org_options[x]]['emoji']}  {x}",
)
selected_org = org_options[selected_display]
meta = ORGANISM_META[selected_org]

# K. pneumoniae has dedicated predictors (uses k-mer + gene features, not gene-only)
if selected_org == "klebsiella_pneumoniae":
    st.info("""
**K. pneumoniae has dedicated predictors** that use the full feature set
(k-mer frequencies + gene presence) for higher accuracy (AUC 0.76–0.89).

The multi-organism predictor uses gene-only features designed for the three
new organisms. For K. pneumoniae, use one of the dedicated pages instead:
""")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.page_link("pages/5_Live_Predictor.py",
                     label="🔮 Live Predictor  (BV-BRC genome ID)",
                     use_container_width=True)
    with col_b:
        st.page_link("pages/14_Offline_Predictor.py",
                     label="⚡ Offline Predictor  (gene toggles)",
                     use_container_width=True)
    with col_c:
        st.page_link("pages/15_FASTA_Upload.py",
                     label="📂 FASTA Upload  (raw assembly)",
                     use_container_width=True)
    st.stop()

with st.spinner(f"Loading {meta['display']} models..."):
    org_models = load_org_models(selected_org)

if not org_models:
    st.warning(f"""
**No models found for {meta['display']}.**

To train models locally, run:
```bash
python src/fetch_multi_organism.py
python src/train_organisms.py --organism {selected_org}
```
""")
else:
    st.success(f"✅ {len(org_models)} models loaded for {meta['display']}")

    # Gene selection
    st.subheader("Select resistance genes present in your strain")

    # Get all gene features from models
    all_genes = set()
    for bundle in org_models.values():
        for feat in bundle["features"]:
            all_genes.add(feat.replace("gene__", "", 1))
    all_genes = sorted(all_genes)

    col_search, col_btn, col_threshold = st.columns([3, 1.2, 1])
    with col_search:
        search = st.text_input("Search genes:", placeholder="e.g. bla, gyr, mec, van...",
                               label_visibility="collapsed")
    with col_threshold:
        threshold = st.slider("Confidence threshold", 0.50, 0.90, 0.65, 0.05,
                              label_visibility="collapsed")

    # Session state for selected genes (persists across interactions)
    state_key = f"selected_genes_{selected_org}"
    if state_key not in st.session_state:
        st.session_state[state_key] = set()

    if search:
        filtered = [g for g in all_genes if search.lower() in g.lower()]
        n_show = min(len(filtered), 30)

        with col_btn:
            if st.button(f"✅ Select all {len(filtered)}", use_container_width=True,
                         help=f"Add all {len(filtered)} matching genes to selection"):
                st.session_state[state_key].update(filtered)
                st.rerun()

        col_clear = st.columns([1])[0]
        st.caption(
            f"**{len(filtered)}** genes matching **'{search}'** "
            f"(showing {n_show}) — {len(st.session_state[state_key])} selected total"
        )

        cols_g = st.columns(3)
        for i, gene in enumerate(filtered[:30]):
            checked = gene in st.session_state[state_key]
            new_val = cols_g[i % 3].checkbox(
                gene[:55], value=checked, key=f"mg_{selected_org}_{gene}"
            )
            if new_val:
                st.session_state[state_key].add(gene)
            else:
                st.session_state[state_key].discard(gene)

        if len(filtered) > 30:
            st.caption(f"Showing first 30 of {len(filtered)}. Refine your search or use 'Select all'.")
    else:
        with col_btn:
            if st.button("🗑 Clear all", use_container_width=True,
                         help="Clear all selected genes"):
                st.session_state[state_key] = set()
                st.rerun()
        if st.session_state[state_key]:
            st.info(f"🔍 **{len(st.session_state[state_key])} genes selected.** "
                    f"Type a gene name to add more, or use a preset below.")
        else:
            st.info(f"🔍 Type a gene name to search {len(all_genes):,} resistance genes for "
                    f"**{meta['display']}**, or use a preset below.")

    selected_genes = st.session_state[state_key].copy()

    # Organism-specific quick presets
    st.markdown("**Quick presets:**")
    presets = {}
    if selected_org == "klebsiella_pneumoniae":
        presets = {
            "ST258 KPC (carbapenem-resistant)": ["KPC", "CTX-M", "SHV", "OqxAB"],
            "ESBL only (carbapenem-susceptible)": ["CTX-M", "TEM", "gyrA"],
        }
    elif selected_org == "escherichia_coli":
        presets = {
            "ESBL UTI strain": ["CTX-M", "TEM", "aac(6')-Ib-cr", "gyrA"],
            "Carbapenem-resistant (NDM)": ["NDM", "CTX-M", "OXA-48"],
        }
    elif selected_org == "staphylococcus_aureus":
        presets = {
            "MRSA (hospital)": ["mecA", "aac(6')-aph(2'')"],
            "MSSA (susceptible)": [],
        }
    elif selected_org == "acinetobacter_baumannii":
        presets = {
            "Carbapenem-resistant (OXA-23)": ["OXA-23", "OXA-51", "armA"],
            "Pan-susceptible baseline": [],
        }

    preset_cols = st.columns(len(presets) + 1)
    for i, (label, genes_pat) in enumerate(presets.items()):
        if preset_cols[i].button(label, use_container_width=True):
            st.session_state[state_key] = set()
            for g_pat in genes_pat:
                matches = [g for g in all_genes if g_pat.lower() in g.lower()]
                st.session_state[state_key].update(matches)
            st.rerun()
    if preset_cols[len(presets)].button("🗑 Reset", use_container_width=True):
        st.session_state[state_key] = set()
        st.rerun()

    selected_genes = st.session_state[state_key].copy()

    # Run prediction
    if selected_genes:
        gene_presence = {g: 1 for g in selected_genes}
        preds = predict_from_genes_multi(gene_presence, org_models, threshold)

        n_r = sum(1 for p in preds if p["verdict"] == "Resistant")
        n_s = sum(1 for p in preds if p["verdict"] == "Susceptible")
        n_u = sum(1 for p in preds if p["verdict"] == "Uncertain")

        # MDR badge
        if n_r >= 3:
            st.error(f"🔴 **Multi-drug resistant** — resistant to {n_r}/{len(preds)} tested antibiotics")
        elif n_r == 0:
            st.success("✅ No resistance predicted")
        else:
            st.warning(f"🟡 Partial resistance — {n_r} antibiotic(s) affected")

        # Result cards
        drug_class = meta.get("drug_class", {})
        short_names = meta.get("short", {})
        for pred in preds:
            ab      = pred["antibiotic"]
            verdict = pred["verdict"]
            prob_r  = pred["prob_r"]
            color   = (RESIST_COLOR if verdict == "Resistant" else
                       SUSCEPT_COLOR if verdict == "Susceptible" else
                       UNCERTAIN_COLOR)
            emoji   = "🔴" if verdict == "Resistant" else "🟢" if verdict == "Susceptible" else "🟡"
            d_class = drug_class.get(ab, "")
            short   = short_names.get(ab, ab[:8])
            bar_pct = int(prob_r * 100)

            st.markdown(f"""
<div style='background:#F8F9FF; border-left:4px solid {color};
     padding:0.5rem 1rem; border-radius:6px; margin-bottom:6px;
     display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem;'>
  <div>
    <b style='color:#1E293B;'>{emoji} {short}</b>
    <span style='color:#64748B; font-size:0.8rem; margin-left:8px;'>{d_class}</span>
  </div>
  <div style='text-align:right; min-width:200px;'>
    <span style='color:{color}; font-weight:bold;'>{verdict}</span>
    <div style='background:#EEF2FF; border-radius:4px; height:6px; margin-top:4px;'>
      <div style='background:{color}; width:{bar_pct}%; height:6px; border-radius:4px;'></div>
    </div>
    <small style='color:#64748B;'>P(R) = {prob_r:.1%}</small>
  </div>
</div>
""", unsafe_allow_html=True)

        # Download
        result_df = pd.DataFrame([{
            "organism":  meta["display"],
            "antibiotic": p["antibiotic"],
            "drug_class": drug_class.get(p["antibiotic"], ""),
            "verdict":    p["verdict"],
            "prob_resistant": round(p["prob_r"], 4),
        } for p in preds])

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download results as CSV",
                data=result_df.to_csv(index=False),
                file_name=f"amr_{selected_org}_prediction.csv",
                mime="text/csv",
            )
        with col_dl2:
            try:
                from pdf_report import generate_pdf
                pdf_preds = [{**p, "short": short_names.get(p["antibiotic"], p["antibiotic"][:8]),
                              "drug_class": drug_class.get(p["antibiotic"], "")} for p in preds]
                pdf_bytes = generate_pdf(pdf_preds, genome_id=f"{meta['display']} (gene toggle)",
                                         source="Multi-Organism Predictor")
                st.download_button(
                    "📄 Download PDF report",
                    data=pdf_bytes,
                    file_name=f"amr_{selected_org}_report.pdf",
                    mime="application/pdf",
                )
            except Exception:
                pass

st.divider()

# ── Section 4: Cross-organism comparison ─────────────────────────────────────
st.header("4. Cross-organism resistance comparison")
st.markdown("Compare resistance rates across all four organisms for shared antibiotics.")

summary_path = ART_DIR / "multi_org_summary.json"
kp_summary   = ART_DIR / "summary.json"

all_summary = []

# Load K. pneumoniae summary (existing)
if kp_summary.exists():
    kp = json.loads(kp_summary.read_text())
    for r in kp:
        all_summary.append({
            "organism":   "K. pneumoniae",
            "antibiotic": r["antibiotic"],
            "auc":        r["test_auc"],
        })

# Load multi-organism summary
if summary_path.exists():
    multi = json.loads(summary_path.read_text())
    org_short = {
        "escherichia_coli":      "E. coli",
        "staphylococcus_aureus": "S. aureus",
        "acinetobacter_baumannii": "A. baumannii",
    }
    for r in multi:
        all_summary.append({
            "organism":   org_short.get(r["organism"], r["organism"]),
            "antibiotic": r["antibiotic"],
            "auc":        r["test_auc"],
        })

if len(all_summary) > 1:
    df_sum = pd.DataFrame(all_summary)
    shared_abs = df_sum.groupby("antibiotic")["organism"].nunique()
    shared_abs = shared_abs[shared_abs > 1].index.tolist()

    if shared_abs:
        df_shared = df_sum[df_sum["antibiotic"].isin(shared_abs)]
        fig_comp = px.bar(
            df_shared, x="antibiotic", y="auc", color="organism",
            barmode="group",
            color_discrete_map={
                "K. pneumoniae": "#e94560",
                "E. coli":       "#ffb86c",
                "S. aureus":     "#f1fa8c",
                "A. baumannii":  "#8be9fd",
            },
            labels={"auc": "ROC-AUC (20% holdout)", "antibiotic": "Antibiotic"},
        )
        fig_comp.add_hline(y=0.8, line_dash="dash", line_color="#50fa7b",
                           annotation_text="Good threshold (0.80)")
        fig_comp.add_hline(y=0.5, line_dash="dot", line_color="#64748B",
                           annotation_text="Random baseline")
        fig_comp.update_layout(
            height=400, margin=dict(t=20, b=20),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
            yaxis=dict(range=[0.4, 1.0]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("Grouped bars = model AUC per organism. Higher = better prediction accuracy.")
    else:
        st.info("Train models for multiple organisms to enable cross-organism comparison.")
else:
    st.info("Cross-organism comparison will appear here once models are trained for additional organisms. Run `python src/train_organisms.py` to get started.")

st.divider()

# ── Section 5: Clinical context ───────────────────────────────────────────────
st.header("5. Clinical significance by organism")

st.markdown("""
| Organism | Global burden | Key resistance threat | Why genomics helps |
|---|---|---|---|
| *K. pneumoniae* | ~700K deaths/year | Carbapenem resistance (KPC, NDM) | ST258 outbreak clone detectable from genome alone |
| *E. coli* | #1 cause of UTI worldwide | ESBL (CTX-M-15) globally spreading | Gene combinations predict treatment failure |
| *S. aureus* | 170K deaths/year (MRSA) | mecA → oxacillin resistance | Single gene nearly diagnostic for MRSA |
| *A. baumannii* | Rising in ICUs | Pan-resistance via OXA integrons | Early detection critical — few drugs left |

**The common thread:** In all four organisms, resistance is encoded in the genome.
Sequencing the pathogen gives you the resistance profile *before* lab susceptibility results arrive —
the key advantage of genomics-based prediction.
""")
