"""
Page 23 — MIC Prediction
Go beyond R/S binary — predict minimum inhibitory concentration ranges.
"""
import json
import pickle
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

ROOT      = Path(__file__).parent.parent.parent
ART_DIR   = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
inject_mobile_css()

st.title("🔬 MIC Prediction")
st.markdown("*Beyond Resistant / Susceptible — predicting the actual inhibitory concentration.*")
st.divider()

# ── EUCAST breakpoints (embedded — used even without artifact) ────────────────
EUCAST_BP = {
    "meropenem":                    (2.0,  8.0),
    "imipenem":                     (2.0,  8.0),
    "ciprofloxacin":                (0.25, 1.0),
    "gentamicin":                   (2.0,  4.0),
    "amikacin":                     (8.0,  16.0),
    "piperacillin/tazobactam":      (8.0,  16.0),
    "trimethoprim/sulfamethoxazole": (2.0,  4.0),
    "ceftazidime":                  (1.0,  4.0),
    "ceftriaxone":                  (1.0,  2.0),
    "tetracycline":                 (1.0,  8.0),
    "levofloxacin":                 (1.0,  2.0),
    "cefepime":                     (1.0,  4.0),
}

MIC_DILUTIONS = [0.001, 0.002, 0.004, 0.008, 0.016, 0.031, 0.063,
                 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

MIC_CAT_COLOR = {"Low": "#10B981", "Intermediate": "#F59E0B", "High": "#EF4444"}
MIC_CAT_BG    = {"Low": "#DCFCE7", "Intermediate": "#FEF9C3", "High": "#FEE2E2"}
MIC_CAT_DESC  = {
    "Low":          "MIC ≤ susceptible breakpoint — standard dosing likely effective",
    "Intermediate": "MIC between breakpoints — may require higher dose or combination therapy",
    "High":         "MIC > resistant breakpoint — standard therapy likely to fail",
}

# ── Section 1: What is MIC? ───────────────────────────────────────────────────
st.header("1. What is MIC and why does it matter?")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
The **Minimum Inhibitory Concentration (MIC)** is the lowest concentration of an antibiotic
that prevents visible bacterial growth after overnight incubation.

It is the **gold standard** for antibiotic susceptibility testing.

**Why go beyond R/S?**

The binary Resistant/Susceptible classification is set at fixed MIC breakpoints, but
clinical outcomes depend on how *far* a strain's MIC is from the breakpoint:

| MIC relative to breakpoint | Clinical implication |
|---|---|
| MIC = 0.001 mg/L (far below S) | Drug is extremely effective — low dose sufficient |
| MIC = 0.5× breakpoint | Effective — standard dosing |
| MIC = 0.9× breakpoint | Still "Susceptible" but borderline — consider higher dose |
| MIC = 1.5× breakpoint | Technically "Resistant" but may respond to dose escalation |
| MIC >> breakpoint | Resistant — drug will fail regardless of dose |

**Three categories** this predictor uses:
- 🟢 **Low** — MIC ≤ EUCAST susceptible breakpoint
- 🟡 **Intermediate** — MIC between susceptible and resistant breakpoints
- 🔴 **High** — MIC > resistant breakpoint
""")
with col2:
    # Illustrative MIC number line
    example_ab = "meropenem"
    bp_s, bp_r = EUCAST_BP[example_ab]
    mics = [0.008, 0.016, 0.031, 0.063, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    colors = [
        "#10B981" if m <= bp_s else "#F59E0B" if m <= bp_r else "#EF4444"
        for m in mics
    ]
    fig_ill = go.Figure(go.Bar(
        x=[str(m) for m in mics],
        y=[1] * len(mics),
        marker_color=colors,
        text=["S" if m <= bp_s else "I" if m <= bp_r else "R" for m in mics],
        textposition="inside",
        textfont=dict(color="white", size=11),
    ))
    fig_ill.add_vline(x=str(bp_s), line_dash="dash", line_color="#065F46",
                      annotation_text=f"S≤{bp_s}", annotation_font_color="#065F46",
                      annotation_position="top")
    fig_ill.add_vline(x=str(bp_r), line_dash="dash", line_color="#991B1B",
                      annotation_text=f"R>{bp_r}", annotation_font_color="#991B1B",
                      annotation_position="top")
    fig_ill.update_layout(
        title=f"Meropenem MIC scale (EUCAST)",
        yaxis=dict(visible=False),
        xaxis=dict(title="MIC (mg/L)"),
        height=220, margin=dict(t=50, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B",
    )
    st.plotly_chart(fig_ill, use_container_width=True)
    st.caption("🟢 Low (susceptible)  🟡 Intermediate  🔴 High (resistant)")

st.divider()

# ── Section 2: MIC distributions ─────────────────────────────────────────────
st.header("2. MIC distributions from 85,000+ K. pneumoniae genomes")

mic_dist_path = ART_DIR / "mic_distributions.json"

if mic_dist_path.exists():
    mic_dist = json.loads(mic_dist_path.read_text())
    available_abs = list(mic_dist.keys())
else:
    st.warning("""
MIC distribution data not yet generated. Run to fetch and process:
```bash
python src/train_mic.py --fetch
```
""")
    available_abs = list(EUCAST_BP.keys())
    mic_dist = {}

ab_sel_dist = st.selectbox("Select antibiotic:", available_abs, key="mic_dist_ab")

if ab_sel_dist in mic_dist:
    d = mic_dist[ab_sel_dist]
    bp_s, bp_r = d["eucast_s"], d["eucast_r"]

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Genomes with MIC data", f"{d['n']:,}")
    m2.metric("MIC₅₀ (median)", f"{d['mic_50']} mg/L")
    m3.metric("MIC₉₀ (90th pct.)", f"{d['mic_90']} mg/L")
    m4.metric("MIC range", f"{d['mic_min']}–{d['mic_max']} mg/L")

    # Histogram of log2(MIC)
    mics    = d["mic_values"]
    labels  = d["mic_labels"]
    log_mics = [np.log2(max(m, 0.001)) for m in mics]

    fig_hist = go.Figure()
    for phenotype, color in [("Susceptible", "#10B981"), ("Resistant", "#EF4444")]:
        subset_log = [l for l, lab in zip(log_mics, labels) if lab == phenotype]
        if subset_log:
            fig_hist.add_trace(go.Histogram(
                x=subset_log,
                name=phenotype,
                marker_color=color,
                opacity=0.7,
                nbinsx=30,
            ))

    log_bp_s = np.log2(bp_s)
    log_bp_r = np.log2(bp_r)
    fig_hist.add_vline(x=log_bp_s, line_dash="dash", line_color="#065F46",
                       annotation_text=f"S≤{bp_s} mg/L",
                       annotation_font_color="#065F46")
    fig_hist.add_vline(x=log_bp_r, line_dash="dash", line_color="#991B1B",
                       annotation_text=f"R>{bp_r} mg/L",
                       annotation_font_color="#991B1B")

    # Shade regions
    fig_hist.add_vrect(x0=-15, x1=log_bp_s,
                       fillcolor="#DCFCE7", opacity=0.15, layer="below",
                       annotation_text="Low", annotation_font_color="#065F46",
                       annotation_position="top left")
    fig_hist.add_vrect(x0=log_bp_s, x1=log_bp_r,
                       fillcolor="#FEF9C3", opacity=0.25, layer="below",
                       annotation_text="Intermediate", annotation_font_color="#92400E",
                       annotation_position="top left")
    fig_hist.add_vrect(x0=log_bp_r, x1=15,
                       fillcolor="#FEE2E2", opacity=0.15, layer="below",
                       annotation_text="High", annotation_font_color="#991B1B",
                       annotation_position="top right")

    # Custom tick labels showing actual MIC values
    tick_vals = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    tick_text = [f"{2**v:.3f}" for v in tick_vals]
    fig_hist.update_layout(
        barmode="overlay",
        xaxis=dict(title="MIC (mg/L) — log₂ scale",
                   tickvals=tick_vals, ticktext=tick_text, tickangle=-45),
        yaxis=dict(title="Count"),
        height=380, margin=dict(t=20, b=60),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(f"EUCAST 2024 breakpoints: Susceptible ≤{bp_s} mg/L | Resistant >{bp_r} mg/L. "
               f"Intermediate zone = dose-dependent susceptibility.")

    # Per-category counts
    n_low  = sum(1 for m in mics if m <= bp_s)
    n_inter = sum(1 for m in mics if bp_s < m <= bp_r)
    n_high = sum(1 for m in mics if m > bp_r)
    n_total = len(mics)

    cat_cols = st.columns(3)
    for col, (cat, n, bg, tc) in zip(cat_cols, [
        ("Low",           n_low,  "#DCFCE7", "#15803D"),
        ("Intermediate",  n_inter, "#FEF9C3", "#92400E"),
        ("High",          n_high, "#FEE2E2",  "#991B1B"),
    ]):
        with col:
            pct = n / n_total * 100 if n_total > 0 else 0
            st.markdown(f"""
<div style='background:{bg}; border-radius:8px; padding:0.7rem 1rem; text-align:center;'>
  <div style='font-size:1.6rem; font-weight:700; color:{tc};'>{n:,}</div>
  <div style='color:{tc}; font-weight:600;'>{cat}</div>
  <div style='color:{tc}; font-size:0.85rem;'>{pct:.1f}% of isolates</div>
</div>""", unsafe_allow_html=True)

else:
    # Show breakpoints only (no distribution data yet)
    if ab_sel_dist in EUCAST_BP:
        bp_s, bp_r = EUCAST_BP[ab_sel_dist]
        st.info(f"**EUCAST 2024 breakpoints for {ab_sel_dist}:** "
                f"Susceptible ≤{bp_s} mg/L | Resistant >{bp_r} mg/L\n\n"
                "Fetch MIC data with `python src/train_mic.py --fetch` to see distributions.")

st.divider()

# ── Section 3: MIC predictor ──────────────────────────────────────────────────
st.header("3. Predict MIC category from resistance genes")

# Check for MIC models
mic_model_dir = MODEL_DIR / "mic"
mic_models_available = mic_model_dir.exists() and any(mic_model_dir.glob("*.pkl"))

if not mic_models_available:
    st.info("""
**MIC prediction models not yet trained.** Run to train them:
```bash
python src/train_mic.py --fetch --train
```
Training uses gene presence/absence to predict Low / Intermediate / High MIC category.
Expected AUC: 0.75–0.95 depending on antibiotic.

While you wait, use Section 4 to explore the EUCAST breakpoints and MIC scales interactively.
""")

else:
    @st.cache_resource(show_spinner=False)
    def load_mic_models():
        models = {}
        for p in mic_model_dir.glob("*.pkl"):
            try:
                with open(p, "rb") as f:
                    bundle = pickle.load(f)
                models[bundle["antibiotic"]] = bundle
            except Exception:
                pass
        return models

    mic_models = load_mic_models()

    if mic_models:
        st.success(f"✅ {len(mic_models)} MIC prediction models loaded")

        # ── Gene input ────────────────────────────────────────────────────────
        st.subheader("Select resistance genes present in the strain")

        # Collect all genes from MIC models
        all_mic_genes = set()
        for bundle in mic_models.values():
            for feat in bundle["features"]:
                all_mic_genes.add(feat.replace("gene__", "", 1))
        all_mic_genes = sorted(all_mic_genes)

        search_mic = st.text_input("Search genes:", placeholder="e.g. bla, gyr, tet...",
                                   key="mic_gene_search")
        filtered_mic = [g for g in all_mic_genes
                        if not search_mic or search_mic.lower() in g.lower()]

        if "mic_selected_genes" not in st.session_state:
            st.session_state["mic_selected_genes"] = set()

        col_search, col_btns = st.columns([3, 2])
        with col_btns:
            c1, c2 = st.columns(2)
            if c1.button(f"✅ Select all {len(filtered_mic)}", use_container_width=True):
                st.session_state["mic_selected_genes"].update(filtered_mic)
                st.rerun()
            if c2.button("🗑 Clear all", use_container_width=True):
                st.session_state["mic_selected_genes"] = set()
                st.rerun()

        st.caption(f"**{len(filtered_mic)}** matching genes | "
                   f"**{len(st.session_state['mic_selected_genes'])}** selected")

        if filtered_mic:
            gcols = st.columns(4)
            for i, gene in enumerate(filtered_mic[:40]):
                checked = gene in st.session_state["mic_selected_genes"]
                new_val = gcols[i % 4].checkbox(
                    gene[:40], value=checked, key=f"mic_gene_{gene}"
                )
                if new_val:
                    st.session_state["mic_selected_genes"].add(gene)
                else:
                    st.session_state["mic_selected_genes"].discard(gene)
            if len(filtered_mic) > 40:
                st.caption(f"Showing first 40 of {len(filtered_mic)}. Refine search to see more.")

        # ── Predict ───────────────────────────────────────────────────────────
        if st.button("🔬 Predict MIC Categories", type="primary", use_container_width=True):
            gene_presence = {g: 1.0 for g in st.session_state["mic_selected_genes"]}
            n_genes_selected = len(gene_presence)

            st.subheader("Predicted MIC categories")
            if n_genes_selected == 0:
                st.info("No genes selected — showing baseline predictions (no resistance genes).")

            results = []
            for ab, bundle in mic_models.items():
                model    = bundle["model"]
                features = bundle["features"]
                x = np.zeros(len(features))
                for i, feat in enumerate(features):
                    gene_name = feat.replace("gene__", "", 1)
                    x[i] = gene_presence.get(gene_name, 0)
                X_input = pd.DataFrame([x], columns=features)
                probs = model.predict_proba(X_input)[0]
                pred_cat_idx = int(np.argmax(probs))
                pred_cat = bundle["categories"][pred_cat_idx]
                bp_s = bundle["bp_s"]
                bp_r = bundle["bp_r"]
                results.append({
                    "antibiotic": ab,
                    "predicted_category": pred_cat,
                    "prob_low": round(probs[0] * 100, 1),
                    "prob_inter": round(probs[1] * 100, 1),
                    "prob_high": round(probs[2] * 100, 1),
                    "bp_s": bp_s,
                    "bp_r": bp_r,
                    "model_auc": bundle.get("mean_auc", 0),
                })

            df_res = pd.DataFrame(results).sort_values("prob_high", ascending=False)

            for _, r in df_res.iterrows():
                cat = r["predicted_category"]
                bg  = MIC_CAT_BG[cat]
                tc  = MIC_CAT_COLOR[cat]
                desc = MIC_CAT_DESC[cat]

                bar_html = (
                    f"<div style='background:#EEF2FF; border-radius:4px; height:8px; margin-top:4px;'>"
                    f"<div style='width:{r['prob_high']:.0f}%; background:#EF4444; height:8px; "
                    f"border-radius:4px; display:inline-block;'></div>"
                    f"</div>"
                )
                st.markdown(f"""
<div style='background:{bg}; border:1px solid {tc}40; border-left:4px solid {tc};
     border-radius:8px; padding:0.6rem 1rem; margin-bottom:6px; display:flex;
     justify-content:space-between; align-items:flex-start; flex-wrap:wrap;'>
  <div style='flex:1; min-width:200px;'>
    <b style='color:#1E293B;'>{r["antibiotic"]}</b>
    <span style='color:#64748B; font-size:0.78rem; margin-left:8px;'>
      EUCAST S≤{r["bp_s"]} | R>{r["bp_r"]} mg/L
    </span><br>
    <small style='color:{tc};'>{desc}</small>
  </div>
  <div style='text-align:right; min-width:220px;'>
    <span style='background:{tc}; color:white; padding:2px 10px; border-radius:4px;
      font-weight:700; font-size:0.9rem;'>{cat}</span><br>
    <small style='color:#64748B;'>
      Low {r["prob_low"]:.0f}% · Inter {r["prob_inter"]:.0f}% · High {r["prob_high"]:.0f}%
    </small>
    {bar_html}
  </div>
</div>""", unsafe_allow_html=True)

st.divider()

# ── Section 4: EUCAST breakpoint reference ────────────────────────────────────
st.header("4. EUCAST 2024 breakpoints — reference table")

st.markdown("""
Clinical breakpoints define at which MIC value an organism is classified as
Susceptible (S), Intermediate (I), or Resistant (R) to a given antibiotic.
They are species-specific and updated annually.
""")

bp_rows = []
for ab, (bp_s, bp_r) in sorted(EUCAST_BP.items()):
    bp_rows.append({
        "Antibiotic": ab,
        "S ≤ (mg/L)": bp_s,
        "R > (mg/L)": bp_r,
        "Intermediate zone": f"{bp_s}–{bp_r} mg/L" if bp_s < bp_r else "None",
        "Drug class": {
            "meropenem": "Carbapenem", "imipenem": "Carbapenem",
            "ciprofloxacin": "Fluoroquinolone", "levofloxacin": "Fluoroquinolone",
            "gentamicin": "Aminoglycoside", "amikacin": "Aminoglycoside",
            "piperacillin/tazobactam": "Beta-lactam/BLI",
            "trimethoprim/sulfamethoxazole": "Folate inhibitor",
            "ceftazidime": "Cephalosporin (3rd gen)",
            "ceftriaxone": "Cephalosporin (3rd gen)",
            "tetracycline": "Tetracycline", "cefepime": "Cephalosporin (4th gen)",
        }.get(ab, "Other"),
    })

df_bp = pd.DataFrame(bp_rows)

def _color_s_breakpoint(val):
    try:
        v = float(val)
        if v <= 0.5:  return "background-color:#DCFCE7; color:#15803D; font-weight:600"
        elif v <= 4:  return "background-color:#DBEAFE; color:#1E40AF; font-weight:600"
        return ""
    except: return ""

st.dataframe(
    df_bp.style
    .map(_color_s_breakpoint, subset=["S ≤ (mg/L)"])
    .format({"S ≤ (mg/L)": "{:.3g}", "R > (mg/L)": "{:.3g}"}),
    use_container_width=True, hide_index=True
)
st.caption("Breakpoints from EUCAST 2024 clinical breakpoint tables for *K. pneumoniae*. "
           "Lower susceptible breakpoints (green) indicate intrinsically more potent drugs.")

st.divider()

# ── Section 5: Why MIC prediction is hard ────────────────────────────────────
st.header("5. Challenges in MIC prediction")

st.markdown("""
| Challenge | Explanation |
|---|---|
| **Continuous target** | MIC is a continuous value but measured in 2-fold dilution steps — creating an ordinal problem |
| **Heterogeneity of measurement** | Different labs use different methods (broth microdilution, E-test, disk diffusion) with different accuracy |
| **Population heteroresistance** | Some strains have subpopulations at different MIC levels — a single value may be misleading |
| **Missing genes** | Some resistance mechanisms aren't gene-based (efflux pump overexpression, porin loss) and won't appear in gene matrices |
| **Breakpoint changes** | EUCAST and CLSI update breakpoints annually — a strain that was "Susceptible" may become "Resistant" without changing its MIC |
| **Limited training data** | MIC data requires careful lab work — far fewer records than binary R/S phenotypes |

**Current model approach:** 3-class ordinal (Low / Intermediate / High) using gene presence/absence features.
Next iteration: regression against log₂(MIC) using both gene and k-mer features.
""")
