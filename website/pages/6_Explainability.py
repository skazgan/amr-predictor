"""
Page 6 — Explainability: feature importances and SHAP per antibiotic
"""
import json
import sys
import pickle
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent.parent.parent
ART_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
inject_mobile_css()
st.title("💡 Explainability")
st.markdown("*Which genes and k-mers actually drive the predictions?*")
st.divider()

ANTIBIOTICS = [
    "ciprofloxacin", "meropenem", "gentamicin",
    "tetracycline", "trimethoprim/sulfamethoxazole", "cefepime",
]

# ── Section 1: Why explainability matters ─────────────────────────────────────
st.header("1. Why explainability matters in medicine")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
A model that says "this strain is 84% likely resistant" is only useful if a clinician
can **trust and verify** the reasoning.

**What we can explain:**
- Which resistance genes were present in this genome?
- Which k-mer patterns contributed most?
- How does the model's decision change if a gene is removed?

This matters because:
- A new resistance mechanism not in CARD will lower confidence → correctly flagged as Uncertain
- If *blaKPC* is detected, the meropenem model should weight it heavily → we can verify it does
""")
with col2:
    st.info("""
**The two explainability tools we use:**

🔹 **Feature Importance** (global)
Average contribution of each feature across all predictions.
Shows what the model generally relies on.

🔹 **SHAP values** (local)
Contribution of each feature to *one specific prediction*.
Shows why *this genome* was classified this way.
""")

st.divider()

# ── Section 2: Feature importances per antibiotic ────────────────────────────
st.header("2. Feature importances — what does each model rely on?")

ab_choice = st.selectbox("Select antibiotic:", ANTIBIOTICS, key="fi_ab")
safe_ab   = ab_choice.replace("/","_").replace(" ","_")
fi_path   = ART_DIR / f"fi_{safe_ab}.json"

if fi_path.exists():
    fi = json.loads(fi_path.read_text())
    if fi:
        df_fi = pd.DataFrame(fi)
        df_fi["type"] = df_fi["feature"].apply(
            lambda x: "Resistance gene" if x.startswith("gene__") else "k-mer"
        )
        df_fi["name"] = df_fi["feature"].str.replace("gene__", "", regex=False)
        df_fi = df_fi.sort_values("importance", ascending=True).tail(20)

        col1, col2 = st.columns([3, 2])
        with col1:
            colors = ["#50fa7b" if t == "Resistance gene" else "#8be9fd"
                      for t in df_fi["type"]]
            fig = go.Figure(go.Bar(
                x=df_fi["importance"], y=df_fi["name"],
                orientation="h", marker_color=colors,
                text=[f"{v:.4f}" for v in df_fi["importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Top 20 features — {ab_choice}",
                xaxis_title="Mean importance (across calibration folds)",
                height=520, margin=dict(t=40, b=10, r=80),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font_color="#1E293B", xaxis_gridcolor="#E2E8F0",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("🟢 Resistance gene &nbsp;&nbsp; 🔵 k-mer")

        with col2:
            n_genes = (df_fi["type"] == "Resistance gene").sum()
            n_kmers = (df_fi["type"] == "k-mer").sum()
            st.metric("Resistance genes in top 20", n_genes)
            st.metric("k-mers in top 20", n_kmers)

            top_gene = df_fi[df_fi["type"]=="Resistance gene"].iloc[-1] if n_genes else None
            if top_gene is not None:
                st.markdown(f"""
**Most important resistance gene:**

> **{top_gene['name']}**

This gene directly confers resistance via a known biological mechanism
catalogued in the CARD database.
""")
            # Pie of gene vs kmer in top 20
            fig2 = go.Figure(go.Pie(
                labels=["Resistance genes", "k-mers"],
                values=[n_genes, n_kmers],
                hole=0.5,
                marker_colors=["#50fa7b", "#8be9fd"],
            ))
            fig2.update_layout(
                title="Feature type split (top 20)",
                height=220, margin=dict(t=40, b=10),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font_color="#1E293B", showlegend=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Section 3: SHAP for a genome ─────────────────────────────────────────────
st.header("3. SHAP — explain one specific prediction")

st.markdown("""
**SHAP (SHapley Additive exPlanations)** computes, for a single genome,
how much each feature *pushed* the prediction towards Resistant or Susceptible.

- Positive SHAP → feature increased P(Resistant)
- Negative SHAP → feature decreased P(Resistant)
""")

shap_genome_id = st.text_input(
    "Genome ID for SHAP analysis:",
    value="573.12783",
    help="Must be a genome in the pre-computed feature matrices.",
)
shap_ab = st.selectbox("Antibiotic:", ANTIBIOTICS, key="shap_ab")
run_shap = st.button("⚡ Compute SHAP values", type="primary")

if run_shap and shap_genome_id:
    safe_ab2 = shap_ab.replace("/","_").replace(" ","_")
    model_path = MODEL_DIR / f"{safe_ab2}.pkl"

    if not model_path.exists():
        st.error("Model not found. Run generate_artifacts.py first.")
    else:
        with st.spinner("Computing SHAP values (this takes ~20 seconds) …"):
            try:
                import shap

                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)

                features = bundle["features"]
                model_pipeline = bundle["model"]

                kmer_cols = [c for c in features if not c.startswith("gene__")]
                gene_cols  = [c for c in features if c.startswith("gene__")]
                gene_bare  = [c.replace("gene__","") for c in gene_cols]

                X_kmer = pd.read_csv(PROC_DIR / "X.csv", index_col=0)
                X_kmer.index = X_kmer.index.astype(str)
                X_gene_raw = pd.read_csv(PROC_DIR / "gene_matrix.csv", index_col=0)
                X_gene_raw.index = X_gene_raw.index.astype(str)
                X_gene_raw = X_gene_raw.drop(columns=["__label__"], errors="ignore")

                gid = shap_genome_id.strip()

                if gid not in X_kmer.index:
                    st.warning(f"Genome {gid} not in k-mer matrix.")
                    st.stop()

                kmer_row = X_kmer.loc[gid, kmer_cols] if kmer_cols else pd.Series(dtype=float)
                gene_row = (X_gene_raw.loc[gid, [c for c in gene_bare if c in X_gene_raw.columns]]
                            .reindex(gene_bare, fill_value=0)
                            .rename(lambda x: "gene__"+x)
                            if gid in X_gene_raw.index else
                            pd.Series(0, index=gene_cols))

                x_single = pd.concat([kmer_row, gene_row]).to_frame().T

                # Transform through select + scale steps of the pipeline
                selector = model_pipeline.named_steps["select"]
                scaler   = model_pipeline.named_steps["scaler"]
                calibrated = model_pipeline.named_steps["clf"]

                x_sel    = selector.transform(x_single)
                x_scaled = scaler.transform(x_sel)

                sel_feature_names = [features[i] for i in selector.get_support(indices=True)]

                # Use first base estimator from calibration
                base_est = calibrated.calibrated_classifiers_[0].estimator
                explainer = shap.TreeExplainer(base_est)
                shap_vals = explainer.shap_values(x_scaled)

                # For binary classification shap_values can be list or array
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

                df_shap = pd.DataFrame({
                    "feature": sel_feature_names,
                    "shap":    sv,
                    "type":    ["Resistance gene" if f.startswith("gene__") else "k-mer"
                                for f in sel_feature_names],
                    "name":    [f.replace("gene__","") for f in sel_feature_names],
                }).sort_values("shap", key=abs, ascending=False).head(20)

                df_shap_plot = df_shap.sort_values("shap", ascending=True)
                colors_shap  = ["#e94560" if v > 0 else "#50fa7b"
                                for v in df_shap_plot["shap"]]

                fig_shap = go.Figure(go.Bar(
                    x=df_shap_plot["shap"], y=df_shap_plot["name"],
                    orientation="h", marker_color=colors_shap,
                ))
                fig_shap.add_vline(x=0, line_color="#1E293B", line_width=1)
                fig_shap.update_layout(
                    title=f"SHAP values — genome {gid} — {shap_ab}",
                    xaxis_title="SHAP value (+ = pushes toward Resistant)",
                    height=520, margin=dict(t=40, b=10),
                    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                    font_color="#1E293B", xaxis_gridcolor="#E2E8F0",
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                pred_prob = bundle["model"].predict_proba(x_single)[0][1]
                verdict   = "Resistant" if pred_prob > 0.5 else "Susceptible"
                st.markdown(f"""
**Prediction for genome {gid} — {shap_ab}:**
- P(Resistant) = **{pred_prob*100:.1f}%** → **{verdict}**
- 🔴 Red bars pushed the model toward **Resistant**
- 🟢 Green bars pushed the model toward **Susceptible**
""")

            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                st.exception(e)

st.divider()

# ── Section 4: All-antibiotic feature type comparison ────────────────────────
st.header("4. Gene vs k-mer contribution — across all antibiotics")

rows = []
for ab in ANTIBIOTICS:
    safe = ab.replace("/","_").replace(" ","_")
    p = ART_DIR / f"fi_{safe}.json"
    if not p.exists():
        continue
    fi = json.loads(p.read_text())
    if not fi:
        continue
    df_fi = pd.DataFrame(fi).head(20)
    n_g = df_fi["feature"].str.startswith("gene__").sum()
    n_k = (~df_fi["feature"].str.startswith("gene__")).sum()
    rows.append({"antibiotic": ab, "genes": int(n_g), "kmers": int(n_k)})

if rows:
    df_comp = pd.DataFrame(rows)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Resistance genes", x=df_comp["antibiotic"],
                          y=df_comp["genes"], marker_color="#50fa7b"))
    fig3.add_trace(go.Bar(name="k-mers", x=df_comp["antibiotic"],
                          y=df_comp["kmers"], marker_color="#8be9fd"))
    fig3.update_layout(
        barmode="stack", yaxis_title="Count in top 20 features",
        height=320, margin=dict(t=20, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    Antibiotics where resistance is dominated by known genes (meropenem — *blaKPC*, *blaOXA*)
    have mostly green bars. Those where resistance is more diffuse or involves many
    small-effect mutations show more blue (k-mers filling in the gaps).
    """)
