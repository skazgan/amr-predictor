"""
Page 7 — Co-Resistance Network: novel findings from our analysis
"""
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent.parent.parent
ART_DIR = ROOT / "artifacts"


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
inject_mobile_css()
st.title("🕸️ Co-Resistance Network")
st.markdown("*Novel findings: which antibiotic resistances travel together — and why.*")
st.info("💡 This page presents original analysis from this project, not reproduced from existing literature.")
st.divider()

SHORT = {
    "ciprofloxacin":                "Cipro",
    "meropenem":                    "Mero",
    "gentamicin":                   "Gent",
    "tetracycline":                 "Tet",
    "trimethoprim/sulfamethoxazole":"TMP/SMX",
    "cefepime":                     "Cef",
}

# Load artifacts
phi_data    = json.loads((ART_DIR / "coresistance_matrix.json").read_text())
mdr_data    = json.loads((ART_DIR / "mdr_prevalence.json").read_text())
mdr_genes   = json.loads((ART_DIR / "mdr_genes.json").read_text())
cross_pred  = json.loads((ART_DIR / "cross_prediction.json").read_text())
counts_data = json.loads((ART_DIR / "coresistance_counts.json").read_text())

# ── Section 1: What is co-resistance ─────────────────────────────────────────
st.header("1. What is co-resistance?")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
Co-resistance means a bacterium is resistant to **multiple antibiotics at the same time**.
This happens because resistance genes often travel together on **plasmids** —
small circular DNA molecules that bacteria can share with each other.

**Example:** A hospital plasmid might carry:
- CTX-M gene → resistance to cefepime
- AAC(3) gene → resistance to gentamicin
- *dhps* gene → resistance to TMP/SMX

One horizontal gene transfer event → instant triple resistance.

**Why this matters clinically:**
If a lab test confirms resistance to drug A, knowing which drugs tend to travel with A
can inform treatment decisions *before* full susceptibility testing is complete.
""")
with col2:
    st.metric("Genomes analysed", f"{mdr_data['total_genomes']:,}")
    st.metric("MDR strains (3+ drugs)", f"{mdr_data['mdr_3plus']} ({mdr_data['pct_mdr']}%)")
    st.metric("Pan-resistant strains", mdr_data['pan_resistant'])
    st.metric("Fully susceptible", f"{mdr_data['susceptible_all']} ({mdr_data['susceptible_all']/mdr_data['total_genomes']*100:.0f}%)")

st.divider()

# ── Section 2: Correlation heatmap ───────────────────────────────────────────
st.header("2. Co-resistance correlation matrix (φ coefficient)")

st.markdown("""
The **phi coefficient (φ)** measures how strongly two binary variables are correlated.
φ = +1 means always co-resistant; φ = 0 means completely independent; φ = -1 means
resistance to one predicts susceptibility to the other.
""")

phi_mat   = np.array(phi_data["phi_matrix"])
short_names = phi_data["short_names"]
pval_mat  = np.array(phi_data["pval_matrix"])

# Significance stars overlay
stars = []
for i in range(len(short_names)):
    row = []
    for j in range(len(short_names)):
        if i == j:
            row.append("")
        else:
            p = pval_mat[i][j]
            row.append("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns")
    stars.append(row)

text_vals = [[f"{phi_mat[i][j]:.3f}<br>{stars[i][j]}" if i != j else "1.000"
              for j in range(len(short_names))]
             for i in range(len(short_names))]

fig_heat = go.Figure(go.Heatmap(
    z=phi_mat,
    x=short_names, y=short_names,
    colorscale="RdBu", zmid=0, zmin=-0.6, zmax=0.6,
    text=text_vals,
    texttemplate="%{text}",
    textfont=dict(size=11),
    showscale=True,
    colorbar=dict(title="φ coefficient"),
))
fig_heat.update_layout(
    height=420, margin=dict(t=20, b=10),
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    font_color="#1E293B",
)
st.plotly_chart(fig_heat, use_container_width=True)
st.caption("*** p<0.001  ** p<0.01  * p<0.05  ns = not significant")

# Key findings callouts
col1, col2, col3 = st.columns(3)
with col1:
    st.success("**Strongest link:** Gent × TMP/SMX (φ=+0.544)\nThese travel on the same plasmids in hospital strains.")
with col2:
    st.warning("**Surprising independence:** Mero × Cef (φ≈0.002)\nBoth are beta-lactams, yet resistance mechanisms are entirely different.")
with col3:
    st.info("**Clinical cluster:** Cipro + TMP/SMX + Cef\nClassic community-acquired MDR signature.")

st.divider()

# ── Section 3: MDR breakdown ──────────────────────────────────────────────────
st.header("3. Multi-drug resistance (MDR) prevalence")

col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("""
**MDR definition:** Resistant to 3 or more antibiotic classes.

The distribution shows that **most strains are not MDR** —
resistance to all drugs simultaneously is rare.

However, the strains that are MDR tend to share specific
plasmid-borne gene clusters — which we can identify.
""")

with col2:
    dist = {int(float(k)): v for k, v in mdr_data["distribution"].items()}
    labels = [f"{k} drug{'s' if k != 1 else ''}" for k in sorted(dist.keys())]
    values = [dist[k] for k in sorted(dist.keys())]
    colors = ["#50fa7b" if k == 0 else
              "#8be9fd" if k <= 2 else
              "#ffb86c" if k <= 4 else "#e94560"
              for k in sorted(dist.keys())]
    fig_bar = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=values, textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis_title="Number of genomes",
        height=300, margin=dict(t=20, b=10),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#1E293B", yaxis_gridcolor="#2d2d44",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("🟢 Susceptible  🔵 1–2 drugs  🟠 3–4 drugs (MDR)  🔴 5–6 drugs")

st.divider()

# ── Section 4: MDR-enriched genes ────────────────────────────────────────────
st.header("4. Which genes drive multi-drug resistance?")

st.markdown("""
We compared gene presence rates in **MDR strains (3+ drugs)** vs **non-MDR strains**.
Genes appearing significantly more in MDR strains are likely on the same plasmids
that spread resistance clusters.
""")

if mdr_genes:
    df_genes = pd.DataFrame(mdr_genes)
    df_genes["name"] = df_genes["gene"].str[:50]
    df_genes = df_genes.sort_values("enrichment", ascending=True)

    pos = df_genes[df_genes["enrichment"] > 0].tail(12)
    neg = df_genes[df_genes["enrichment"] < 0].head(5)
    df_plot = pd.concat([neg, pos])

    colors_g = ["#50fa7b" if v < 0 else "#e94560" for v in df_plot["enrichment"]]
    fig_genes = go.Figure(go.Bar(
        x=df_plot["enrichment"], y=df_plot["name"],
        orientation="h", marker_color=colors_g,
        text=[f"{v:+.3f}" for v in df_plot["enrichment"]],
        textposition="outside",
    ))
    fig_genes.add_vline(x=0, line_color="#1E293B", line_width=1)
    fig_genes.update_layout(
        title="Gene enrichment: MDR vs non-MDR strains",
        xaxis_title="Enrichment (rate in MDR − rate in non-MDR)",
        height=440, margin=dict(t=40, b=10, r=80),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#1E293B", xaxis_gridcolor="#2d2d44",
    )
    st.plotly_chart(fig_genes, use_container_width=True)
    st.caption("🔴 More common in MDR strains (likely on resistance plasmids) &nbsp; 🟢 Less common in MDR strains")

    st.markdown("""
**What the top genes tell us:**

| Gene | Why it matters |
|---|---|
| **CTX-M family** (ESBL) | Extended-spectrum beta-lactamase — breaks down cefepime and other cephalosporins |
| **Sulfonamide resistance** | *dhps* mutation — blocks folate synthesis, confers TMP/SMX resistance |
| **AAC(3) acetyltransferase** | Modifies gentamicin chemically, making it ineffective |
| **MdtABC-TolC** (depleted) | General efflux pump — less important when specific gene-based resistance is present |

These three genes (CTX-M + sulfonamide + AAC3) together explain the **Gent × TMP/SMX × Cef**
co-resistance cluster seen in the correlation matrix.
""")

st.divider()

# ── Section 5: Cross-antibiotic prediction ───────────────────────────────────
st.header("5. Can knowing one resistance predict another?")

st.markdown("""
We trained a model using **only the resistance labels** of the other 5 antibiotics
as features — no genome sequence at all.
If the model can predict the 6th antibiotic better than chance,
it means antibiotic resistances are informationally linked.
""")

if cross_pred:
    df_cp = pd.DataFrame(cross_pred).sort_values("label_only_auc", ascending=False)
    df_cp["short"] = df_cp["antibiotic"].map(SHORT)

    fig_cp = go.Figure()
    fig_cp.add_trace(go.Bar(
        name="Baseline (majority class)",
        x=df_cp["short"], y=df_cp["baseline_auc"],
        marker_color="#2d2d44",
    ))
    fig_cp.add_trace(go.Bar(
        name="Using other antibiotic labels",
        x=df_cp["short"], y=df_cp["label_only_auc"],
        marker_color="#e94560",
        text=[f"+{b:.3f}" for b in df_cp["boost"]],
        textposition="outside",
    ))
    fig_cp.add_hline(y=0.5, line_dash="dot", line_color="#64748B",
                     annotation_text="Random (0.50)")
    fig_cp.update_layout(
        barmode="overlay",
        yaxis=dict(title="ROC-AUC", range=[0.4, 0.85]),
        height=340, margin=dict(t=20, b=10),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#1E293B", yaxis_gridcolor="#2d2d44",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cp, use_container_width=True)

    st.markdown("""
**Key insight:** TMP/SMX (AUC=0.751) and Cipro (AUC=0.739) can be predicted
from other antibiotic labels alone — with no genome sequence at all.
Meropenem (AUC=0.578) and Tetracycline (AUC=0.576) are much more independent.

**Clinical implication:**
> If a patient's *K. pneumoniae* is confirmed resistant to gentamicin,
> empiric treatment should avoid TMP/SMX and cefepime as well —
> even before susceptibility testing for those drugs is complete.
""")

st.divider()

# ── Section 6: Summary of novel findings ─────────────────────────────────────
st.header("6. Summary of novel findings")

st.markdown("""
| Finding | Result | Clinical relevance |
|---|---|---|
| Strongest co-resistance pair | **Gent × TMP/SMX (φ=0.544)** | Confirmed by plasmid biology |
| Most independent pair | **Mero × Cef (φ≈0.002)** | Carbapenem resistance evolves separately |
| MDR prevalence | **4.9% of strains** | Relatively manageable — not a pandemic yet |
| Top MDR-driving gene | **CTX-M family ESBL** | Priority surveillance target |
| Best cross-predictable antibiotic | **TMP/SMX from others (AUC=0.751)** | Actionable for empiric treatment |
| Least cross-predictable | **Meropenem (AUC=0.578)** | Must always test independently |

These findings are consistent with known plasmid biology in *K. pneumoniae* but
**quantified at scale** across 5,737 genomes — providing statistical confirmation
of clinical observations and concrete φ coefficients for each drug pair.
""")
