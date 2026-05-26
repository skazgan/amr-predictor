"""
Page 20 — Outbreak Detection: cluster K. pneumoniae MLST lineages by resistance similarity
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

st.title("🔬 Outbreak Detection")
st.markdown("*Cluster K. pneumoniae strains by resistance-profile similarity to flag potential hospital outbreaks.*")
st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
mlst_path = ART_DIR / "mlst_analysis.json"
if not mlst_path.exists():
    st.error("Run `python src/mlst_analysis.py` to generate MLST data.")
    st.stop()

mlst_data = json.loads(mlst_path.read_text())
st_profiles = pd.DataFrame(mlst_data["st_profiles"])
antibiotics = mlst_data.get("antibiotics", [])
short_names = mlst_data.get("short_names", {})

DRUG_COLS = [f"{short_names.get(ab, ab)}_pct_R" for ab in antibiotics
             if f"{short_names.get(ab, ab)}_pct_R" in st_profiles.columns]

st_profiles = st_profiles[st_profiles["n_genomes"] >= 5].copy()
st_profiles["st_label"] = "ST" + st_profiles["st"].astype(str)

# ── Section 1: How outbreak detection works ───────────────────────────────────
st.header("1. How outbreak detection works in genomic epidemiology")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
A hospital outbreak occurs when **multiple patients are infected by the same bacterial clone**
— a genetically related strain that has spread between patients, often via hands, equipment,
or shared environment.

**Genomic surveillance detects outbreaks by:**

1. **MLST typing** — assigning each strain a Sequence Type (ST) based on 7 housekeeping genes.
   Strains with the same ST are closely related.

2. **Resistance profile similarity** — outbreak strains typically share the same
   resistance genes (acquired together on a plasmid), giving identical or near-identical
   antibiogram patterns.

3. **Spatio-temporal clustering** — outbreak strains appear in the same ward/hospital
   within a short time window.

**This page** clusters K. pneumoniae MLST types by their resistance profiles
using **PCA (Principal Component Analysis)** — STs that cluster together in
resistance space are candidates for further genomic investigation.
""")

with col2:
    st.info("""
**Key outbreak lineages to watch:**

🔴 **ST11** — Dominant MDR lineage in Asia; carries KPC carbapenemase

🔴 **ST258** — Most common CRKP in US and Europe; KPC-producing

🟠 **ST14** — Emerging ESBL lineage globally; CTX-M producers

🟠 **ST15** — Pan-European spread; often CTX-M-15

🟡 **ST307** — High-risk clone; increasing globally

Strains of these STs with very high resistance rates should trigger
immediate contact tracing and enhanced infection control.
""")

st.divider()

# ── Section 2: PCA clustering ─────────────────────────────────────────────────
st.header("2. MLST lineage clustering by resistance profile")

st.markdown("""
Each point is one MLST Sequence Type (ST). Position reflects **resistance pattern similarity**
across all modelled antibiotics. ST types that appear close together have similar antibiograms —
consistent with sharing the same resistance plasmid or being from the same outbreak cluster.
""")

# PCA computation using only numpy (no sklearn needed in this scope)
feature_matrix = st_profiles[DRUG_COLS].fillna(st_profiles[DRUG_COLS].mean())

# Standardise
X = feature_matrix.values.astype(float)
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

# Covariance matrix → eigendecomposition (manual PCA)
cov = np.cov(X_scaled.T)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, order]
pcs = X_scaled @ eigvecs[:, :2]
var_explained = eigvals[order][:2] / eigvals.sum() * 100

st_profiles["PC1"] = pcs[:, 0]
st_profiles["PC2"] = pcs[:, 1]

# MDR cluster detection: flag STs with >50% resistance across most drugs
mean_resist = feature_matrix.mean(axis=1)
st_profiles["mean_resist"] = mean_resist.values

# Colour by MDR %
color_scale = [
    [0.00, "#10B981"],
    [0.30, "#FCD34D"],
    [0.60, "#F97316"],
    [1.00, "#EF4444"],
]

# Filter controls
col_filter1, col_filter2, col_filter3 = st.columns(3)
with col_filter1:
    min_genomes = st.slider("Minimum genomes per ST", 5, 100, 20)
with col_filter2:
    mdr_threshold = st.slider("Highlight MDR above (% mean resistance):", 0, 100, 50)
with col_filter3:
    show_labels = st.checkbox("Show ST labels", value=True)

df_plot = st_profiles[st_profiles["n_genomes"] >= min_genomes].copy()

fig_pca = go.Figure()

# Non-MDR background points
df_non_mdr = df_plot[df_plot["mean_resist"] < mdr_threshold]
if not df_non_mdr.empty:
    fig_pca.add_trace(go.Scatter(
        x=df_non_mdr["PC1"], y=df_non_mdr["PC2"],
        mode="markers" + ("+text" if show_labels else ""),
        text=df_non_mdr["st_label"] if show_labels else None,
        textposition="top center",
        textfont=dict(size=9, color="#94A3B8"),
        marker=dict(
            size=np.clip(np.sqrt(df_non_mdr["n_genomes"]) * 2, 6, 30),
            color=df_non_mdr["mean_resist"],
            colorscale=color_scale,
            cmin=0, cmax=100,
            opacity=0.6,
            line=dict(width=1, color="#CBD5E1"),
        ),
        customdata=df_non_mdr[["st", "n_genomes", "pct_mdr", "mean_resist"]].values,
        hovertemplate=(
            "<b>ST%{customdata[0]}</b><br>"
            "Genomes: %{customdata[1]:,}<br>"
            "MDR rate: %{customdata[2]:.1f}%<br>"
            "Mean resistance: %{customdata[3]:.1f}%<extra></extra>"
        ),
        name="Standard STs",
        showlegend=True,
    ))

# High-MDR / outbreak candidate points
df_mdr = df_plot[df_plot["mean_resist"] >= mdr_threshold]
if not df_mdr.empty:
    fig_pca.add_trace(go.Scatter(
        x=df_mdr["PC1"], y=df_mdr["PC2"],
        mode="markers" + ("+text" if show_labels else ""),
        text=df_mdr["st_label"] if show_labels else None,
        textposition="top center",
        textfont=dict(size=10, color="#991B1B", family="Arial Black"),
        marker=dict(
            size=np.clip(np.sqrt(df_mdr["n_genomes"]) * 2.5, 10, 40),
            color="#EF4444",
            opacity=0.9,
            line=dict(width=2, color="#991B1B"),
            symbol="circle",
        ),
        customdata=df_mdr[["st", "n_genomes", "pct_mdr", "mean_resist", "notable"]].values,
        hovertemplate=(
            "<b>ST%{customdata[0]}</b><br>"
            "Genomes: %{customdata[1]:,}<br>"
            "MDR rate: %{customdata[2]:.1f}%<br>"
            "Mean resistance: %{customdata[3]:.1f}%<br>"
            "<i>%{customdata[4]}</i><extra></extra>"
        ),
        name=f"⚠️ Outbreak candidates (mean R ≥ {mdr_threshold}%)",
        showlegend=True,
    ))

fig_pca.update_layout(
    xaxis_title=f"PC1 ({var_explained[0]:.1f}% variance explained)",
    yaxis_title=f"PC2 ({var_explained[1]:.1f}% variance explained)",
    height=520, margin=dict(t=20, b=30),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font_color="#1E293B",
    xaxis_gridcolor="#E2E8F0", yaxis_gridcolor="#E2E8F0",
    legend=dict(orientation="h", yanchor="bottom", y=1.01),
)
st.plotly_chart(fig_pca, use_container_width=True)
st.caption(f"Point size ∝ √(number of genomes). Colour: green = low resistance, red = high resistance. "
           f"Showing {len(df_plot)} STs with ≥{min_genomes} genomes.")

st.divider()

# ── Section 3: High-risk ST table ─────────────────────────────────────────────
st.header("3. High-risk lineage table")

st.markdown("MLST Sequence Types ranked by mean resistance rate across all antibiotics.")

df_rank = df_plot.sort_values("mean_resist", ascending=False).head(20)
display_cols = ["st_label", "n_genomes", "pct_mdr", "mean_resist", "notable"] + DRUG_COLS[:6]
col_rename = {"st_label": "ST", "n_genomes": "Genomes", "pct_mdr": "MDR %",
              "mean_resist": "Mean R %", "notable": "Known lineage"}
col_rename.update({c: c.replace("_pct_R", "") for c in DRUG_COLS[:6]})

df_rank_disp = df_rank[[c for c in display_cols if c in df_rank.columns]].rename(columns=col_rename)

def _color_resist(val):
    try:
        v = float(val)
        if v >= 70: return "background-color:#FEE2E2; color:#991B1B; font-weight:600"
        elif v >= 40: return "background-color:#FEF9C3; color:#713F12; font-weight:600"
        elif v >= 20: return "background-color:#DCFCE7; color:#15803D"
        return ""
    except Exception:
        return ""

num_cols = [c for c in df_rank_disp.columns if c not in ["ST", "Known lineage"]]
st.dataframe(
    df_rank_disp.style
    .map(_color_resist, subset=[c for c in num_cols if "%" in c or c in ["MDR %", "Mean R %"]
                                 or any(d in c for d in ["Cipro","Mero","Gent","Tet","TMP","Cef","Amik","Imi","Pip","Levo"])])
    .format({c: "{:.1f}%" for c in num_cols}),
    use_container_width=True, hide_index=True,
)

st.divider()

# ── Section 4: Resistance profile of a single ST ──────────────────────────────
st.header("4. Deep-dive: resistance profile of one ST")

available_sts = sorted(st_profiles["st"].tolist(), key=lambda x: int(x) if str(x).isdigit() else 9999)
st_sel = st.selectbox("Select Sequence Type:", [f"ST{s}" for s in available_sts], key="st_deep")
st_num = st_sel.replace("ST", "")

row = st_profiles[st_profiles["st"] == st_num]
if row.empty:
    st.info("ST not found in dataset.")
else:
    row = row.iloc[0]
    notable = row.get("notable", "")
    n = int(row["n_genomes"])
    mdr_pct = float(row["pct_mdr"])
    mean_r  = float(row["mean_resist"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Genomes", f"{n:,}")
    col2.metric("MDR rate", f"{mdr_pct:.1f}%")
    col3.metric("Mean resistance", f"{mean_r:.1f}%")
    col4.metric("Risk level",
                "🔴 HIGH" if mean_r >= 60 else "🟠 MOD" if mean_r >= 30 else "🟢 LOW")

    if notable:
        st.info(f"📌 {notable}")

    # Resistance bars per drug
    drug_labels, drug_values, drug_colors = [], [], []
    for col in DRUG_COLS:
        drug_short = col.replace("_pct_R", "")
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            drug_labels.append(drug_short)
            drug_values.append(float(val))
            drug_colors.append(
                "#EF4444" if val >= 70 else
                "#F97316" if val >= 40 else
                "#FCD34D" if val >= 20 else
                "#10B981"
            )

    if drug_labels:
        fig_st = go.Figure(go.Bar(
            x=drug_labels, y=drug_values,
            marker_color=drug_colors,
            text=[f"{v:.1f}%" for v in drug_values],
            textposition="outside",
        ))
        fig_st.add_hline(y=70, line_dash="dash", line_color="#EF4444",
                         annotation_text="High resistance (70%)",
                         annotation_font_color="#EF4444")
        fig_st.update_layout(
            title=f"{st_sel} — resistance rate per antibiotic",
            yaxis=dict(title="% Resistant", range=[0, 115]),
            height=340, margin=dict(t=50, b=10, r=120),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
        )
        st.plotly_chart(fig_st, use_container_width=True)

st.divider()

# ── Section 5: Outbreak investigation guide ───────────────────────────────────
st.header("5. How to investigate a suspected outbreak")

st.markdown("""
| Step | Action | Tool |
|---|---|---|
| **1 — Detect** | Flag ≥2 patients with same ST and similar antibiogram within 30 days | MLST + antibiogram clustering (this page) |
| **2 — Confirm** | Whole-genome SNP analysis — outbreak strains differ by <25 SNPs | cgMLST, BEAST, Snippy |
| **3 — Trace** | Identify common exposure: ward, device, healthcare worker, food | Epidemiological investigation |
| **4 — Control** | Contact precautions, enhanced hand hygiene, cohorting, environmental cleaning | Infection control team |
| **5 — Report** | Notify public health authority if ≥3 cases or novel resistance mechanism | National surveillance (CDC, ECDC, PHE) |
| **6 — Document** | Publish or report cluster — contributes to global surveillance | BV-BRC, NCBI SRA, GLASS |

**Key thresholds for outbreak alert:**
- Same ST + same resistance pattern + ≤30 days → **Possible cluster** (investigate)
- Same ST + WGS distance <25 SNPs + ≤14 days + same ward → **Confirmed outbreak**
- Novel carbapenem resistance mechanism (KPC, NDM, OXA-48) → **Notifiable** in most countries
""")
