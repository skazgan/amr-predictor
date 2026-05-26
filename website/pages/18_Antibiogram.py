"""
Page 18 — Antibiogram Heatmap: resistance rates across all 4 organisms × all antibiotics
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

st.title("🧫 Antibiogram Heatmap")
st.markdown("*All 4 organisms × all antibiotics — resistance rates at a glance, computed from 160,000+ genomes.*")
st.divider()

# ── Load & combine data ───────────────────────────────────────────────────────
stats_kp = json.loads((ART_DIR / "dataset_stats.json").read_text())
multi_sum = json.loads((ART_DIR / "multi_org_summary.json").read_text()) if (ART_DIR / "multi_org_summary.json").exists() else []

ORG_DISPLAY = {
    "escherichia_coli":        "E. coli",
    "staphylococcus_aureus":   "S. aureus",
    "acinetobacter_baumannii": "A. baumannii",
}
ORG_COLOR = {
    "K. pneumoniae": "#EF4444",
    "E. coli":       "#F97316",
    "S. aureus":     "#D97706",
    "A. baumannii":  "#3B82F6",
}
ORG_ORDER = ["K. pneumoniae", "E. coli", "S. aureus", "A. baumannii"]

AB_PRETTY = {
    "ciprofloxacin":                "Ciprofloxacin",
    "meropenem":                    "Meropenem",
    "gentamicin":                   "Gentamicin",
    "tetracycline":                 "Tetracycline",
    "trimethoprim/sulfamethoxazole":"TMP/SMX",
    "cefepime":                     "Cefepime",
    "amikacin":                     "Amikacin",
    "imipenem":                     "Imipenem",
    "piperacillin/tazobactam":      "Pip/Taz",
    "levofloxacin":                 "Levofloxacin",
    "ampicillin":                   "Ampicillin",
    "ceftriaxone":                  "Ceftriaxone",
    "oxacillin":                    "Oxacillin",
    "vancomycin":                   "Vancomycin",
    "clindamycin":                  "Clindamycin",
    "erythromycin":                 "Erythromycin",
    "colistin":                     "Colistin",
}
AB_CLASS = {
    "ciprofloxacin":                "Fluoroquinolone",
    "meropenem":                    "Carbapenem",
    "gentamicin":                   "Aminoglycoside",
    "tetracycline":                 "Tetracycline",
    "trimethoprim/sulfamethoxazole":"Folate inhibitor",
    "cefepime":                     "Cephalosporin (4th gen)",
    "amikacin":                     "Aminoglycoside",
    "imipenem":                     "Carbapenem",
    "piperacillin/tazobactam":      "Beta-lactam/BLI",
    "levofloxacin":                 "Fluoroquinolone",
    "ampicillin":                   "Penicillin",
    "ceftriaxone":                  "Cephalosporin (3rd gen)",
    "oxacillin":                    "Penicillinase-resistant penicillin",
    "vancomycin":                   "Glycopeptide",
    "clindamycin":                  "Lincosamide",
    "erythromycin":                 "Macrolide",
    "colistin":                     "Polymyxin",
}

rows = []
# K. pneumoniae
for r in stats_kp:
    n_r, n_s = r["n_resistant"], r["n_susceptible"]
    n_total = n_r + n_s
    rows.append({
        "organism":     "K. pneumoniae",
        "antibiotic":   r["antibiotic"],
        "pct_resistant": round(n_r / n_total * 100, 1) if n_total > 0 else None,
        "n_resistant":  n_r,
        "n_total":      n_total,
        "drug_class":   r.get("drug_class", AB_CLASS.get(r["antibiotic"], "Other")),
    })
# Multi-organism
for r in multi_sum:
    org = ORG_DISPLAY.get(r["organism"], r["organism"])
    n_r = r.get("n_resistant", 0)
    n_total = r.get("n_total", 0)
    rows.append({
        "organism":     org,
        "antibiotic":   r["antibiotic"],
        "pct_resistant": round(n_r / n_total * 100, 1) if n_total > 0 else None,
        "n_resistant":  n_r,
        "n_total":      n_total,
        "drug_class":   AB_CLASS.get(r["antibiotic"], "Other"),
    })

df = pd.DataFrame(rows)
df["ab_label"] = df["antibiotic"].map(AB_PRETTY).fillna(df["antibiotic"].str.title())

# ── Section 1: Explanation ────────────────────────────────────────────────────
st.header("1. What is an antibiogram?")
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
An **antibiogram** is a clinical summary showing what percentage of bacterial isolates
are resistant to each antibiotic — organized as a grid.

Hospitals publish annual antibiograms to guide **empiric therapy** — choosing a drug
before culture results are available (which takes 24–72 hours).

**This antibiogram** is computed from **160,000+ whole-genome sequences** across
4 critical pathogens from BV-BRC (2000–2024), representing isolates from 74+ countries.

| Colour | Meaning |
|---|---|
| 🟢 Green | <20% resistant — drug usually works |
| 🟡 Yellow | 20–40% resistant — use with caution |
| 🟠 Orange | 40–70% resistant — often fails |
| 🔴 Red | >70% resistant — likely ineffective empirically |
""")
with col2:
    non_null = df.dropna(subset=["pct_resistant"])
    overall_mean = non_null["pct_resistant"].mean()
    worst = non_null.loc[non_null["pct_resistant"].idxmax()]
    best  = non_null.loc[non_null["pct_resistant"].idxmin()]

    st.metric("Mean resistance rate across all combinations", f"{overall_mean:.1f}%")
    st.markdown(
        f"**Most resistant:** {worst['organism']} — {worst['ab_label']}: "
        f"<span style='color:#EF4444;font-weight:700'>{worst['pct_resistant']:.1f}%</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"**Least resistant:** {best['organism']} — {best['ab_label']}: "
        f"<span style='color:#10B981;font-weight:700'>{best['pct_resistant']:.1f}%</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"**Combinations modelled:** {len(non_null):,}")

st.divider()

# ── Section 2: Main heatmap ───────────────────────────────────────────────────
st.header("2. Full antibiogram — all organisms × all antibiotics")

pivot = df.pivot_table(index="ab_label", columns="organism", values="pct_resistant", aggfunc="mean")
pivot = pivot.reindex(columns=[o for o in ORG_ORDER if o in pivot.columns])
# Sort antibiotics by mean resistance (highest at top)
pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

# Build hover text matrix
pivot_count = df.pivot_table(index="ab_label", columns="organism", values="n_total", aggfunc="sum")

hover_text = []
for ab in pivot.index:
    row_hover = []
    for org in pivot.columns:
        pct = pivot.loc[ab, org]
        n = pivot_count.loc[ab, org] if ab in pivot_count.index and org in pivot_count.columns else 0
        if np.isnan(pct):
            row_hover.append(f"<b>{ab}</b><br>{org}<br>No data")
        else:
            row_hover.append(f"<b>{ab}</b><br>{org}<br>{pct:.1f}% resistant<br>{int(n):,} genomes")
    hover_text.append(row_hover)

text_matrix = [[f"{v:.0f}%" if not np.isnan(v) else "—" for v in row] for row in pivot.values]

fig_heat = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale=[
        [0.00, "#10B981"],
        [0.20, "#34D399"],
        [0.40, "#FCD34D"],
        [0.70, "#F97316"],
        [1.00, "#EF4444"],
    ],
    zmin=0, zmax=100,
    text=text_matrix,
    texttemplate="%{text}",
    textfont=dict(size=12, color="#1E293B"),
    hovertext=hover_text,
    hovertemplate="%{hovertext}<extra></extra>",
    showscale=True,
    colorbar=dict(
        title="% Resistant",
        ticksuffix="%",
        tickvals=[0, 20, 40, 70, 100],
        ticktext=["0% (Excellent)", "20%", "40%", "70%", "100% (Worst)"],
    ),
))
fig_heat.update_layout(
    height=max(380, len(pivot) * 36 + 80),
    margin=dict(t=20, b=10, l=10, r=10),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font_color="#1E293B",
    xaxis=dict(side="top", tickfont=dict(size=13, color="#1E293B")),
    yaxis=dict(tickfont=dict(size=12)),
)
st.plotly_chart(fig_heat, use_container_width=True)
st.caption("Resistance % computed from genome-level susceptibility labels. "
           "Blanks (—) = antibiotic not modelled for that organism.")

st.divider()

# ── Section 3: Organism drill-down ───────────────────────────────────────────
st.header("3. Organism drill-down")

org_sel = st.selectbox("Select organism:", ORG_ORDER, key="org_drill")
df_org = df[df["organism"] == org_sel].dropna(subset=["pct_resistant"]).sort_values("pct_resistant", ascending=False)

if not df_org.empty:
    col1, col2 = st.columns([3, 2])
    with col1:
        bar_colors = [
            "#EF4444" if p >= 70 else
            "#F97316" if p >= 40 else
            "#FCD34D" if p >= 20 else
            "#10B981"
            for p in df_org["pct_resistant"]
        ]
        fig_bar = go.Figure(go.Bar(
            x=df_org["ab_label"],
            y=df_org["pct_resistant"],
            marker_color=bar_colors,
            text=[f"{p:.1f}%" for p in df_org["pct_resistant"]],
            textposition="outside",
            customdata=df_org[["n_resistant", "n_total"]].values,
            hovertemplate="<b>%{x}</b><br>%{y:.1f}% resistant<br>%{customdata[0]:,} / %{customdata[1]:,} genomes<extra></extra>",
        ))
        fig_bar.add_hline(y=70, line_dash="dash", line_color="#EF4444",
                          annotation_text="High resistance (70%)",
                          annotation_font_color="#EF4444")
        fig_bar.add_hline(y=20, line_dash="dash", line_color="#10B981",
                          annotation_text="Low resistance (20%)",
                          annotation_font_color="#10B981")
        fig_bar.update_layout(
            yaxis=dict(title="% Resistant", range=[0, 115]),
            height=360, margin=dict(t=20, b=10, r=140),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown(f"**{org_sel} — summary**")
        n_high    = (df_org["pct_resistant"] >= 70).sum()
        n_mod     = ((df_org["pct_resistant"] >= 20) & (df_org["pct_resistant"] < 70)).sum()
        n_low     = (df_org["pct_resistant"] < 20).sum()
        mean_pct  = df_org["pct_resistant"].mean()
        first_line = df_org[df_org["pct_resistant"] < 30]["ab_label"].tolist()

        st.metric("Mean resistance", f"{mean_pct:.1f}%")
        st.metric("High-resistance drugs (≥70%)", n_high)
        st.metric("Low-resistance drugs (<20%)", n_low)

        if first_line:
            st.markdown(f"""
<div style='background:#F0FDF4; border:1px solid #86EFAC; border-radius:8px;
     padding:0.8rem 1rem; margin-top:0.5rem;'>
<b style='color:#15803D;'>Best empiric options (<30% resistant):</b><br>
{', '.join(first_line)}
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style='background:#FFF5F5; border:1px solid #FCA5A5; border-radius:8px;
     padding:0.8rem 1rem; margin-top:0.5rem;'>
<b style='color:#991B1B;'>⚠️ No drug below 30% resistance threshold.<br>
Consider combination therapy or last-resort agents.</b>
</div>""", unsafe_allow_html=True)

    # Detail table
    df_display = df_org[["ab_label", "drug_class", "pct_resistant", "n_resistant", "n_total"]].copy()
    df_display.columns = ["Antibiotic", "Drug Class", "% Resistant", "N Resistant", "N Total"]

    def _color_pct(val):
        try:
            v = float(val)
            if v >= 70:  return "background-color:#FEE2E2; color:#991B1B; font-weight:600"
            elif v >= 40: return "background-color:#FEF9C3; color:#713F12; font-weight:600"
            elif v >= 20: return "background-color:#FFF7ED; color:#92400E"
            else:         return "background-color:#DCFCE7; color:#15803D; font-weight:600"
        except Exception:
            return ""

    st.dataframe(
        df_display.style
        .map(_color_pct, subset=["% Resistant"])
        .format({"% Resistant": "{:.1f}%", "N Resistant": "{:,}", "N Total": "{:,}"}),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ── Section 4: Drug cross-organism comparison ─────────────────────────────────
st.header("4. How does one drug perform across all four organisms?")

all_drugs = sorted(df["ab_label"].unique().tolist())
drug_sel = st.selectbox("Select antibiotic:", all_drugs, key="drug_across")

df_drug = df[df["ab_label"] == drug_sel].dropna(subset=["pct_resistant"]).sort_values("pct_resistant", ascending=False)
if not df_drug.empty:
    fig_drug = go.Figure(go.Bar(
        x=df_drug["organism"],
        y=df_drug["pct_resistant"],
        marker_color=[ORG_COLOR.get(o, "#6366F1") for o in df_drug["organism"]],
        text=[f"{p:.1f}%" for p in df_drug["pct_resistant"]],
        textposition="outside",
        customdata=df_drug[["n_resistant", "n_total"]].values,
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% resistant<br>%{customdata[0]:,} / %{customdata[1]:,} genomes<extra></extra>",
    ))
    fig_drug.add_hline(y=70, line_dash="dash", line_color="#EF4444",
                       annotation_text="High-resistance threshold (70%)",
                       annotation_font_color="#EF4444")
    fig_drug.update_layout(
        title=f"{drug_sel} — resistance rate by organism",
        yaxis=dict(title="% Resistant", range=[0, 115]),
        height=320, margin=dict(t=50, b=10, r=200),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
    )
    st.plotly_chart(fig_drug, use_container_width=True)

    drug_class = df_drug.iloc[0]["drug_class"]
    st.info(f"**Drug class:** {drug_class}")

st.divider()

# ── Section 5: Drug class comparison ─────────────────────────────────────────
st.header("5. Resistance by drug class — cross-organism overview")

df_class = df.dropna(subset=["pct_resistant"]).copy()
df_class_grouped = (
    df_class.groupby(["organism", "drug_class"])["pct_resistant"]
    .mean().reset_index()
)
df_class_grouped = df_class_grouped[df_class_grouped["organism"].isin(ORG_ORDER)]

fig_class = px.box(
    df_class, x="drug_class", y="pct_resistant", color="organism",
    color_discrete_map=ORG_COLOR,
    category_orders={"organism": ORG_ORDER},
    labels={"pct_resistant": "% Resistant", "drug_class": "Drug Class", "organism": "Organism"},
    points="all",
)
fig_class.update_layout(
    height=400, margin=dict(t=20, b=60, r=20),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
    legend=dict(orientation="h", yanchor="bottom", y=1.01),
    xaxis=dict(tickangle=-30),
)
st.plotly_chart(fig_class, use_container_width=True)
st.caption("Each dot is one antibiotic within that class. Box shows median and IQR across antibiotics in the class.")
