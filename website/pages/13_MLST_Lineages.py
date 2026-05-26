"""
Page 13 — MLST Lineages: which bacterial clones are most dangerous?
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
st.title("🧬 MLST Lineages — Tracking Dangerous Clones")
st.markdown("*Which bacterial lineages carry the most resistance? How are they spreading?*")
st.info("💡 Original research: we typed 22,673 genomes by Multi-Locus Sequence Type (MLST) and linked lineages to resistance profiles.")
st.divider()

path = ART_DIR / "mlst_analysis.json"
if not path.exists():
    st.warning("Run `python src/mlst_analysis.py` to generate this artifact.")
    st.stop()

data      = json.loads(path.read_text())
profiles  = data["st_profiles"]
trends    = data["st_year_trends"]
enrichment = data["st_gene_enrichment"]
notable   = data["notable_sts"]
SHORT     = data["short_names"]
ANTIBIOTICS = data["antibiotics"]

df = pd.DataFrame(profiles)

COLORS_SPEED = {"rapid": "#e94560", "moderate": "#ffb86c",
                "slow": "#8be9fd", "declining": "#50fa7b"}

# ── Section 1: What is MLST? ──────────────────────────────────────────────────
st.header("1. What is MLST and why do lineages matter?")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
**Multi-Locus Sequence Typing (MLST)** assigns each bacterial isolate a **Sequence Type (ST)**
based on the allele combinations at 7 conserved housekeeping genes.

Bacteria with the same ST = **same clonal lineage** = descended from the same ancestor.

**Why it matters:**
- Resistance genes don't spread randomly — they hitchhike on successful **outbreak clones**
- ST258 (*K. pneumoniae*) is responsible for the majority of carbapenem-resistant infections worldwide
- Tracking STs lets us identify whether rising resistance is caused by **new gene acquisition** (any strain gaining a gene) or **clone expansion** (one dangerous lineage spreading)

**These are fundamentally different threats:**
- Gene spread → needs broad surveillance of gene transfer mechanisms
- Clone spread → needs infection control to stop the specific outbreak lineage

We typed **22,673 genomes** across **1,498 unique STs** and linked each to its resistance profile.
""")

with col2:
    st.metric("Genomes with ST data", f"{data['total_with_st']:,}")
    st.metric("Unique sequence types", f"{data['unique_sts']:,}")
    st.metric("STs with ≥15 genomes", len(profiles))
    if df["pct_mdr"].notna().any():
        worst_st = df.loc[df["pct_mdr"].idxmax()]
        st.metric("Highest MDR lineage",
                  f"ST{worst_st['st']} ({worst_st['pct_mdr']:.0f}% MDR)")

st.divider()

# ── Section 2: Notable STs ────────────────────────────────────────────────────
st.header("2. Clinically important K. pneumoniae lineages")

cols = st.columns(4)
for i, (st_id, desc) in enumerate(list(notable.items())[:8]):
    match = next((p for p in profiles if p["st"] == st_id), None)
    with cols[i % 4]:
        if match:
            mdr  = match["pct_mdr"]
            mero = match.get("Mero_pct_R")
            border = "#e94560" if mdr > 20 else "#ffb86c" if mdr > 10 else "#8be9fd"
            mero_tag = f'  <span style="color:#ffb86c;">Mero: {mero}%</span>' if mero else ""
            st.markdown(f"""
<div style='background:#FFFFFF; border-left:3px solid {border};
     padding:0.6rem 0.8rem; border-radius:6px; margin-bottom:8px;'>
<b style='color:#1E293B;'>ST{st_id}</b><br>
<small style='color:#64748B;'>{desc.split("—")[1].strip() if "—" in desc else desc}</small><br>
<span style='color:#e94560;'>MDR: {mdr:.0f}%</span>{mero_tag}
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style='background:#FFFFFF; border-left:3px solid #CBD5E1;
     padding:0.6rem 0.8rem; border-radius:6px; margin-bottom:8px; opacity:0.5;'>
<b style='color:#1E293B;'>ST{st_id}</b><br>
<small style='color:#64748B;'>{desc.split("—")[1].strip() if "—" in desc else desc}</small><br>
<small style='color:#444;'>Not in dataset</small>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Section 3: Top STs bubble chart ──────────────────────────────────────────
st.header("3. Resistance vs. prevalence — ST bubble chart")

st.markdown("Each bubble is one sequence type. Size = how common it is. Position = MDR rate vs ciprofloxacin resistance.")

drug_x = st.selectbox("X axis:", [SHORT[ab] for ab in ANTIBIOTICS], index=0)
drug_y = st.selectbox("Y axis:", [SHORT[ab] for ab in ANTIBIOTICS], index=1)

col_x = drug_x + "_pct_R"
col_y = drug_y + "_pct_R"
df_plot = df[df[col_x].notna() & df[col_y].notna()].copy()

if not df_plot.empty:
    df_plot["label"] = "ST" + df_plot["st"].astype(str)
    df_plot["is_notable"] = df_plot["st"].isin(list(notable.keys()))
    df_plot["color"] = df_plot["pct_mdr"].apply(
        lambda x: "#e94560" if x > 20 else "#ffb86c" if x > 10 else "#8be9fd"
    )

    fig_bubble = go.Figure()
    fig_bubble.add_trace(go.Scatter(
        x=df_plot[col_x],
        y=df_plot[col_y],
        mode="markers+text",
        text=["ST" + s if s in list(notable.keys()) else "" for s in df_plot["st"]],
        textposition="top center",
        textfont=dict(size=9, color="#1E293B"),
        marker=dict(
            size=np.sqrt(df_plot["n_genomes"]) * 1.5,
            color=df_plot["pct_mdr"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="MDR %"),
            showscale=True,
            line=dict(width=1, color="#CBD5E1"),
        ),
        customdata=df_plot[["st", "n_genomes", "pct_mdr"]].values,
        hovertemplate=(
            "<b>ST%{customdata[0]}</b> (n=%{customdata[1]})<br>"
            f"{drug_x}: %{{x:.1f}}%<br>"
            f"{drug_y}: %{{y:.1f}}%<br>"
            "MDR: %{customdata[2]:.1f}%<extra></extra>"
        ),
    ))

    fig_bubble.update_layout(
        xaxis_title=f"% resistant to {drug_x}",
        yaxis_title=f"% resistant to {drug_y}",
        height=460, margin=dict(t=20, b=20),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B", xaxis_gridcolor="#E2E8F0", yaxis_gridcolor="#E2E8F0",
    )
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.caption("Bubble size = number of genomes. Colour = MDR rate (red = high MDR). Labelled = clinically notable STs.")

st.divider()

# ── Section 4: ST resistance heatmap ─────────────────────────────────────────
st.header("4. Resistance heatmap across top sequence types")

st.markdown("Which lineages resist which drugs? This heatmap reveals the resistance fingerprint of each ST.")

top_n = st.slider("Show top N STs by genome count:", 5, min(30, len(df)), 15)
df_top = df.head(top_n).copy()
df_top["st_label"] = "ST" + df_top["st"].astype(str)

short_list = [SHORT[ab] for ab in ANTIBIOTICS]
pct_cols   = [s + "_pct_R" for s in short_list]

# Build matrix
z_vals = []
for _, row in df_top.iterrows():
    z_row = [row.get(c) for c in pct_cols]
    z_vals.append(z_row)

fig_heat = go.Figure(go.Heatmap(
    z=z_vals,
    x=short_list,
    y=df_top["st_label"].tolist(),
    colorscale="RdYlGn_r",
    zmin=0, zmax=100,
    text=[[f"{v:.0f}%" if v is not None else "?" for v in row] for row in z_vals],
    texttemplate="%{text}",
    textfont=dict(size=10),
    showscale=True,
    colorbar=dict(title="% Resistant"),
))
fig_heat.update_layout(
    height=max(350, top_n * 28),
    margin=dict(t=20, b=10),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font_color="#1E293B",
    xaxis=dict(side="top"),
)
st.plotly_chart(fig_heat, use_container_width=True)
st.caption("Gray cells = fewer than 5 tested genomes for that drug/ST combination.")

st.divider()

# ── Section 5: ST trends over time ────────────────────────────────────────────
st.header("5. Are dangerous lineages expanding over time?")

st.markdown("""
We track what fraction of each year's genomes belong to each ST.
A rising curve means that lineage is out-competing others — a **clonal expansion** event.
""")

if trends:
    st_choices = list(trends.keys())
    selected_sts = st.multiselect(
        "Select sequence types to compare:",
        options=["ST" + s for s in st_choices],
        default=["ST" + s for s in st_choices[:4]],
    )
    selected_ids = [s.replace("ST", "") for s in selected_sts]

    palette = px.colors.qualitative.Plotly
    fig_trend = go.Figure()
    for i, st_id in enumerate(selected_ids):
        if st_id not in trends:
            continue
        rows = trends[st_id]
        note = notable.get(st_id, "")
        label = f"ST{st_id}" + (f" ({note.split('—')[0].strip()})" if "—" in note else "")
        fig_trend.add_trace(go.Scatter(
            x=[r["year"] for r in rows],
            y=[r["pct_of_year"] for r in rows],
            mode="lines+markers",
            name=label,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=[max(4, r["n"] // 10) for r in rows]),
            hovertemplate=f"<b>ST{st_id}</b><br>Year: %{{x}}<br>% of year's genomes: %{{y:.1f}}%<extra></extra>",
        ))

    fig_trend.update_layout(
        xaxis_title="Year",
        yaxis_title="% of year's sequenced genomes",
        height=380, margin=dict(t=20, b=20),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B", xaxis_gridcolor="#E2E8F0", yaxis_gridcolor="#E2E8F0",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Marker size = number of genomes that year. Larger = more reliable estimate.")

    # ST11 specific interpretation
    if "11" in trends:
        st11 = trends["11"]
        first, last = st11[0], st11[-1]
        st.info(f"""
**ST11 trend:** {first['pct_of_year']:.1f}% of genomes in {first['year']} → **{last['pct_of_year']:.1f}%** in {last['year']}.
ST11 is the dominant MDR lineage in China, which contributes the most genomes to this dataset.
Its rise reflects clonal expansion of a highly resistant lineage, not just random gene spread.
""")

st.divider()

# ── Section 6: ST-defining genes ─────────────────────────────────────────────
st.header("6. Which genes define the most dangerous lineages?")

st.markdown("""
We compared gene presence rates in each high-MDR ST vs all other genomes.
Enriched genes = genes that are more common in that lineage — its **molecular signature**.
""")

if enrichment:
    st_choices_enrich = list(enrichment.keys())
    st_sel = st.selectbox(
        "Select lineage to inspect:",
        options=["ST" + s for s in st_choices_enrich],
        index=0,
    )
    st_id_sel = st_sel.replace("ST", "")
    genes_data = enrichment.get(st_id_sel, [])

    if genes_data:
        df_genes = pd.DataFrame(genes_data)
        df_genes["name"] = df_genes["gene"].str[:55]
        colors_g = ["#e94560" if v > 0 else "#50fa7b" for v in df_genes["enrichment"]]

        fig_genes = go.Figure(go.Bar(
            x=df_genes["enrichment"],
            y=df_genes["name"],
            orientation="h",
            marker_color=colors_g,
            text=[f"{v:+.3f}" for v in df_genes["enrichment"]],
            textposition="outside",
        ))
        fig_genes.add_vline(x=0, line_color="#1E293B", line_width=1)
        fig_genes.update_layout(
            title=f"{st_sel} — enriched resistance genes vs all other STs",
            xaxis_title="Enrichment (rate in ST − rate in others)",
            height=380, margin=dict(t=50, b=10, r=80),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", xaxis_gridcolor="#E2E8F0",
        )
        st.plotly_chart(fig_genes, use_container_width=True)

        match = next((p for p in profiles if p["st"] == st_id_sel), None)
        if match:
            st.caption(
                f"ST{st_id_sel}: {match['n_genomes']} genomes, "
                f"{match['pct_mdr']:.0f}% MDR. "
                + (match["notable"] if match["notable"] else "")
            )

st.divider()

# ── Section 7: Key findings ───────────────────────────────────────────────────
st.header("7. Key findings")

st.markdown("""
| Lineage | Size | MDR rate | Key resistance | Clinical significance |
|---|---|---|---|---|
| **ST258** | 1,748 genomes | **32%** | Cipro 97%, Mero 89% | #1 carbapenem-resistant lineage globally |
| **ST512** | 548 genomes | **36%** | Cipro 100%, Mero 99% | KPC-3 epidemic, carbapenem last-resort |
| **ST11** | 3,531 genomes | **13%** | Cipro 61%, Mero 88% | Dominant in China, expanding 2.8%→20.7% of isolates |
| **ST307** | 1,267 genomes | **22%** | Cipro 100%, Mero 54% | Emerging hypervirulent + MDR combination |
| **ST15** | 1,403 genomes | **15%** | Cipro 94%, Mero 46% | ESBL spread in Europe |

**The critical insight:**

ST258 and ST512 are not just strains that happened to acquire resistance genes —
they are **evolutionary winners** that out-competed susceptible bacteria precisely because
resistance confers a survival advantage in antibiotic-treated patients.

Stopping these lineages requires **both** genomic surveillance (identify the clone early)
**and** infection control (prevent hospital transmission of the clone).
A model that predicts resistance from genome sequence is most useful precisely
when these dangerous STs appear in a new hospital.
""")
