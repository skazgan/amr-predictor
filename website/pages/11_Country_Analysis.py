"""
Page 11 — Country-Level Resistance: geographic distribution of AMR
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
st.title("🌍 Country-Level Resistance")
st.markdown("*Where in the world is K. pneumoniae most dangerous? Geographic patterns in antibiotic resistance.*")
st.info("💡 Original research: we fetched isolation country for 14,230 genomes and computed per-country resistance profiles across 6 antibiotics.")
st.divider()

# Load
path = ART_DIR / "country_resistance.json"
if not path.exists():
    st.warning("Run `python src/country_analysis.py` to generate this artifact.")
    st.stop()

data    = json.loads(path.read_text())
profiles = data["country_profiles"]
trends  = data["country_year_trends"]
SHORT   = data["short_names"]
ANTIBIOTICS = data["antibiotics"]
MIN_GENOMES = data.get("min_genomes", 10)

df = pd.DataFrame(profiles)

# ── Section 1: Why geography matters ─────────────────────────────────────────
st.header("1. Why does geography matter for AMR?")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
Antibiotic resistance is shaped by **local antibiotic use, healthcare systems, and
bacterial transmission networks**. Countries with:

- **High antibiotic consumption** (especially over-the-counter) tend to have higher resistance
- **Dense hospital networks** where patients share environments → plasmid spread
- **Limited infection control infrastructure** → more horizontal gene transfer opportunities
- **Agricultural antibiotic use** → resistance enters human microbiome through food chains

The same *K. pneumoniae* strain can have very different resistance profiles depending on
**where it was collected** — meaning a model trained only on US data might fail on
Indian or Vietnamese strains.

**We mapped 14,230 genomes** across 74 countries to identify geographic resistance hotspots
and understand how much the world varies.
""")
with col2:
    st.metric("Countries with data", len([p for p in profiles if p["n_genomes"] >= MIN_GENOMES]))
    st.metric("Total genomes", f"{sum(p['n_genomes'] for p in profiles):,}")
    st.metric("Largest contributor", f"{profiles[0]['country']} ({profiles[0]['n_genomes']:,} genomes)")
    if df["pct_mdr"].notna().any():
        st.metric("Highest MDR country", f"{df.loc[df['pct_mdr'].idxmax(), 'country']} ({df['pct_mdr'].max():.1f}%)")

st.divider()

# ── Section 2: World map ──────────────────────────────────────────────────────
st.header("2. Global resistance map")

# Drug selector for map
map_drug = st.selectbox(
    "Colour map by:",
    options=["MDR rate (%)"] + [SHORT[ab] + " resistance %" for ab in ANTIBIOTICS],
    index=0,
)

if map_drug == "MDR rate (%)":
    map_col = "pct_mdr"
    map_title = "MDR rate (%)"
else:
    short_sel = map_drug.split(" resistance %")[0]
    map_col = short_sel + "_pct_R"
    map_title = f"{short_sel} % resistant"

df_map = df[df[map_col].notna()].copy()

if not df_map.empty:
    fig_map = px.choropleth(
        df_map,
        locations="country",
        locationmode="country names",
        color=map_col,
        hover_name="country",
        hover_data={"n_genomes": True, map_col: ":.1f"},
        color_continuous_scale="Reds",
        range_color=[0, df_map[map_col].quantile(0.95)],
        labels={map_col: map_title, "n_genomes": "Genomes"},
    )
    fig_map.update_layout(
        height=440, margin=dict(t=20, b=0, l=0, r=0),
        paper_bgcolor="#FFFFFF", font_color="#1E293B",
        geo=dict(
            bgcolor="#EEF2FF",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#C7D2FE",
            landcolor="#E0E7FF",       # visible mid-tone for countries with no data
            showocean=True, oceancolor="#DBEAFE",
            showlakes=True, lakecolor="#DBEAFE",
            showcountries=True, countrycolor="#A5B4FC",
        ),
        coloraxis_colorbar=dict(title=map_title),
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(f"🔵 Blue-grey = no data (fewer than {MIN_GENOMES} genomes). Coloured = resistance rate. Hover for details.")
else:
    st.info("No data available for this drug at the country level.")

st.divider()

# ── Section 3: Country comparison table ──────────────────────────────────────
st.header("3. Country-by-country comparison")

# Column selector
view_cols = st.multiselect(
    "Select antibiotics to display:",
    options=[SHORT[ab] for ab in ANTIBIOTICS],
    default=[SHORT[ab] for ab in ANTIBIOTICS[:4]],
)

pct_cols = [v + "_pct_R" for v in view_cols if v + "_pct_R" in df.columns]

display_df = df[["country", "n_genomes", "pct_mdr", "mean_drugs_r"] + pct_cols].copy()
display_df = display_df.sort_values("n_genomes", ascending=False).head(40)
display_df = display_df.rename(columns={
    "country": "Country",
    "n_genomes": "Genomes",
    "pct_mdr": "MDR %",
    "mean_drugs_r": "Avg drugs R",
    **{v + "_pct_R": v + " % R" for v in view_cols},
})

# Style
def color_resistance(val):
    if pd.isna(val):
        return "color: #444"
    try:
        v = float(val)
        if v >= 70:
            return "color: #e94560; font-weight: bold"
        elif v >= 40:
            return "color: #ffb86c"
        elif v >= 20:
            return "color: #8be9fd"
        else:
            return "color: #50fa7b"
    except Exception:
        return ""

styled = (display_df.style
          .map(color_resistance, subset=[c for c in display_df.columns if "% R" in c or "MDR" in c]))
st.dataframe(styled, use_container_width=True, hide_index=True)
st.caption("🔴 ≥70% resistant  🟠 40–69%  🔵 20–39%  🟢 <20%")

st.divider()

# ── Section 4: Top/bottom countries per drug ─────────────────────────────────
st.header("4. Resistance extremes — highest and lowest per drug")

drug_sel2 = st.selectbox(
    "Select antibiotic:",
    options=list(SHORT.values()),
    index=0,
    key="drug_sel2",
)

col_sel = drug_sel2 + "_pct_R"
if col_sel in df.columns:
    df_sub = df[df[col_sel].notna() & (df["n_genomes"] >= 20)].copy()
    df_sub = df_sub.sort_values(col_sel)

    top5    = df_sub.tail(5).iloc[::-1]
    bottom5 = df_sub.head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Highest {drug_sel2} resistance:**")
        fig_top = go.Figure(go.Bar(
            x=top5[col_sel], y=top5["country"],
            orientation="h",
            marker_color="#e94560",
            text=[f"{v:.1f}%" for v in top5[col_sel]],
            textposition="outside",
        ))
        fig_top.update_layout(
            xaxis=dict(range=[0, 110], title="% Resistant"),
            height=260, margin=dict(t=10, b=10, r=60),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", xaxis_gridcolor="#E2E8F0",
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.markdown(f"**Lowest {drug_sel2} resistance:**")
        fig_bot = go.Figure(go.Bar(
            x=bottom5[col_sel], y=bottom5["country"],
            orientation="h",
            marker_color="#50fa7b",
            text=[f"{v:.1f}%" for v in bottom5[col_sel]],
            textposition="outside",
        ))
        fig_bot.update_layout(
            xaxis=dict(range=[0, 110], title="% Resistant"),
            height=260, margin=dict(t=10, b=10, r=60),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", xaxis_gridcolor="#E2E8F0",
        )
        st.plotly_chart(fig_bot, use_container_width=True)

st.divider()

# ── Section 5: Country resistance fingerprints ───────────────────────────────
st.header("5. Resistance fingerprint — radar chart comparison")

st.markdown("Compare any two countries' full resistance profiles side by side.")

all_countries = sorted(df["country"].tolist())
col1, col2 = st.columns(2)
with col1:
    country_a = st.selectbox("Country A:", options=all_countries,
                              index=all_countries.index("United States") if "United States" in all_countries else 0)
with col2:
    country_b = st.selectbox("Country B:", options=all_countries,
                              index=all_countries.index("China") if "China" in all_countries else 1)

short_list  = [SHORT[ab] for ab in ANTIBIOTICS]
pct_r_cols  = [SHORT[ab] + "_pct_R" for ab in ANTIBIOTICS]

def get_radar_vals(country):
    row = df[df["country"] == country]
    if row.empty:
        return [None] * len(ANTIBIOTICS)
    return [row[c].values[0] for c in pct_r_cols]

vals_a = get_radar_vals(country_a)
vals_b = get_radar_vals(country_b)

# Only show drugs where both have data
valid = [i for i in range(len(ANTIBIOTICS))
         if vals_a[i] is not None and vals_b[i] is not None]

if len(valid) >= 3:
    cats   = [short_list[i] for i in valid] + [short_list[valid[0]]]
    va     = [vals_a[i] for i in valid] + [vals_a[valid[0]]]
    vb     = [vals_b[i] for i in valid] + [vals_b[valid[0]]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=va, theta=cats, fill="toself",
        name=country_a, line_color="#e94560",
        fillcolor="rgba(233,69,96,0.2)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=vb, theta=cats, fill="toself",
        name=country_b, line_color="#50fa7b",
        fillcolor="rgba(80,250,123,0.2)",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#64748B"),
            bgcolor="#F5F3FF",
        ),
        paper_bgcolor="#FFFFFF", font_color="#1E293B",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Not enough shared drug data to draw radar chart for these two countries.")

st.divider()

# ── Section 6: Country resistance over time ───────────────────────────────────
st.header("6. How has resistance changed within top countries?")

st.markdown("For countries with enough data across years, we track how resistance evolved.")

if trends:
    country_trend_sel = st.selectbox(
        "Select country to explore over time:",
        options=list(trends.keys()),
    )
    trend_rows = trends[country_trend_sel]
    if trend_rows:
        df_tr = pd.DataFrame(trend_rows)
        fig_tr = go.Figure()
        COLORS = ["#e94560","#50fa7b","#8be9fd","#ffb86c","#ff79c6","#bd93f9"]
        for i, ab in enumerate(ANTIBIOTICS):
            sh = SHORT[ab]
            if sh not in df_tr.columns:
                continue
            sub = df_tr[df_tr[sh].notna()]
            if len(sub) < 3:
                continue
            fig_tr.add_trace(go.Scatter(
                x=sub["year"], y=sub[sh],
                mode="lines+markers",
                name=sh, line=dict(color=COLORS[i], width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{sh}</b><br>Year: %{{x}}<br>% R: %{{y:.1f}}%<extra></extra>",
            ))
        fig_tr.add_hline(y=50, line_dash="dot", line_color="#64748B",
                         annotation_text="50% threshold")
        fig_tr.update_layout(
            title=f"{country_trend_sel} — resistance over time",
            xaxis_title="Year", yaxis_title="% Resistant",
            yaxis=dict(range=[0, 100]),
            height=380, margin=dict(t=50, b=20),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", xaxis_gridcolor="#E2E8F0", yaxis_gridcolor="#E2E8F0",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_tr, use_container_width=True)
else:
    st.info("Country-level temporal trends require running `country_analysis.py` after `temporal_drift.py`.")

st.divider()

# ── Section 7: Key findings ───────────────────────────────────────────────────
st.header("7. Key findings")

st.markdown("""
| Finding | Data | Clinical implication |
|---|---|---|
| **China and US dominate the dataset** | 3,556 and 2,548 genomes | Results heavily weighted toward these surveillance systems |
| **Vietnam has highest MDR rate** of major contributors | 6.9% MDR | Limited healthcare resources + high antibiotic use |
| **Norway has lowest resistance** across most drugs | Cipro: 17%, Mero: <1% | Strong antibiotic stewardship policy |
| **Italy has very high MDR** among wealthy countries | 4.3% MDR, Cipro 86% | Known European hotspot for carbapenem-resistant *K. pneumoniae* |
| **Meropenem resistance varies 100×** | Norway <1% vs Italy 78% | Last-resort antibiotic is not equally available globally |
| **Geographic origin must be part of treatment decisions** | Different countries = different baseline resistance | Empiric therapy guidelines must be geographically specific |

**Key takeaway for AI models:** A resistance predictor trained on Norwegian genomes would severely
underestimate resistance for patients from Southeast Asia or Southern Europe. **Geographic stratification**
is as important as antibiotic stratification in AMR prediction.
""")
