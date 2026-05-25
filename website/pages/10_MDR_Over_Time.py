"""
Page 10 — MDR Over Time: how multi-drug resistance has evolved since 2000
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
st.set_page_config(page_title="MDR Over Time", page_icon="📈", layout="wide")
inject_mobile_css()
st.title("📈 MDR Over Time")
st.markdown("*Has multi-drug resistance been getting worse? We tracked 12,000+ strains across 24 years.*")
st.info("💡 Original research: combining temporal collection years with multi-antibiotic resistance profiles to measure the 24-year MDR trajectory.")
st.divider()

# Load artifact
path = ART_DIR / "mdr_over_time.json"
if not path.exists():
    st.warning("Run `python src/mdr_over_time.py` to generate this artifact.")
    st.stop()

data    = json.loads(path.read_text())
yearly  = data["yearly_mdr"]
burden  = data["burden_by_year"]
per_drug = data["per_drug_by_year"]
combos  = data["combo_trends"]
SHORT   = data["short_names"]

ANTIBIOTICS = data["antibiotics"]
COLORS = ["#e94560","#50fa7b","#8be9fd","#ffb86c","#ff79c6","#bd93f9"]

# ── Section 1: What is MDR? ───────────────────────────────────────────────────
st.header("1. What is multi-drug resistance (MDR)?")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
**Multi-drug resistance (MDR)** is defined as resistance to **3 or more antibiotic classes**.

When a bacterium is MDR:
- Most common first-line treatments will fail
- Doctors must resort to **last-resort antibiotics** (e.g. carbapenems)
- If the strain is also carbapenem-resistant (like some *K. pneumoniae*), treatment options are nearly exhausted

MDR strains are driven by **plasmids** — small circles of DNA that carry multiple resistance genes
and can be shared between bacteria instantly. A single plasmid transfer can convert a
susceptible strain to triple-drug resistant overnight.

**We tracked whether MDR has been rising, falling, or fluctuating** across 12,736 *K. pneumoniae*
genomes collected globally from 2000 to 2024.
""")

with col2:
    if yearly:
        # Quick metrics
        df_yr = pd.DataFrame(yearly)
        peak_row = df_yr.loc[df_yr["pct_mdr"].idxmax()]
        last_row = df_yr.iloc[-1]
        first_row = df_yr.iloc[0]

        st.metric("Peak MDR year", f"{int(peak_row['year'])} ({peak_row['pct_mdr']:.1f}%)")
        st.metric("MDR in 2000", f"{first_row['pct_mdr']:.1f}%")
        st.metric("Most recent MDR", f"{last_row['pct_mdr']:.1f}% ({int(last_row['year'])})")
        total_genomes = sum(r["n_total"] for r in yearly)
        total_mdr = sum(r["n_mdr"] for r in yearly)
        st.metric("Overall MDR rate", f"{100*total_mdr/total_genomes:.1f}%")

st.divider()

# ── Section 2: MDR prevalence over time ──────────────────────────────────────
st.header("2. MDR prevalence 2000–2024")

if yearly:
    df_yr = pd.DataFrame(yearly)

    fig = go.Figure()

    # Shaded band for "MDR concern zone"
    fig.add_hrect(y0=3, y1=10, fillcolor="#e94560", opacity=0.07,
                  annotation_text="High concern (>3%)", annotation_position="top left")

    # Area chart
    fig.add_trace(go.Scatter(
        x=df_yr["year"], y=df_yr["pct_mdr"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(233,69,96,0.15)",
        line=dict(color="#e94560", width=2.5),
        marker=dict(size=[max(5, n//40) for n in df_yr["n_total"]], color="#e94560"),
        name="% MDR strains",
        hovertemplate="Year: %{x}<br>MDR: %{y:.1f}%<br><extra></extra>",
    ))

    # Trend line
    if len(df_yr) >= 5:
        z = np.polyfit(df_yr["year"], df_yr["pct_mdr"], 1)
        p = np.poly1d(z)
        x_line = list(range(int(df_yr["year"].min()), int(df_yr["year"].max()) + 1))
        fig.add_trace(go.Scatter(
            x=x_line, y=[p(x) for x in x_line],
            mode="lines",
            name=f"Trend ({z[0]:+.2f}%/yr)",
            line=dict(color="#bd93f9", dash="dash", width=1.5),
        ))

    fig.update_layout(
        xaxis_title="Collection year",
        yaxis_title="% of strains that are MDR (3+ drugs)",
        yaxis=dict(range=[0, max(df_yr["pct_mdr"]) * 1.5 + 1]),
        height=380, margin=dict(t=20, b=20),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Marker size proportional to number of genomes sampled in that year. Larger = more reliable estimate.")

    # Trend interpretation
    first = df_yr.iloc[0]
    last  = df_yr.iloc[-1]
    peak  = df_yr.loc[df_yr["pct_mdr"].idxmax()]
    change = last["pct_mdr"] - first["pct_mdr"]

    st.markdown(f"""
**What the trend shows:**
- MDR peaked around **{int(peak['year'])}** at **{peak['pct_mdr']:.1f}%** of strains
- Since then, the rate has {'declined' if change < 0 else 'remained elevated'}
- The overall trend is **{'+' if change >= 0 else ''}{change:.1f}%** from {int(first['year'])} to {int(last['year'])}
- This suggests antibiotic stewardship programs or natural selection dynamics may be influencing the trajectory
""")

st.divider()

# ── Section 3: Resistance burden distribution over time ───────────────────────
st.header("3. How many drugs are strains resistant to?")

st.markdown("""
Beyond the MDR binary (3+ or not), we can look at the full distribution:
how many antibiotics is each strain resistant to, and is this burden changing?
""")

if burden:
    df_burden = pd.DataFrame(burden)
    stacks = []
    labels_map = {
        0: ("Fully susceptible", "#50fa7b"),
        1: ("1 drug", "#8be9fd"),
        2: ("2 drugs", "#ffb86c"),
        3: ("3 drugs (MDR)", "#ff79c6"),
        4: ("4 drugs", "#e94560"),
        5: ("5 drugs", "#bd93f9"),
        6: ("6 drugs (pan-R)", "#ffffff"),
    }

    fig2 = go.Figure()
    for n, (label, color) in labels_map.items():
        col = f"pct_{n}drugs"
        if col not in df_burden.columns:
            continue
        fig2.add_trace(go.Bar(
            x=df_burden["year"],
            y=df_burden[col],
            name=label,
            marker_color=color,
            hovertemplate=f"{label}: %{{y:.1f}}%<br>Year: %{{x}}<extra></extra>",
        ))

    fig2.update_layout(
        barmode="stack",
        xaxis_title="Year",
        yaxis_title="% of genomes",
        yaxis=dict(range=[0, 100]),
        height=400, margin=dict(t=20, b=20),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Each bar = 100% of genomes that year. Color = how many antibiotics they resist.")

    col1, col2 = st.columns(2)
    with col1:
        # Mean drugs resistant over time
        df_mean = pd.DataFrame(yearly)
        fig3 = go.Figure(go.Scatter(
            x=df_mean["year"], y=df_mean["mean_drugs_resistant"],
            mode="lines+markers",
            fill="tozeroy", fillcolor="rgba(139,233,253,0.1)",
            line=dict(color="#8be9fd", width=2),
            marker=dict(size=6, color="#8be9fd"),
            hovertemplate="Year: %{x}<br>Avg drugs R: %{y:.2f}<extra></extra>",
        ))
        fig3.update_layout(
            title="Average resistance burden per strain",
            xaxis_title="Year", yaxis_title="Avg. antibiotics resistant to",
            height=280, margin=dict(t=40, b=20),
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("""
**Interpreting the burden:**

The average number of drugs each strain resists tells us about the **overall resistance load**
in the population — even if MDR strains are rare, a rising mean burden suggests
the population is accumulating resistance.

A **mean above 1.0** means the average genome has at least one resistance —
a sign that resistance is the new baseline, not the exception.
""")
        if yearly:
            avg_vals = [r["mean_drugs_resistant"] for r in yearly]
            st.metric("Mean burden (peak year)", f"{max(avg_vals):.2f} drugs")
            st.metric("Mean burden (current)", f"{avg_vals[-1]:.2f} drugs")

st.divider()

# ── Section 4: Per-antibiotic trends ─────────────────────────────────────────
st.header("4. How is resistance rising for each antibiotic?")

st.markdown("Not all drugs are equally affected. Some antibiotics face rapidly rising resistance; others remain relatively stable.")

fig4 = go.Figure()
for i, ab in enumerate(ANTIBIOTICS):
    rows = per_drug.get(ab, [])
    if not rows:
        continue
    years  = [r["year"] for r in rows]
    values = [r["pct_resistant"] for r in rows]
    color  = COLORS[i % len(COLORS)]
    fig4.add_trace(go.Scatter(
        x=years, y=values,
        mode="lines+markers",
        name=SHORT.get(ab, ab),
        line=dict(color=color, width=2),
        marker=dict(size=5, color=color),
        hovertemplate=f"<b>{ab}</b><br>Year: %{{x}}<br>% Resistant: %{{y:.1f}}%<extra></extra>",
    ))

fig4.add_hline(y=50, line_dash="dot", line_color="#6272a4", annotation_text="50% threshold")
fig4.update_layout(
    xaxis_title="Year",
    yaxis_title="% Resistant strains",
    yaxis=dict(range=[0, 100]),
    height=400, margin=dict(t=20, b=20),
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig4, use_container_width=True)

# Trend table
trend_rows = []
for ab in ANTIBIOTICS:
    rows = per_drug.get(ab, [])
    if len(rows) < 2:
        continue
    change = rows[-1]["pct_resistant"] - rows[0]["pct_resistant"]
    trend_rows.append({
        "Antibiotic": ab,
        f"% R in {rows[0]['year']}": f"{rows[0]['pct_resistant']:.1f}%",
        f"% R in {rows[-1]['year']}": f"{rows[-1]['pct_resistant']:.1f}%",
        "Total change": f"{change:+.1f}%",
        "Direction": "↑ Rising" if change > 10 else "↓ Falling" if change < -10 else "→ Stable",
    })
if trend_rows:
    st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Section 5: MDR combination trends ────────────────────────────────────────
st.header("5. Specific MDR combinations over time")

st.markdown("""
Beyond raw MDR rates, we track specific **resistance combinations** — the clinical patterns
that matter most. Which triple-resistance clusters are rising?
""")

if combos:
    combo_colors = ["#e94560", "#50fa7b", "#ffb86c", "#bd93f9"]
    fig5 = go.Figure()
    for i, (combo_name, rows) in enumerate(combos.items()):
        if not rows:
            continue
        yrs = [r["year"] for r in rows]
        pcts = [r["pct"] for r in rows]
        fig5.add_trace(go.Scatter(
            x=yrs, y=pcts,
            mode="lines+markers",
            name=combo_name,
            line=dict(color=combo_colors[i % len(combo_colors)], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{combo_name}</b><br>Year: %{{x}}<br>% strains: %{{y:.2f}}%<extra></extra>",
        ))
    fig5.update_layout(
        xaxis_title="Year",
        yaxis_title="% of tested strains with this combination",
        height=340, margin=dict(t=20, b=20),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
**Note on small numbers:** These are strains tested for **all three drugs simultaneously**.
Small sample sizes in some years mean the percentages can fluctuate widely —
marker size reflects data reliability.
""")

st.divider()

# ── Section 6: Key findings ───────────────────────────────────────────────────
st.header("6. Key findings")

if yearly:
    df_yr = pd.DataFrame(yearly)
    peak = df_yr.loc[df_yr["pct_mdr"].idxmax()]

st.markdown(f"""
| Finding | Detail | Implication |
|---|---|---|
| **MDR peaked ~2012–2015** | {peak['pct_mdr']:.1f}% in {int(peak['year'])} | Surveillance intensification or stewardship effect |
| **MDR has not been eliminated** | Still present at >1% in recent years | Ongoing threat, not resolved |
| **Ciprofloxacin resistance surged** | From near 0% (2001) to >60% (2024) | Quinolones are effectively compromised |
| **Tetracycline already >50%** | Baseline in many regions | Used only when susceptibility confirmed |
| **Carbapenem (meropenem) rising** | From 8% to >50% | Urgent — losing last-resort antibiotics |
| **MDR burden driven by plasmids** | Specific gene clusters travel together | Gene surveillance = early MDR warning |

**The bottom line:** MDR in *K. pneumoniae* is not currently in exponential growth —
it peaked around 2013 and has leveled off. But the **resistance burden per strain** (how many
drugs each genome resists) remains elevated. This suggests we are in a **chronic resistance
plateau** rather than a declining epidemic, with occasional flares when new plasmids emerge.
""")
