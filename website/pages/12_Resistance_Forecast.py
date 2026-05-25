"""
Page 12 — Resistance Forecasting: projecting 2025-2030
"""
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent.parent.parent
ART_DIR = ROOT / "artifacts"


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
st.set_page_config(page_title="Resistance Forecast", page_icon="🔮", layout="wide")
inject_mobile_css()
st.title(f"🔮 Resistance Forecast {FORECAST_START}–{FORECAST_END}")
st.markdown("*If current trends continue — what does antibiotic resistance look like in 5 years?*")
st.info(f"💡 Original research: we fit linear and logistic growth models to historical resistance data, then project forward {FORECAST_START}–{FORECAST_END} with confidence intervals.")
st.divider()

# Load
path = ART_DIR / "resistance_forecast.json"
if not path.exists():
    st.warning("Run `python src/resistance_forecast.py` to generate this artifact.")
    st.stop()

_raw = json.loads(path.read_text())
# Support both old flat list format and new dict format
if isinstance(_raw, dict):
    results       = _raw["results"]
    FORECAST_START = _raw.get("forecast_start", 2027)
    FORECAST_END   = _raw.get("forecast_end",   2032)
else:
    results        = _raw
    FORECAST_START = results[0]["forecast"][0]["year"] if results else 2027
    FORECAST_END   = results[0]["forecast"][-1]["year"] if results else 2032


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


COLORS = {
    "ciprofloxacin":                "#e94560",
    "meropenem":                    "#50fa7b",
    "gentamicin":                   "#8be9fd",
    "tetracycline":                 "#ffb86c",
    "trimethoprim/sulfamethoxazole":"#ff79c6",
    "cefepime":                     "#bd93f9",
}

# ── Section 1: How do we forecast? ───────────────────────────────────────────
st.header("1. How do we forecast resistance?")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
We use two mathematical models to project resistance rates:

**Linear model:**
> R(t) = slope × t + intercept

Simple: if resistance has been rising 2%/year, it will continue at 2%/year.
Good for short-term, steady trends. The confidence interval widens further out in time,
reflecting growing uncertainty.

**Logistic model (S-curve):**
> R(t) = L / (1 + e^{−k(t−t₀)})

Biologically realistic: resistance can't exceed 100%. The S-curve models the typical
epidemic pattern — slow start, rapid spread, then saturation as the susceptible
population is exhausted. It has a natural **ceiling (L)** — the maximum resistance
the model predicts even if trends continue.

**Model selection:** We pick the model with higher R² (goodness of fit) on historical data.
Confidence intervals use 80% prediction bands — meaning 4 out of 5 futures should
fall within the shaded region.

⚠️ **Forecasting caveat:** These projections assume **no major policy change, new drug introduction,
or novel outbreak**. They are extrapolations, not predictions. Use them to understand direction
and urgency, not exact future values.
""")

with col2:
    # Summary metrics
    crossing = [r for r in results if r.get("threshold_50_year")]
    logistic_count = sum(1 for r in results if r["model_info"]["model"] == "logistic")

    st.metric("Antibiotics analysed", len(results))
    st.metric("Using logistic model", logistic_count)
    st.metric("Projected to exceed 50%", len(crossing))
    if results:
        fc2030 = [r["forecast"][-1]["predicted"] for r in results]
        st.metric("Highest 2030 projection",
                  f"{max(fc2030):.1f}% ({results[fc2030.index(max(fc2030))]['short_name']})")

st.divider()

# ── Section 2: All-antibiotic forecast overview ───────────────────────────────
st.header(f"2. {FORECAST_START}–{FORECAST_END} forecast overview")

st.markdown("Historical data (solid lines) extended into the future (dashed) with 80% confidence bands.")

fig_all = go.Figure()

for r in results:
    ab    = r["antibiotic"]
    color = COLORS.get(ab, "#cdd6f4")
    hist  = r["historical"]
    fc    = r["forecast"]

    # Historical line
    fig_all.add_trace(go.Scatter(
        x=[h["year"] for h in hist],
        y=[h["pct_resistant"] for h in hist],
        mode="lines",
        name=r["short_name"],
        line=dict(color=color, width=2),
        legendgroup=ab,
        showlegend=True,
        hovertemplate=f"<b>{ab}</b><br>Year: %{{x}}<br>Historical: %{{y:.1f}}%<extra></extra>",
    ))

    # Forecast confidence band
    fig_all.add_trace(go.Scatter(
        x=[f["year"] for f in fc] + [f["year"] for f in fc][::-1],
        y=[f["upper_80"] for f in fc] + [f["lower_80"] for f in fc][::-1],
        fill="toself",
        fillcolor=hex_to_rgba(color, 0.12) if color.startswith("#") else "rgba(128,128,128,0.12)",
        line=dict(width=0),
        showlegend=False, legendgroup=ab,
        hoverinfo="skip",
    ))

    # Forecast line (dashed)
    # Connect last historical to first forecast
    connect_x = [hist[-1]["year"]] + [f["year"] for f in fc]
    connect_y = [hist[-1]["pct_resistant"]] + [f["predicted"] for f in fc]
    fig_all.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        showlegend=False, legendgroup=ab,
        hovertemplate=f"<b>{ab}</b><br>Year: %{{x}}<br>Forecast: %{{y:.1f}}%<extra></extra>",
    ))

# 50% threshold line
fig_all.add_hline(y=50, line_dash="dot", line_color="#6272a4",
                  annotation_text="50% — majority resistant",
                  annotation_position="bottom right")

fig_all.add_vline(x=FORECAST_START - 0.5, line_dash="dash", line_color="#6272a4",
                  annotation_text="Forecast →", annotation_position="top left")

fig_all.update_layout(
    xaxis_title="Year",
    yaxis_title="% Resistant strains",
    yaxis=dict(range=[0, 105]),
    height=450, margin=dict(t=20, b=20),
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_all, use_container_width=True)
st.caption("Shaded bands = 80% prediction interval. Dashed = forecast. Solid = historical data.")

st.divider()

# ── Section 3: Individual antibiotic deep-dives ───────────────────────────────
st.header("3. Deep-dive: individual antibiotic forecast")

ab_options = {r["antibiotic"]: r["short_name"] for r in results}
selected_ab = st.selectbox(
    "Select antibiotic:",
    options=list(ab_options.keys()),
    format_func=lambda x: ab_options[x],
)

sel = next((r for r in results if r["antibiotic"] == selected_ab), None)
if sel:
    col1, col2 = st.columns([3, 2])
    with col1:
        hist = sel["historical"]
        fc   = sel["forecast"]
        color = COLORS.get(selected_ab, "#cdd6f4")

        fig_single = go.Figure()

        # Historical
        fig_single.add_trace(go.Scatter(
            x=[h["year"] for h in hist],
            y=[h["pct_resistant"] for h in hist],
            mode="lines+markers",
            name="Historical data",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
        ))

        # CI band
        fig_single.add_trace(go.Scatter(
            x=[f["year"] for f in fc] + [f["year"] for f in fc][::-1],
            y=[f["upper_80"] for f in fc] + [f["lower_80"] for f in fc][::-1],
            fill="toself",
            fillcolor="rgba(100,100,200,0.15)",
            line=dict(width=0),
            name="80% confidence interval",
        ))

        # Forecast
        connect_x = [hist[-1]["year"]] + [f["year"] for f in fc]
        connect_y = [hist[-1]["pct_resistant"]] + [f["predicted"] for f in fc]
        fig_single.add_trace(go.Scatter(
            x=connect_x, y=connect_y,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#bd93f9", width=2.5, dash="dash"),
            marker=dict(size=8, symbol="diamond", color="#bd93f9"),
            hovertemplate="Year: %{x}<br>Forecast: %{y:.1f}%<extra></extra>",
        ))

        fig_single.add_hline(y=50, line_dash="dot", line_color="#6272a4",
                              annotation_text="50% majority-resistant threshold")
        fig_single.add_vline(x=FORECAST_START - 0.5, line_dash="dash", line_color="#444",
                              annotation_text="Now →")

        fig_single.update_layout(
            title=f"{selected_ab} — {sel['model_info']['model']} model forecast",
            xaxis_title="Year", yaxis_title="% Resistant",
            yaxis=dict(range=[0, 105]),
            height=380, margin=dict(t=50, b=20),
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_single, use_container_width=True)

    with col2:
        mi = sel["model_info"]
        st.markdown("**Model parameters:**")
        st.metric("Model", mi["model"].capitalize())
        st.metric("R² (historical fit)", f"{mi['r2']:.3f}")
        st.metric("RMSE", f"{mi['rmse']:.2f}%")
        if mi["model"] == "logistic":
            st.metric("Saturation ceiling (L)", f"{mi.get('L', '?'):.1f}%")
            st.metric("Growth rate (k)", f"{mi.get('k', '?'):.4f}")
        else:
            st.metric("Annual slope", f"{mi.get('slope', '?'):+.3f}%/yr")

        if sel.get("threshold_50_year"):
            st.error(f"⚠️ Projected to cross **50% resistance** around **{sel['threshold_50_year']}**")
        else:
            st.success(f"✅ Projected to stay below 50% through {FORECAST_END}")

    # Forecast table — full width below the two columns
    st.markdown("**Forecast table:**")
    fc_df = pd.DataFrame(sel["forecast"])
    fc_df.columns = ["Year", "Predicted %", "Lower 80%", "Upper 80%"]
    st.dataframe(fc_df, hide_index=True, use_container_width=True)

st.divider()

# ── Section 4: Forecast summary table ────────────────────────────────────────
st.header(f"4. {FORECAST_END} projection summary")

summary_rows = []
for r in sorted(results, key=lambda x: -x["forecast"][-1]["predicted"]):
    fc_end   = r["forecast"][-1]
    fc_start = r["forecast"][0]
    current_year = r.get("current_year", "latest")
    summary_rows.append({
        "Antibiotic": r["antibiotic"],
        f"Current ({current_year})": f"{r['current_pct']:.1f}%",
        f"{FORECAST_START} forecast": f"{fc_start['predicted']:.1f}% [{fc_start['lower_80']:.1f}–{fc_start['upper_80']:.1f}%]",
        f"{FORECAST_END} forecast": f"{fc_end['predicted']:.1f}% [{fc_end['lower_80']:.1f}–{fc_end['upper_80']:.1f}%]",
        "Model": r["model_info"]["model"].capitalize(),
        "Crosses 50%": "⚠️ Yes" if r.get("threshold_50_year") else "✅ No",
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Section 5: What would change the forecast? ───────────────────────────────
st.header("5. What could change these projections?")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
### 📉 Factors that could improve the outlook

- **New antibiotic classes** entering clinical use (e.g. cefiderocol, ceftazidime-avibactam)
- **Antibiotic stewardship programs** reducing selection pressure
- **Phage therapy** as alternative to antibiotics
- **Rapid diagnostics** enabling targeted rather than empiric treatment
- **Vaccine development** against *K. pneumoniae* reducing infection rates
- **International AMR targets** (UN 2024 declaration)
""")

with col2:
    st.markdown("""
### 📈 Factors that could worsen the outlook

- **Continued over-the-counter antibiotic use** (especially Asia, Africa)
- **Agricultural antibiotic use** as growth promoters
- **Carbapenem spread** — currently limited but accelerating (meropenem data)
- **New plasmid types** with novel gene combinations
- **Climate change** extending transmission seasons
- **Healthcare system stress** → reduced infection control
""")

st.divider()

# ── Section 6: Clinical decision framework ─────────────────────────────────
st.header("6. What this means for treatment decisions today")

st.markdown("""
| Antibiotic | Current status | 2030 projection | Recommendation |
|---|---|---|---|
| **Ciprofloxacin** | >50% resistant NOW | Stable high (53%) | Do not use empirically without susceptibility data |
| **Meropenem** | 53% resistant | Stable high (51%) | Last-resort status justified; test always |
| **Tetracycline** | 78% resistant | Likely stays high | Avoid empiric use; check local rates |
| **Cefepime** | 64% resistant | ~50% (plateau) | Test before prescribing |
| **TMP/SMX** | 43% resistant | Approaching 50% | Monitor closely; empiric use risky |
| **Gentamicin** | 23% resistant | Wide uncertainty | Still viable with susceptibility confirmation |

**The core message:** For *K. pneumoniae* infections, empiric antibiotic choice (before
susceptibility results arrive) is increasingly dangerous. Our model provides an
**individualized prediction** — factoring in the specific genome's resistance gene
profile — rather than relying on population-average resistance rates.

This is exactly why genomics-based AMR prediction matters: the population average
says "50% resistant", but your specific patient's strain may be susceptible or
resistant based on its specific gene complement. **Precision matters.**
""")
