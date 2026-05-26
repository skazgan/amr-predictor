"""
Page 8 — Temporal Drift: how resistance patterns change over time
"""
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

ROOT    = Path(__file__).parent.parent.parent
ART_DIR = ROOT / "artifacts"


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css, page_info_expander
inject_mobile_css()
page_info_expander("""
**Temporal drift** — The gradual change in a model's predictive accuracy over time, caused by shifts in the underlying bacterial population. As new resistant clones emerge and old ones disappear, a model trained on historical data may become less accurate.

**Training cutoff** — The latest year of genome data included when training the model. Data collected after this date are "unseen" to the model and represent real-world drift.

**Why it matters clinically** — Resistance prevalence in hospitals changes year-to-year. A model trained in 2018 may under-predict the carbapenem resistance rates seen in 2024, because resistant lineages like ST258 have expanded since then.

**AUC over time** — We evaluate how model accuracy (AUC) changes when applied to genomes from progressively later time periods. A flat line = stable model; a declining line = drift is occurring.

**Retraining** — The practical response to drift: periodically retraining the model on more recent data to keep it aligned with current resistance epidemiology, analogous to updating local antibiograms annually.
""")
st.title("📅 Temporal Drift")
st.markdown("*How do resistance patterns — and model accuracy — change over time?*")
st.info("💡 Original analysis: we split our dataset by collection year to reveal resistance trends and model decay.")
st.divider()

ANTIBIOTICS = [
    "ciprofloxacin", "meropenem", "gentamicin",
    "tetracycline", "trimethoprim/sulfamethoxazole", "cefepime",
]
SHORT = {
    "ciprofloxacin":                "Cipro",
    "meropenem":                    "Mero",
    "gentamicin":                   "Gent",
    "tetracycline":                 "Tet",
    "trimethoprim/sulfamethoxazole":"TMP/SMX",
    "cefepime":                     "Cef",
}
COLORS = ["#e94560","#50fa7b","#8be9fd","#ffb86c","#ff79c6","#bd93f9"]

# Check artifacts exist
required = ["temporal_prevalence.json", "temporal_model_decay.json",
            "temporal_gene_trends.json"]
missing  = [f for f in required if not (ART_DIR / f).exists()]
if missing:
    st.warning(f"Some artifacts not yet generated: {missing}\n\n"
               "Run `python src/temporal_drift.py` to generate them.")
    st.stop()

prevalence  = json.loads((ART_DIR / "temporal_prevalence.json").read_text())
decay_data  = json.loads((ART_DIR / "temporal_model_decay.json").read_text())
gene_trends = json.loads((ART_DIR / "temporal_gene_trends.json").read_text())

# ── Section 1: Why temporal drift matters ─────────────────────────────────────
st.header("1. Why temporal drift matters")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
A model trained on 2015–2019 genomes may fail on 2022 genomes — not because
the model is wrong, but because **the bacteria have evolved**.

Resistance genes spread through hospitals and communities over time.
A drug that was 20% resistant in 2015 may be 40% resistant in 2023.
A model that doesn't know this will underpredict resistance on new strains.

**This is called temporal drift** — the gap between the world the model was
trained on and the world it now faces.

We measure it by:
1. Training on genomes collected **before 2020**
2. Testing on genomes collected **2020 and later**
3. Comparing this AUC vs a random 80/20 split (what the model reports normally)
""")
with col2:
    # How many genomes have year data
    n_with_year = sum(
        1 for ab_rows in prevalence.values()
        for r in ab_rows
    )
    all_years = sorted(set(
        r["year"]
        for ab_rows in prevalence.values()
        for r in ab_rows
    ))
    st.metric("Genomes with collection year", f"{n_with_year:,}")
    if all_years:
        st.metric("Year range", f"{min(all_years)} – {max(all_years)}")
    st.metric("Antibiotics analysed", len([ab for ab, rows in prevalence.items() if rows]))

st.divider()

# ── Section 2: Resistance trends over time ────────────────────────────────────
st.header("2. Resistance rates over time — per antibiotic")

st.markdown("How has the percentage of resistant strains changed year by year?")

fig = go.Figure()
for i, ab in enumerate(ANTIBIOTICS):
    rows = prevalence.get(ab, [])
    if not rows:
        continue
    years     = [r["year"] for r in rows]
    pct_r     = [r["pct_resistant"] for r in rows]
    totals    = [r["total"] for r in rows]
    color     = COLORS[i % len(COLORS)]

    fig.add_trace(go.Scatter(
        x=years, y=pct_r, mode="lines+markers",
        name=SHORT[ab], line=dict(color=color, width=2),
        marker=dict(size=[max(5, t//20) for t in totals], color=color),
        hovertemplate=(
            f"<b>{ab}</b><br>"
            "Year: %{x}<br>"
            "Resistant: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

fig.add_hline(y=50, line_dash="dot", line_color="#64748B",
              annotation_text="50% threshold")
fig.update_layout(
    xaxis_title="Collection year",
    yaxis_title="% Resistant strains",
    yaxis=dict(range=[0, 100]),
    height=420, margin=dict(t=20, b=40),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font_color="#1E293B", xaxis_gridcolor="#E2E8F0", yaxis_gridcolor="#E2E8F0",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Marker size = number of genomes in that year. Larger = more data, more reliable.")

# Trend table
trend_rows = []
for ab in ANTIBIOTICS:
    rows = prevalence.get(ab, [])
    if len(rows) < 2:
        continue
    first_yr  = rows[0]["year"]
    last_yr   = rows[-1]["year"]
    first_pct = rows[0]["pct_resistant"]
    last_pct  = rows[-1]["pct_resistant"]
    change    = last_pct - first_pct
    trend_rows.append({
        "Antibiotic": ab,
        f"% R in {first_yr}": f"{first_pct:.1f}%",
        f"% R in {last_yr}":  f"{last_pct:.1f}%",
        "Change":              f"{change:+.1f}%",
        "Direction":          "↑ Rising" if change > 5 else "↓ Falling" if change < -5 else "→ Stable",
    })

if trend_rows:
    st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Section 3: Model decay ────────────────────────────────────────────────────
st.header("3. Model decay — does performance drop on newer strains?")

st.markdown("""
We train on genomes from **before 2020** and test on **2020 and later**.
The difference vs. a random split shows how much the model has drifted.

- **Positive drift** = AUC is lower on new data → the model is becoming outdated
- **Near-zero drift** = the model generalises well across time
""")

if decay_data:
    df_decay = pd.DataFrame(decay_data)
    df_decay["short"] = df_decay["antibiotic"].map(SHORT)
    df_decay = df_decay.dropna(subset=["auc_temporal","auc_random"])

    col1, col2 = st.columns([3, 2])
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Random split AUC (normal)",
            x=df_decay["short"],
            y=df_decay["auc_random"],
            marker_color="#50fa7b",
        ))
        fig2.add_trace(go.Bar(
            name="Temporal split AUC (train <2020, test ≥2020)",
            x=df_decay["short"],
            y=df_decay["auc_temporal"],
            marker_color="#e94560",
        ))
        fig2.update_layout(
            barmode="group",
            yaxis=dict(title="ROC-AUC", range=[0.5, 1.0]),
            height=340, margin=dict(t=20, b=10),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("**Drift per antibiotic:**")
        for _, row in df_decay.sort_values("drift", ascending=False).iterrows():
            drift = row["drift"]
            color = "#e94560" if drift > 0.05 else "#ffb86c" if drift > 0.02 else "#50fa7b"
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"padding:0.4rem 0.8rem; background:#FFFFFF; border-radius:6px; "
                f"margin-bottom:4px; border-left:3px solid {color};'>"
                f"<span style='color:#1E293B;'>{row['short']}</span>"
                f"<span style='color:{color}; font-weight:bold;'>{drift:+.3f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("""
<br>
<small style='color:#64748B;'>
🔴 > 0.05 = significant drift<br>
🟡 0.02–0.05 = moderate drift<br>
🟢 < 0.02 = stable
</small>
""", unsafe_allow_html=True)

    # Interpretation
    worst = df_decay.loc[df_decay["drift"].idxmax()] if not df_decay.empty else None
    best  = df_decay.loc[df_decay["drift"].idxmin()] if not df_decay.empty else None
    if worst is not None and best is not None:
        st.markdown(f"""
**What this tells us:**

The **{worst['antibiotic']}** model shows the most drift ({worst['drift']:+.3f}) —
meaning resistance patterns for this antibiotic have changed more since 2020.
This model would benefit from retraining on recent data.

The **{best['antibiotic']}** model is the most stable ({best['drift']:+.3f}) —
its resistance mechanisms haven't shifted as much, so the model stays accurate on new strains.
""")

st.divider()

# ── Section 4: Gene frequency trends ─────────────────────────────────────────
st.header("4. Resistance gene frequency trends")

st.markdown("""
Which resistance genes are spreading over time?
A rising gene frequency means that mechanism is becoming more common in the bacterial population.
""")

if gene_trends:
    gene_choice = st.selectbox(
        "Select gene to explore:",
        options=[g["gene"][:60] for g in gene_trends],
        index=0,
    )
    selected = next((g for g in gene_trends if g["gene"][:60] == gene_choice), None)
    if selected and selected["trend"]:
        df_trend = pd.DataFrame(selected["trend"])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_trend["year"], y=df_trend["frequency"],
            mode="lines+markers",
            marker=dict(size=[max(5, n//10) for n in df_trend["n"]], color="#ffb86c"),
            line=dict(color="#ffb86c", width=2),
            hovertemplate="Year: %{x}<br>Frequency: %{y:.1f}%<extra></extra>",
        ))
        # Linear trend line
        if len(df_trend) >= 3:
            import numpy as np
            z = np.polyfit(df_trend["year"], df_trend["frequency"], 1)
            p = np.poly1d(z)
            x_line = list(range(int(df_trend["year"].min()), int(df_trend["year"].max())+1))
            fig3.add_trace(go.Scatter(
                x=x_line, y=[p(x) for x in x_line],
                mode="lines", name=f"Trend ({z[0]:+.2f}%/yr)",
                line=dict(color="#e94560", dash="dash", width=1.5),
            ))

        fig3.update_layout(
            title=f"{gene_choice[:50]} — frequency over time",
            xaxis_title="Year", yaxis_title="% of genomes carrying this gene",
            yaxis=dict(range=[0, max(df_trend["frequency"])*1.3]),
            height=320, margin=dict(t=40, b=10),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font_color="#1E293B", xaxis_gridcolor="#E2E8F0", yaxis_gridcolor="#E2E8F0",
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Marker size = number of genomes sampled in that year.")

st.divider()

# ── Section 5: Key findings ───────────────────────────────────────────────────
st.header("5. Key temporal findings")

st.markdown("""
| Finding | Clinical implication |
|---|---|
| **Rising resistance** in some antibiotics | Treatment guidelines must be updated more frequently |
| **Model drift varies by antibiotic** | Models need retraining on different schedules per drug |
| **Gene frequency shifts** reveal spreading mechanisms | Surveillance should prioritise rising genes |
| **Pre-2020 training still useful for stable drugs** | Not all models need constant retraining |

**The broader lesson:** An AMR prediction system in production needs a **retraining schedule** tied to how fast each drug's resistance landscape changes — not a one-size-fits-all update cycle.
""")
