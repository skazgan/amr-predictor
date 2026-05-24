"""
Page 9 — Gene Emergence Curves: epidemic-style spread of resistance genes
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

st.set_page_config(page_title="Gene Emergence", page_icon="🧬", layout="wide")
st.title("🧬 Gene Emergence Curves")
st.markdown("*How do resistance genes spread through bacterial populations — like epidemic waves?*")
st.info("💡 Original research: We fit epidemic-style growth curves to each gene's year-by-year prevalence, revealing which resistance mechanisms are accelerating vs declining.")
st.divider()

# Load artifact
emergence_path = ART_DIR / "gene_emergence.json"
if not emergence_path.exists():
    st.warning("Run `python src/gene_emergence.py` to generate this artifact.")
    st.stop()

data = json.loads(emergence_path.read_text())

# ── Section 1: The concept ────────────────────────────────────────────────────
st.header("1. Resistance genes spread like epidemics")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
Think of each resistance gene as an **infectious agent** spreading through the bacterial population.

When a new resistance gene appears on a plasmid:
1. It starts at low frequency — only a few bacteria carry it
2. If it confers a **survival advantage** (selection pressure from antibiotic use), it spreads
3. It follows an **S-shaped logistic curve** — slow start, rapid spread, then plateau as it saturates
4. Eventually it either **dominates** (high plateau) or **declines** (better mechanisms replace it)

**This is called horizontal gene transfer (HGT)** — bacteria share resistance genes directly,
bypassing inheritance. A resistance gene that appears in one bacterium today can be in a million
bacteria by tomorrow.

We track each gene's frequency per year from 2000–2024, then fit:
- **Logistic model** (S-curve): L / (1 + e^{-k(t-t₀)}) — for genes still spreading
- **Linear model**: slope × year — for genes with steady trends
""")

with col2:
    # Summary metrics
    total = len(data)
    rapid    = sum(1 for r in data if r["growth_model"]["emergence_speed"] == "rapid")
    moderate = sum(1 for r in data if r["growth_model"]["emergence_speed"] == "moderate")
    slow_n   = sum(1 for r in data if r["growth_model"]["emergence_speed"] == "slow")
    declining = sum(1 for r in data if r["growth_model"]["emergence_speed"] == "declining")

    st.metric("Genes analysed", total)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🔴 Rapid (>2%/yr)", rapid)
        st.metric("🟠 Moderate", moderate)
    with col_b:
        st.metric("🟡 Slow rising", slow_n)
        st.metric("🟢 Declining", declining)

    if data:
        earliest = min(r["emergence_year"] for r in data if r["emergence_year"])
        latest   = max(r["emergence_year"] for r in data if r["emergence_year"])
        st.metric("Emergence years span", f"{earliest} – {latest}")

st.divider()

# ── Section 2: Speed classification overview ─────────────────────────────────
st.header("2. Spread speed classification")

st.markdown("""
We classify each gene by its linear slope (% of genomes carrying it, per year):
- 🔴 **Rapid**: > +2%/year — actively spreading, potential epidemic
- 🟠 **Moderate**: +0.5% to +2%/year — steady expansion
- 🟡 **Slow**: 0 to +0.5%/year — slight increase or stable
- 🟢 **Declining**: negative slope — being replaced or selected against
""")

df_all = pd.DataFrame([{
    "gene": r["gene"][:50],
    "emergence_year": r["emergence_year"] or 0,
    "current_frequency": r["current_frequency"],
    "peak_frequency": r["peak_frequency"],
    "slope": r["growth_model"]["slope"],
    "speed": r["growth_model"]["emergence_speed"],
    "model": r["growth_model"]["model"],
    "r2": r["growth_model"]["r2"],
    "overall_prevalence": r["overall_prevalence"],
} for r in data])

speed_colors = {
    "rapid":    "#e94560",
    "moderate": "#ffb86c",
    "slow":     "#8be9fd",
    "declining": "#50fa7b",
}

fig_overview = go.Figure()
for speed in ["rapid", "moderate", "slow", "declining"]:
    sub = df_all[df_all["speed"] == speed]
    if sub.empty:
        continue
    fig_overview.add_trace(go.Bar(
        name=speed.capitalize(),
        x=sub["gene"],
        y=sub["slope"],
        marker_color=speed_colors[speed],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Slope: %{y:+.2f}%/yr<br>"
            "<extra></extra>"
        ),
    ))

fig_overview.add_hline(y=0, line_color="#cdd6f4", line_width=1)
fig_overview.update_layout(
    barmode="overlay",
    xaxis_title="Resistance gene",
    yaxis_title="Spread rate (%/year)",
    height=400,
    margin=dict(t=20, b=120),
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    font_color="#cdd6f4", yaxis_gridcolor="#2d2d44",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    xaxis=dict(tickangle=45),
)
st.plotly_chart(fig_overview, use_container_width=True)
st.caption("Each bar is one gene. Height = how fast it's spreading (or declining) per year.")

st.divider()

# ── Section 3: Individual emergence curves ────────────────────────────────────
st.header("3. Individual gene epidemic curves")

st.markdown("""
Select any gene to see its year-by-year prevalence curve — the trajectory of how that
resistance mechanism spread through *K. pneumoniae* populations over 24 years.
""")

# Filter controls
col1, col2 = st.columns([2, 1])
with col1:
    gene_names = [r["gene"][:60] for r in data]
    selected_gene = st.selectbox(
        "Choose a gene to explore:",
        options=gene_names,
        index=0,
    )
with col2:
    speed_filter = st.multiselect(
        "Filter by speed:",
        options=["rapid", "moderate", "slow", "declining"],
        default=["rapid", "moderate", "slow", "declining"],
    )

selected = next((r for r in data if r["gene"][:60] == selected_gene), None)

if selected:
    yearly = selected["yearly"]
    df_yr  = pd.DataFrame(yearly)
    gm     = selected["growth_model"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Emerged", str(selected["emergence_year"] or "Before 2000"))
    with col2:
        st.metric("Peak", f"{selected['peak_frequency']:.1f}% in {selected['peak_year']}")
    with col3:
        st.metric("Current", f"{selected['current_frequency']:.1f}%")
    with col4:
        speed_emoji = {"rapid":"🔴","moderate":"🟠","slow":"🟡","declining":"🟢"}
        st.metric("Speed", f"{speed_emoji.get(gm['emergence_speed'],'')}{gm['emergence_speed']}")

    fig_gene = go.Figure()

    # Data points
    fig_gene.add_trace(go.Scatter(
        x=df_yr["year"],
        y=df_yr["frequency"],
        mode="lines+markers",
        name="Observed frequency",
        line=dict(color="#ffb86c", width=2),
        marker=dict(size=[max(5, n//15) for n in df_yr["n_total"]], color="#ffb86c"),
        hovertemplate="Year: %{x}<br>Frequency: %{y:.1f}%<br><extra></extra>",
    ))

    # Fit line
    if len(df_yr) >= 4:
        x_range = np.linspace(df_yr["year"].min(), df_yr["year"].max(), 100)
        if gm["model"] == "logistic" and "plateau" in gm and "growth_k" in gm and "midpoint" in gm:
            L  = gm["plateau"] / 100
            k  = gm["growth_k"]
            x0 = gm["midpoint"]
            y_fit = L / (1 + np.exp(-k * (x_range - x0)))
            label = f"Logistic fit (L={gm['plateau']:.0f}%, R²={gm['r2']:.3f})"
            fit_color = "#e94560"
        else:
            # Linear
            slope = gm["slope"] / 100
            intercept = df_yr["frequency"].mean() / 100 - slope * df_yr["year"].mean()
            y_fit = slope * x_range + intercept
            label = f"Linear fit ({gm['slope']:+.2f}%/yr, R²={gm['r2']:.3f})"
            fit_color = "#bd93f9"

        fig_gene.add_trace(go.Scatter(
            x=x_range, y=y_fit * 100,
            mode="lines", name=label,
            line=dict(color=fit_color, width=2, dash="dash"),
        ))

    # Emergence year marker
    if selected["emergence_year"]:
        fig_gene.add_vline(
            x=selected["emergence_year"],
            line_dash="dot", line_color="#6272a4",
            annotation_text=f"First detected ({selected['emergence_year']})",
            annotation_position="top right",
        )

    fig_gene.update_layout(
        title=f"{selected_gene[:60]} — epidemic curve",
        xaxis_title="Collection year",
        yaxis_title="% of genomes carrying this gene",
        yaxis=dict(range=[0, max(df_yr["frequency"]) * 1.4]),
        height=380, margin=dict(t=50, b=20),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_gene, use_container_width=True)
    st.caption("Marker size = number of genomes sampled in that year. Larger = more reliable estimate.")

    # Model details
    with st.expander("📊 Model details"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Model type", gm["model"].capitalize())
            st.metric("R² (goodness of fit)", f"{gm['r2']:.3f}")
        with col_b:
            st.metric("Slope", f"{gm['slope']:+.3f}%/yr")
            if "p_value" in gm:
                st.metric("p-value", f"{gm['p_value']:.4f}")
        with col_c:
            if gm["model"] == "logistic":
                st.metric("Plateau (L)", f"{gm.get('plateau', '?')}%")
                st.metric("Inflection year", str(gm.get("midpoint", "?")))
            st.metric("Overall prevalence", f"{selected['overall_prevalence']}%")

st.divider()

# ── Section 4: Multi-gene comparison ─────────────────────────────────────────
st.header("4. Compare multiple genes simultaneously")

st.markdown("Overlay multiple genes to see which ones are racing ahead — and which are fading out.")

# Filter by speed
filtered = [r for r in data if r["growth_model"]["emergence_speed"] in speed_filter]
if not filtered:
    st.info("Select at least one speed category above.")
else:
    fig_multi = go.Figure()
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    for i, gene_data in enumerate(filtered[:15]):  # cap at 15 for readability
        if not gene_data["yearly"]:
            continue
        df_g = pd.DataFrame(gene_data["yearly"])
        speed = gene_data["growth_model"]["emergence_speed"]
        color = speed_colors.get(speed, "#cdd6f4")
        fig_multi.add_trace(go.Scatter(
            x=df_g["year"],
            y=df_g["frequency"],
            mode="lines+markers",
            name=gene_data["gene"][:35],
            line=dict(color=palette[i % len(palette)], width=1.5),
            marker=dict(size=5),
            hovertemplate=(
                f"<b>{gene_data['gene'][:40]}</b><br>"
                "Year: %{x}<br>Freq: %{y:.1f}%<extra></extra>"
            ),
        ))

    fig_multi.update_layout(
        xaxis_title="Year",
        yaxis_title="% of genomes carrying gene",
        height=440, margin=dict(t=20, b=10),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
        legend=dict(orientation="v", x=1.01, y=1),
    )
    st.plotly_chart(fig_multi, use_container_width=True)
    st.caption(f"Showing {min(len(filtered), 15)} genes matching selected speed categories. Use the filter above to focus.")

st.divider()

# ── Section 5: Emergence year timeline ───────────────────────────────────────
st.header("5. When did each gene first appear?")

st.markdown("This timeline shows when each gene crossed the 1% frequency threshold — its 'emergence' into the bacterial population.")

df_timeline = df_all[df_all["emergence_year"] > 0].sort_values("emergence_year")
if not df_timeline.empty:
    colors_tl = [speed_colors.get(s, "#cdd6f4") for s in df_timeline["speed"]]
    fig_tl = go.Figure(go.Scatter(
        x=df_timeline["emergence_year"],
        y=df_timeline["current_frequency"],
        mode="markers+text",
        text=df_timeline["gene"].str[:25],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=df_timeline["overall_prevalence"].clip(5, 30),
            color=colors_tl,
            line=dict(width=1, color="#1e1e2e"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Emerged: %{x}<br>"
            "Current freq: %{y:.1f}%<extra></extra>"
        ),
    ))
    fig_tl.update_layout(
        xaxis_title="Year of emergence (first detected at ≥1%)",
        yaxis_title="Current frequency (%)",
        height=420, margin=dict(t=20, b=10),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
    )
    # Legend annotation
    for speed, color in speed_colors.items():
        fig_tl.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=10),
            name=speed.capitalize(), showlegend=True,
        ))
    fig_tl.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_tl, use_container_width=True)
    st.caption("Bubble size = overall prevalence across all genomes. Position = when it emerged vs how common it is today.")

st.divider()

# ── Section 6: Key findings ───────────────────────────────────────────────────
st.header("6. Key findings")

# Find interesting examples
top_moderate = [r for r in data if r["growth_model"]["emergence_speed"] == "moderate"]
top_declining = [r for r in data if r["growth_model"]["emergence_speed"] == "declining"]
fastest_rise = max(data, key=lambda x: x["growth_model"]["slope"]) if data else None
fastest_fall = min(data, key=lambda x: x["growth_model"]["slope"]) if data else None

col1, col2, col3 = st.columns(3)
with col1:
    if fastest_rise:
        st.success(
            f"**Fastest expanding gene:**\n\n"
            f"**{fastest_rise['gene'][:40]}**\n\n"
            f"Slope: {fastest_rise['growth_model']['slope']:+.2f}%/yr\n\n"
            f"Now in {fastest_rise['current_frequency']:.0f}% of strains"
        )
with col2:
    if fastest_fall:
        st.info(
            f"**Fastest declining gene:**\n\n"
            f"**{fastest_fall['gene'][:40]}**\n\n"
            f"Slope: {fastest_fall['growth_model']['slope']:+.2f}%/yr\n\n"
            f"May reflect annotation changes or replacement by newer mechanisms"
        )
with col3:
    logistic_genes = [r for r in data if r["growth_model"]["model"] == "logistic"]
    st.warning(
        f"**Logistic (S-curve) fit:**\n\n"
        f"**{len(logistic_genes)} genes** show S-shaped growth —\n\n"
        f"classic epidemic spread pattern,\n\n"
        f"suggesting saturation is approaching"
    )

st.markdown("""
**What this analysis reveals:**

| Pattern | Biological meaning | Clinical implication |
|---|---|---|
| **Moderate/rapid spread** | Gene is gaining selection advantage from antibiotic use | Emerging resistance threat to track |
| **Plateau in logistic fit** | Gene has saturated the population — most strains carry it | Consider it baseline resistance, not new threat |
| **Declining genes** | Being replaced by shorter/more efficient gene variants, or annotation change | Historical artifact, less clinically relevant |
| **Late emergence (post-2005)** | New resistance mechanisms entering the population | Highest surveillance priority |

**The epidemic curve framework** — borrowed from infectious disease epidemiology — gives us a language
to describe how *molecular* resistance spreads. Each new resistance gene is, in a sense, its own epidemic.
The tools for tracking one are the same tools for tracking the other.
""")
