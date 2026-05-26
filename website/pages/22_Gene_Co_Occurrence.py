"""
Page 22 — Gene Co-Occurrence Network
Which resistance genes always appear together — and why?
"""
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent.parent.parent
ART_DIR = ROOT / "artifacts"
PROC_DIR = ROOT / "data" / "processed"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css, page_info_expander
inject_mobile_css()
page_info_expander("""
**Gene co-occurrence** — Two resistance genes are "co-occurring" if they tend to be found together in the same isolate more often than expected by chance. This usually happens because they are physically linked on the same plasmid, transposon, or integron.

**Why it matters** — Co-occurring genes spread and disappear together. If *blaKPC* and *blaOXA* co-occur strongly, detecting one in a patient's isolate should raise suspicion of the other even if not directly tested. It also explains why certain treatment combinations fail simultaneously.

**φ (phi) coefficient** — A statistical measure of association between two binary variables (gene present/absent). Ranges from −1 to +1:
- φ close to **+1** → genes almost always co-occur (strong positive link, shown in blue)
- φ close to **0** → no association
- φ close to **−1** → genes are mutually exclusive (shown in red — rare)

**Network layout** — Nodes (circles) = resistance genes. Edges (lines) = strong co-occurrence links (|φ| above your threshold). Thicker lines = stronger association. The circular layout places all genes equidistantly; position has no biological meaning.

**Prevalence filter** — Genes present in <5% or >95% of genomes are excluded by default (too rare to analyse reliably, or too universal to be informative).

**Integron** — A mobile genetic element that can capture, stack, and co-express multiple resistance genes. A major reason why gene co-occurrence networks have dense clusters.
""")

st.title("🕸️ Gene Co-Occurrence Network")
st.markdown("*Which resistance genes always appear together — and why?*")
st.divider()

# ── Section 1: Background ──────────────────────────────────────────────────────
st.header("1. Why gene co-occurrence matters")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
Resistance genes don't travel alone. They cluster on **mobile genetic elements**:

- **Plasmids** — circular DNA molecules that bacteria share via conjugation.
  A single conjugative plasmid can carry 5–20 resistance genes simultaneously.

- **Integrons** — gene cassette arrays that accumulate resistance genes over time.
  *A. baumannii* integrons can carry dozens of genes.

- **Transposons** — "jumping genes" that relocate entire resistance cassettes between
  plasmids and chromosomes.

**Clinical consequence:** When one gene is present, its co-occurring partners are
almost certainly present too — even before susceptibility testing confirms it.

The **phi coefficient (φ)** measures pairwise co-occurrence:
- φ ≈ +1 → genes almost always found together (same plasmid)
- φ ≈ 0 → independent (different MGEs)
- φ < 0 → rarely found together (mutually exclusive resistance strategies)
""")
with col2:
    st.info("""
**Co-resistance network vs. Co-occurrence network**

The **Co-Resistance Network** page (🕸️ sidebar) shows correlations between
*antibiotic phenotypes* — which drugs fail together.

This page shows correlations between *resistance genes* — which actual
DNA sequences travel together on plasmids.

The two are related but not identical:
- One gene can confer resistance to multiple drugs (e.g., CTX-M breaks down
  cefepime AND ceftriaxone)
- One resistance phenotype can be caused by multiple genes
""")

st.divider()

# ── Load gene matrix ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading gene matrix…")
def load_gene_matrix():
    gm_path = PROC_DIR / "gene_matrix.csv"
    if not gm_path.exists():
        return None
    gm = pd.read_csv(gm_path, index_col=0)
    gm = gm.drop(columns=["__label__"], errors="ignore")
    return gm.fillna(0).astype(float)

gm = load_gene_matrix()

if gm is None:
    st.error("Gene matrix not found. Run `python src/generate_artifacts.py` first.")
    st.stop()

# ── Section 2: Gene prevalence ────────────────────────────────────────────────
st.header("2. Most prevalent resistance genes in K. pneumoniae")

gene_prevalence = gm.mean().sort_values(ascending=False)
n_genomes = len(gm)

st.markdown(f"*Based on {n_genomes:,} K. pneumoniae genomes. "
            f"Prevalence = fraction of genomes carrying each gene.*")

col_filter, col_min = st.columns([3, 1])
with col_filter:
    search_gene = st.text_input("Filter genes by name:", placeholder="e.g. bla, gyr, mcr, tet...")
with col_min:
    min_prev = st.slider("Min prevalence (%)", 0, 50, 5) / 100

filtered_prev = gene_prevalence[gene_prevalence >= min_prev]
if search_gene:
    filtered_prev = filtered_prev[filtered_prev.index.str.contains(search_gene, case=False)]

top_genes = filtered_prev.head(30)

if top_genes.empty:
    st.info("No genes match the current filters.")
else:
    bar_colors = [
        "#EF4444" if p >= 0.5 else
        "#F97316" if p >= 0.25 else
        "#6366F1" if p >= 0.1 else
        "#94A3B8"
        for p in top_genes.values
    ]
    fig_prev = go.Figure(go.Bar(
        x=top_genes.values * 100,
        y=top_genes.index.tolist(),
        orientation="h",
        marker_color=bar_colors,
        text=[f"{p*100:.1f}%" for p in top_genes.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Prevalence: %{x:.1f}%<br>"
                      f"~{int(top_genes.values[0]*n_genomes):,} genomes<extra></extra>",
    ))
    fig_prev.update_layout(
        xaxis=dict(title="Prevalence (%)", range=[0, min(top_genes.max()*130, 110)]),
        height=max(300, len(top_genes) * 24 + 60),
        margin=dict(t=10, b=10, r=60),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B", xaxis_gridcolor="#E2E8F0",
    )
    st.plotly_chart(fig_prev, use_container_width=True)
    st.caption("🔴 ≥50% genomes  🟠 25–50%  🟣 10–25%  ⬜ <10%")

st.divider()

# ── Section 3: Co-occurrence heatmap ─────────────────────────────────────────
st.header("3. Gene co-occurrence heatmap (phi coefficient)")

st.markdown("""
Select a set of genes to examine. The heatmap shows **phi coefficients** between
each pair — computed across all K. pneumoniae genomes.
""")

# Default: top 15 most prevalent genes with meaningful prevalence
default_candidates = gene_prevalence[
    (gene_prevalence >= 0.05) & (gene_prevalence <= 0.95)
].head(20).index.tolist()

selected_genes = st.multiselect(
    "Select genes for co-occurrence analysis:",
    options=gene_prevalence[gene_prevalence >= 0.01].index.tolist(),
    default=default_candidates[:15],
    max_selections=25,
    help="Select 5–20 genes. Phi coefficient computed pairwise.",
)

if len(selected_genes) < 3:
    st.info("Select at least 3 genes to compute the co-occurrence matrix.")
else:
    with st.spinner("Computing phi coefficients…"):
        X = gm[selected_genes].values.astype(float)
        n = len(X)

        # Vectorised phi coefficient computation
        # phi(X,Y) = (n11*n00 - n10*n01) / sqrt(n1. * n0. * n.1 * n.0)
        phi_mat = np.zeros((len(selected_genes), len(selected_genes)))
        for i in range(len(selected_genes)):
            for j in range(len(selected_genes)):
                if i == j:
                    phi_mat[i, j] = 1.0
                    continue
                xi, xj = X[:, i], X[:, j]
                n11 = ((xi == 1) & (xj == 1)).sum()
                n10 = ((xi == 1) & (xj == 0)).sum()
                n01 = ((xi == 0) & (xj == 1)).sum()
                n00 = ((xi == 0) & (xj == 0)).sum()
                denom = np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
                phi_mat[i, j] = (n11*n00 - n10*n01) / denom if denom > 0 else 0.0

    # Short gene labels
    gene_labels = [g[:30] + "…" if len(g) > 30 else g for g in selected_genes]

    text_mat = [[f"{phi_mat[i,j]:.2f}" if i != j else "1.00"
                 for j in range(len(selected_genes))]
                for i in range(len(selected_genes))]

    fig_heat = go.Figure(go.Heatmap(
        z=phi_mat,
        x=gene_labels, y=gene_labels,
        colorscale=[
            [0.0,  "#3B82F6"],   # strong negative: blue
            [0.35, "#DBEAFE"],
            [0.5,  "#F8FAFF"],   # near zero: white
            [0.65, "#FEE2E2"],
            [1.0,  "#EF4444"],   # strong positive: red
        ],
        zmid=0, zmin=-0.6, zmax=0.6,
        text=text_mat,
        texttemplate="%{text}",
        textfont=dict(size=10),
        showscale=True,
        colorbar=dict(title="φ", tickvals=[-0.6,-0.3,0,0.3,0.6]),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>φ = %{z:.3f}<extra></extra>",
    ))
    fig_heat.update_layout(
        height=max(420, len(selected_genes) * 28 + 80),
        margin=dict(t=20, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B",
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Top co-occurring pairs
    pairs = []
    for i in range(len(selected_genes)):
        for j in range(i+1, len(selected_genes)):
            pairs.append({
                "Gene A": selected_genes[i],
                "Gene B": selected_genes[j],
                "phi": round(phi_mat[i, j], 3),
                "abs_phi": abs(phi_mat[i, j]),
            })
    df_pairs = pd.DataFrame(pairs).sort_values("abs_phi", ascending=False).head(15)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**Strongest positive co-occurrence (same plasmid)**")
        top_pos = df_pairs[df_pairs["phi"] > 0].head(8)
        for _, r in top_pos.iterrows():
            color = "#EF4444" if r["phi"] > 0.4 else "#F97316" if r["phi"] > 0.2 else "#94A3B8"
            st.markdown(
                f"<div style='background:#FFF5F5; border-left:3px solid {color}; "
                f"padding:4px 8px; border-radius:4px; margin-bottom:4px; font-size:0.83rem;'>"
                f"<b>{r['Gene A'][:25]}</b> × <b>{r['Gene B'][:25]}</b> "
                f"<span style='color:{color}; float:right;'>φ={r['phi']:+.3f}</span></div>",
                unsafe_allow_html=True
            )
    with col_neg:
        st.markdown("**Strongest negative co-occurrence (mutually exclusive)**")
        top_neg = df_pairs[df_pairs["phi"] < 0].head(8)
        if top_neg.empty:
            st.info("No strong negative associations in this gene set.")
        for _, r in top_neg.iterrows():
            color = "#3B82F6"
            st.markdown(
                f"<div style='background:#EFF6FF; border-left:3px solid {color}; "
                f"padding:4px 8px; border-radius:4px; margin-bottom:4px; font-size:0.83rem;'>"
                f"<b>{r['Gene A'][:25]}</b> × <b>{r['Gene B'][:25]}</b> "
                f"<span style='color:{color}; float:right;'>φ={r['phi']:+.3f}</span></div>",
                unsafe_allow_html=True
            )

st.divider()

# ── Section 4: Network graph ──────────────────────────────────────────────────
st.header("4. Co-occurrence network graph")

st.markdown("Edges connect genes with |φ| above the threshold. "
            "Red = co-occurring, blue = mutually exclusive. Node size ∝ prevalence.")

if len(selected_genes) >= 3:
    phi_thresh = st.slider("Edge threshold |φ| ≥", 0.10, 0.60, 0.25, 0.05)

    # Build network positions (circular layout)
    n_nodes = len(selected_genes)
    angles  = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    # Add small random perturbation to avoid overlap
    np.random.seed(42)
    radii = 1 + np.random.uniform(-0.15, 0.15, n_nodes)
    node_x = radii * np.cos(angles)
    node_y = radii * np.sin(angles)

    prevalences = [float(gene_prevalence.get(g, 0)) for g in selected_genes]

    # Edge traces
    edge_traces = []
    for i in range(len(selected_genes)):
        for j in range(i + 1, len(selected_genes)):
            phi_val = phi_mat[i, j]
            if abs(phi_val) < phi_thresh:
                continue
            color = f"rgba(239,68,68,{min(abs(phi_val)*1.5, 0.8):.2f})" if phi_val > 0 \
                else f"rgba(59,130,246,{min(abs(phi_val)*1.5, 0.8):.2f})"
            width = max(1, abs(phi_val) * 8)
            edge_traces.append(go.Scatter(
                x=[node_x[i], node_x[j], None],
                y=[node_y[i], node_y[j], None],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Node trace
    node_sizes = [max(12, p * 60) for p in prevalences]
    node_labels = [g[:20] + "…" if len(g) > 20 else g for g in selected_genes]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9, color="#1E293B"),
        marker=dict(
            size=node_sizes,
            color=[p * 100 for p in prevalences],
            colorscale=[[0, "#C7D2FE"], [0.5, "#6366F1"], [1, "#312E81"]],
            cmin=0, cmax=80,
            showscale=True,
            colorbar=dict(title="Prevalence %", x=1.02),
            line=dict(width=1.5, color="#FFFFFF"),
        ),
        customdata=[[g, f"{p*100:.1f}%"] for g, p in zip(selected_genes, prevalences)],
        hovertemplate="<b>%{customdata[0]}</b><br>Prevalence: %{customdata[1]}<extra></extra>",
        showlegend=False,
    )

    fig_net = go.Figure(data=edge_traces + [node_trace])
    fig_net.update_layout(
        height=560,
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5]),
        margin=dict(t=20, b=20, l=20, r=60),
        plot_bgcolor="#FAFBFF", paper_bgcolor="#FFFFFF",
        hovermode="closest",
    )

    n_edges = sum(1 for i in range(len(selected_genes))
                  for j in range(i+1, len(selected_genes))
                  if abs(phi_mat[i,j]) >= phi_thresh)
    st.plotly_chart(fig_net, use_container_width=True)
    st.caption(f"Showing {n_edges} edges with |φ| ≥ {phi_thresh:.2f}. "
               f"🔴 Red = co-occurring  🔵 Blue = mutually exclusive  "
               f"Node colour = prevalence.")

st.divider()

# ── Section 5: Gene lookup ────────────────────────────────────────────────────
st.header("5. Gene lookup — what travels with this gene?")

lookup_gene = st.selectbox(
    "Select a gene to find its co-occurring partners:",
    options=gene_prevalence[gene_prevalence >= 0.01].index.tolist(),
    key="lookup_gene",
)

if lookup_gene in gm.columns:
    xi = gm[lookup_gene].values.astype(float)
    carriers = int(xi.sum())
    prev_pct = carriers / n_genomes * 100

    st.markdown(f"**{lookup_gene}** present in **{carriers:,}** / {n_genomes:,} genomes "
                f"({prev_pct:.1f}% prevalence)")

    # Compute phi with all other genes (vectorised)
    others = [g for g in gm.columns if g != lookup_gene and gene_prevalence.get(g, 0) >= 0.01]
    phis = []
    for g in others:
        xj = gm[g].values.astype(float)
        n11 = ((xi==1)&(xj==1)).sum(); n10 = ((xi==1)&(xj==0)).sum()
        n01 = ((xi==0)&(xj==1)).sum(); n00 = ((xi==0)&(xj==0)).sum()
        denom = np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
        phi_val = (n11*n00 - n10*n01)/denom if denom > 0 else 0.0
        if abs(phi_val) >= 0.15:
            phis.append({"gene": g, "phi": round(phi_val, 3),
                         "prevalence_pct": round(gene_prevalence[g]*100, 1),
                         "n_co_occur": int(n11)})

    df_lookup = pd.DataFrame(phis).sort_values("phi", ascending=False)

    if df_lookup.empty:
        st.info("No strong co-occurring partners found (|φ| < 0.15).")
    else:
        col_co, col_excl = st.columns(2)
        with col_co:
            st.markdown("**Co-occurring partners (φ > 0)**")
            pos = df_lookup[df_lookup["phi"] > 0].head(12)
            if not pos.empty:
                def _phi_color(v):
                    try:
                        f = float(v)
                        if f >= 0.5: return "background-color:#FEE2E2;color:#991B1B;font-weight:600"
                        elif f >= 0.3: return "background-color:#FEF9C3;color:#713F12;font-weight:600"
                        return ""
                    except: return ""
                st.dataframe(
                    pos[["gene","phi","prevalence_pct","n_co_occur"]]
                    .rename(columns={"gene":"Gene","phi":"φ","prevalence_pct":"Prev %","n_co_occur":"Co-occur"})
                    .style.map(_phi_color, subset=["φ"])
                    .format({"φ":"{:+.3f}","Prev %":"{:.1f}%","Co-occur":"{:,}"}),
                    use_container_width=True, hide_index=True
                )
        with col_excl:
            st.markdown("**Mutually exclusive partners (φ < 0)**")
            neg = df_lookup[df_lookup["phi"] < 0].sort_values("phi").head(8)
            if neg.empty:
                st.info("No strong negative associations.")
            else:
                st.dataframe(
                    neg[["gene","phi","prevalence_pct","n_co_occur"]]
                    .rename(columns={"gene":"Gene","phi":"φ","prevalence_pct":"Prev %","n_co_occur":"Co-occur"})
                    .style.format({"φ":"{:+.3f}","Prev %":"{:.1f}%","Co-occur":"{:,}"}),
                    use_container_width=True, hide_index=True
                )
