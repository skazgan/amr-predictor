"""
Page 2 — The Data: dataset statistics and class balance
"""
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

ROOT    = Path(__file__).parent.parent.parent
ART_DIR = ROOT / "artifacts"


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
st.set_page_config(page_title="The Data", page_icon="📊", layout="wide")
inject_mobile_css()
st.title("📊 The Data")
st.markdown("*Where it comes from, how much there is, and how we prepared it.*")
st.divider()

stats   = json.loads((ART_DIR / "dataset_stats.json").read_text())
summary = json.loads((ART_DIR / "summary.json").read_text())
df_stats = pd.DataFrame(stats)

# ── Section 1: Data source ────────────────────────────────────────────────────
st.header("1. Data source — BV-BRC")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
All data comes from the **Bacterial and Viral Bioinformatics Resource Center (BV-BRC)**,
the world's largest free database of bacterial genomes with antibiotic susceptibility labels.

**What BV-BRC provides for each genome:**
- The full DNA sequence as a FASTA file (~4 MB per genome)
- Minimum inhibitory concentration (MIC) measurements
- Binary R/S phenotype labels per antibiotic
- Annotated resistance gene presence/absence

**How we downloaded the data:**

```python
# Query BV-BRC API for K. pneumoniae genomes
GET /api/genome_amr/
    ?eq(taxon_id, 573)           # 573 = K. pneumoniae
    &eq(antibiotic, meropenem)
    &select(genome_id, resistant_phenotype)
    &limit(10000)
```

This returns a list of genome IDs with their R/S labels.
We then download each genome's FASTA file and gene annotations separately.
""")

with col2:
    total_genomes = df_stats["total"].sum()
    total_r = df_stats["n_resistant"].sum()
    total_s = df_stats["n_susceptible"].sum()

    st.metric("Total labeled genomes", f"{total_genomes:,}")
    st.metric("Resistant labels", f"{total_r:,}", f"{total_r/total_genomes*100:.0f}%")
    st.metric("Susceptible labels", f"{total_s:,}", f"{total_s/total_genomes*100:.0f}%")
    st.metric("Unique genome IDs", "~17,000")
    st.info("Many genomes appear in multiple antibiotics — FASTAs are shared, only labels differ.")

st.divider()

# ── Section 2: Per-antibiotic counts ──────────────────────────────────────────
st.header("2. Genomes per antibiotic")

fig = go.Figure()
df_sorted = df_stats.sort_values("total", ascending=True)
fig.add_trace(go.Bar(
    name="Resistant", x=df_sorted["n_resistant"], y=df_sorted["antibiotic"],
    orientation="h", marker_color="#e94560",
    text=df_sorted["n_resistant"], textposition="inside",
))
fig.add_trace(go.Bar(
    name="Susceptible", x=df_sorted["n_susceptible"], y=df_sorted["antibiotic"],
    orientation="h", marker_color="#50fa7b",
    text=df_sorted["n_susceptible"], textposition="inside",
))
fig.update_layout(
    barmode="stack",
    xaxis_title="Number of genomes",
    height=350, margin=dict(l=10, r=20, t=20, b=40),
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    font_color="#1E293B", xaxis_gridcolor="#2d2d44",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 3: Class balance ──────────────────────────────────────────────────
st.header("3. Class balance")

col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("""
**Why balance matters:**

If 90% of samples are Resistant, a model that always predicts "Resistant" gets 90% accuracy
— but learns nothing.

We **balanced** each antibiotic dataset to equal R/S counts (up to 2,000 each)
so the model must actually learn the difference.

**Balancing method:**  Random sampling with `random_state=42` for reproducibility.
""")
with col2:
    fig2 = go.Figure()
    for row in df_stats.to_dict("records"):
        total = row["n_resistant"] + row["n_susceptible"]
        pct_r = row["n_resistant"] / total * 100
        fig2.add_trace(go.Bar(
            name=row["antibiotic"],
            x=[row["antibiotic"]],
            y=[pct_r],
            marker_color="#e94560" if pct_r > 55 else "#50fa7b" if pct_r < 45 else "#8be9fd",
            showlegend=False,
            text=[f"{pct_r:.0f}%R"],
            textposition="outside",
        ))
    fig2.add_hline(y=50, line_dash="dash", line_color="#64748B",
                   annotation_text="Perfect balance")
    fig2.update_layout(
        yaxis=dict(title="% Resistant", range=[0, 70]),
        height=280, margin=dict(t=20, b=10),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#1E293B", yaxis_gridcolor="#2d2d44",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Section 4: Data pipeline ──────────────────────────────────────────────────
st.header("4. Full data pipeline")

st.markdown("""
```
BV-BRC API
    │
    ├─ genome_amr endpoint  → metadata.csv  (genome_id, R/S label)  per antibiotic
    │
    └─ genome_sequence endpoint → data/raw/<genome_id>.fasta         shared

data/raw/  (~17,000 FASTA files, ~4 MB each, ~68 GB total)
    │
    ├─ features.py    → 6-mer counts  → data/processed/X.csv     (genomes × 4096)
    │
    └─ gene_features.py → resistance gene presence/absence
                       → data/processed/gene_matrix.csv  (genomes × ~1600 genes)

Per antibiotic:
    metadata.csv  ──┐
    X.csv (k-mers)  ├─ SelectKBest(256 k-mers) + all genes → train → models/<ab>.pkl
    gene_matrix.csv ┘
```
""")

st.divider()

# ── Section 5: Train / test split ────────────────────────────────────────────
st.header("5. How we evaluate — train/test split + cross-validation")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Hold-out test set (20%)**

Before any training, we reserve 20% of each antibiotic's dataset as a
**test set** — the model never sees these genomes during training.
Final AUC numbers are measured on this set.

**5-fold cross-validation (on the 80% train set)**

The training data is split into 5 folds. The model trains on 4,
validates on 1, and rotates — giving 5 independent AUC estimates.
This detects overfitting early.
""")

with col2:
    # Visual of CV
    fig3 = go.Figure()
    colors_cv = ["#e94560", "#e94560", "#e94560", "#e94560", "#ffb86c"]
    labels_cv = ["Train", "Train", "Train", "Train", "Validate"]
    for i in range(5):
        fold_colors = ["#ffb86c" if j == i else "#2d2d44" for j in range(5)]
        fold_labels = ["Validate" if j == i else "Train" for j in range(5)]
    fig3.add_trace(go.Heatmap(
        z=[[1 if j != i else 2 for j in range(5)] for i in range(5)],
        colorscale=[[0, "#2d2d44"], [0.5, "#2d2d44"], [0.5, "#ffb86c"], [1, "#ffb86c"]],
        showscale=False,
        xgap=3, ygap=3,
    ))
    fig3.update_layout(
        title="5-Fold Cross-Validation (each row = one fold)",
        xaxis=dict(tickvals=list(range(5)), ticktext=[f"Fold {i+1}" for i in range(5)]),
        yaxis=dict(tickvals=list(range(5)), ticktext=[f"Round {i+1}" for i in range(5)]),
        height=240, margin=dict(t=40, b=10),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#1E293B",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("🟧 = Validation fold  |  ⬛ = Training folds")
