"""
Page 3 — Features: k-mers and resistance genes explained
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
from utils import inject_mobile_css
inject_mobile_css()
st.title("🔬 Feature Engineering")
st.markdown("*How we turn 5.5 million DNA letters into numbers a model can learn from.*")
st.divider()

# ── Section 1: The problem ────────────────────────────────────────────────────
st.header("1. The problem: you can't feed raw DNA to a model")

st.markdown("""
A genome is a string like `ATGCTTACGGATCGATCG...` — 5.5 million characters long.
Machine learning models expect **fixed-length numeric vectors**.

We use **two complementary feature types** that capture different aspects of the genome:
""")

col1, col2 = st.columns(2)
with col1:
    st.info("""
**Feature Type 1: k-mers** 🧮

Count every 6-letter substring in the genome.
4⁶ = **4,096 possible 6-mers**.

Fast, requires no prior knowledge.
Captures sequence-level patterns anywhere in the genome.
""")
with col2:
    st.success("""
**Feature Type 2: Resistance Genes** 🧬

Binary presence/absence of ~1,600 known resistance genes
(from the CARD database, queried via BV-BRC).

Biologically meaningful.
Directly encodes known resistance mechanisms.
""")

st.divider()

# ── Section 2: k-mers ────────────────────────────────────────────────────────
st.header("2. k-mers — counting DNA substrings")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
A **k-mer** is a fixed-length substring of a DNA sequence.

For k=6, we slide a window of width 6 along the genome and count every substring we see:

```
Sequence:  A T G C T A G C ...
           ┌─────────┐
           │ ATGCTA  │  count: 3
           └─────────┘
             ┌─────────┐
             │ TGCTAG  │  count: 1
             └─────────┘
               ┌─────────┐
               │ GCTAGC  │  count: 7
               └─────────┘
```

Each genome becomes a **vector of 4,096 counts**:
```
genome_1: [AAAAAA=12, AAAAAC=3, AAAAAG=7, ..., TTTTTT=9]
genome_2: [AAAAAA=9,  AAAAAC=5, AAAAAG=11, ..., TTTTTT=4]
```
This is analogous to a **bag-of-words** representation in NLP.
""")

with col2:
    # Show example k-mer counts as a bar chart
    example_kmers = ["ATGCTA", "GCTTAC", "CTTACG", "TTACGG", "TACGGA", "ACGGAT"]
    example_counts = [412, 388, 401, 395, 378, 410]
    fig = go.Figure(go.Bar(
        x=example_counts, y=example_kmers, orientation="h",
        marker_color="#8be9fd",
    ))
    fig.update_layout(
        title="Example: 6-mer counts in one genome",
        xaxis_title="Count", height=280, margin=dict(t=40, b=10, l=10),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#1E293B", xaxis_gridcolor="#2d2d44",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Why k=6?** It's the sweet spot:
- k=4 → only 256 features, too little information
- k=6 → 4,096 features, captures meaningful patterns
- k=8 → 65,536 features, memory-heavy with diminishing returns

After counting all 4,096 k-mers, we apply **feature selection** (SelectKBest with mutual
information) to keep only the **top 256 k-mers** most correlated with resistance for each
antibiotic.
""")

st.divider()

# ── Section 3: Resistance genes ──────────────────────────────────────────────
st.header("3. Resistance genes — the biological signal")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
The **CARD (Comprehensive Antibiotic Resistance Database)** catalogues thousands of genes
known to confer resistance. BV-BRC annotates each genome against this database.

For each genome we get a **binary vector**:
```
genome_1: [gyrA=1, CTX-M-15=1, blaKPC=0, aac6=1, ...]
genome_2: [gyrA=0, CTX-M-15=0, blaKPC=1, aac6=0, ...]
```
1 = gene detected, 0 = absent

**Why this works so well:**
The relationship is often direct — if *gyrA* is mutated, the strain
IS resistant to ciprofloxacin. The model doesn't need to "discover" this;
it just needs to learn the weight of the evidence.
""")

with col2:
    # Show top genes for meropenem from artifacts
    fi_path = ART_DIR / "fi_meropenem.json"
    if fi_path.exists():
        fi = json.loads(fi_path.read_text())
        gene_fi = [(r["feature"].replace("gene__",""), r["importance"])
                   for r in fi if r["feature"].startswith("gene__")][:10]
        if gene_fi:
            genes, imps = zip(*gene_fi)
            fig2 = go.Figure(go.Bar(
                x=list(imps), y=list(genes), orientation="h",
                marker_color="#50fa7b",
            ))
            fig2.update_layout(
                title="Top resistance genes — Meropenem model",
                xaxis_title="Importance", height=320,
                margin=dict(t=40, b=10, l=10),
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="#1E293B", xaxis_gridcolor="#2d2d44",
            )
            st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Section 4: Combining both ─────────────────────────────────────────────────
st.header("4. Combining k-mers + genes")

st.markdown("""
The final feature matrix for each antibiotic stacks both types side by side:

```
         ◄─── 256 top k-mers ───►  ◄────── ~1,600 resistance genes ──────►
genome_1: [12, 3, 7, ... 9,  1, 0,  1,  0,  1, ...]
genome_2: [9,  5, 11, ... 4,  0, 1,  0,  1,  0, ...]
          ↑ k-mer counts (normalised)     ↑ gene presence (0/1)
```

**Why use both?**
- Genes alone miss unknown variants — novel mutations not in CARD yet
- k-mers alone are noisy — the signal is buried in 5.5M letters
- Together, genes provide the strong known signal; k-mers fill the gaps

**AUC improvement from combining:**
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("k-mers only", "0.66 AUC", "baseline")
with col2:
    st.metric("Genes only", "0.82 AUC", "+0.16 vs k-mers")
with col3:
    st.metric("k-mers + genes", "0.83 AUC", "+0.01 vs genes alone")

st.markdown("""
The combined approach outperforms either alone — genes provide the primary signal,
k-mers add a small but consistent boost from sequence patterns not yet captured in gene databases.
""")

st.divider()

# ── Section 5: Interactive k-mer explorer ────────────────────────────────────
st.header("5. Try it — k-mer counter")

st.markdown("Enter a DNA sequence to see its 6-mer counts:")
seq_input = st.text_input(
    "DNA sequence (A/T/G/C only):",
    value="ATGCTTACGGATCGATCGTAGCTAGCTA",
    max_chars=200,
)

if seq_input:
    seq = seq_input.upper().replace(" ", "")
    invalid = set(seq) - set("ATGC")
    if invalid:
        st.error(f"Invalid characters: {invalid}. Only A, T, G, C allowed.")
    elif len(seq) < 6:
        st.warning("Sequence must be at least 6 characters long.")
    else:
        from collections import Counter
        kmers = Counter(seq[i:i+6] for i in range(len(seq)-5))
        top_k = kmers.most_common(15)
        kmer_labels, kmer_vals = zip(*top_k)
        fig3 = go.Figure(go.Bar(
            x=list(kmer_labels), y=list(kmer_vals),
            marker_color="#ff79c6",
            text=list(kmer_vals), textposition="outside",
        ))
        fig3.update_layout(
            title=f"Top 15 6-mers in your sequence ({len(seq)} bp, {len(kmers)} unique k-mers)",
            yaxis_title="Count", height=300,
            margin=dict(t=40, b=10),
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="#1E293B", yaxis_gridcolor="#2d2d44",
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"A full genome has ~5,500,000 bp → ~5,499,995 6-mers across 4,096 unique combinations.")
