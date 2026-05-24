"""
Page 1 — The Biology: plain-English explanation with visuals
"""
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="The Biology", page_icon="🦠", layout="wide")

st.title("🦠 The Biology")
st.markdown("*Everything you need to understand before the code makes sense.*")
st.divider()

# ── Section 1: Bacteria & DNA ─────────────────────────────────────────────────
st.header("1. Bacteria and DNA")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
Bacteria are single-celled organisms. Each one carries a **genome** — a long string
of DNA written in just four letters: **A, T, G, C**.

| Organism | Genome size |
|---|---|
| *Klebsiella pneumoniae* (our target) | ~5.5 million letters |
| Human | ~3.2 billion letters |
| Common cold virus | ~30,000 letters |

When we **sequence** a bacterium, we get a text file (FASTA format) that looks like this:
```
>strain_001
ATGCTTACGGATCGATCGTAGCTAGCTAGCTAGCTAGCTA...
```
That string of letters **is the raw input** to our machine learning model.
""")

with col2:
    # Simple donut to show genome composition
    fig = go.Figure(go.Pie(
        labels=["A", "T", "G", "C"],
        values=[25.1, 25.2, 24.9, 24.8],
        hole=0.55,
        marker_colors=["#ff79c6", "#50fa7b", "#8be9fd", "#ffb86c"],
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Typical K. pneumoniae base composition",
        height=280, margin=dict(t=40, b=10),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 2: Antibiotics ────────────────────────────────────────────────────
st.header("2. What do antibiotics do?")

st.markdown("""
Antibiotics are chemicals that kill bacteria or stop them from reproducing.
Each drug targets a **specific bacterial mechanism**:
""")

mechanisms = {
    "Ciprofloxacin (Fluoroquinolone)":          "Blocks DNA replication enzymes (gyrA / parC)",
    "Meropenem (Carbapenem)":                   "Destroys the bacterial cell wall",
    "Gentamicin (Aminoglycoside)":              "Jams the bacterial ribosome (protein factory)",
    "Tetracycline":                             "Blocks amino acid delivery to the ribosome",
    "Trimethoprim/Sulfamethoxazole (Folate)":   "Shuts down the folate synthesis pathway",
    "Cefepime (Cephalosporin)":                 "Prevents cell-wall cross-linking",
}
for drug, mech in mechanisms.items():
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown(f"**{drug}**")
    with col_b:
        st.markdown(f"🎯 {mech}")

st.divider()

# ── Section 3: Resistance ─────────────────────────────────────────────────────
st.header("3. How does resistance develop?")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
#### Mutation in a target gene
The antibiotic's "docking site" changes shape so the drug can no longer bind.

**Example:** A single mutation in *gyrA* (one letter changed out of 5.5 million)
makes ciprofloxacin unable to block DNA replication.
""")
    st.info("🧬 One mutation = entire drug class rendered useless")

with col2:
    st.markdown("""
#### Acquisition of a resistance gene
The bacterium picks up a gene from another bacterium (via a process called
horizontal gene transfer) that either **breaks down** the drug or **pumps it out**.

**Example:** The *blaKPC* gene produces an enzyme that destroys carbapenems
like meropenem — our last-resort antibiotic.
""")
    st.warning("⚠️ Resistance genes can spread between patients in hospitals")

st.divider()

# ── Section 4: Why K. pneumoniae ──────────────────────────────────────────────
st.header("4. Why Klebsiella pneumoniae?")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("WHO Priority", "CRITICAL", help="Highest WHO priority pathogen for new antibiotic development")
with col2:
    st.metric("Multi-drug resistant strains", "~30%", help="Proportion of clinical isolates resistant to 3+ drug classes")
with col3:
    st.metric("Mortality if carbapenem-resistant", "~50%", help="Estimated mortality for carbapenem-resistant K. pneumoniae infections")

st.markdown("""
*K. pneumoniae* causes pneumonia, bloodstream infections, and UTIs — especially
in hospitalised patients. It is one of the fastest-evolving resistant pathogens,
which is why **rapid genomic resistance prediction matters**.

Traditional lab tests (antibiotic susceptibility testing) take **24–72 hours**.
A genomic ML model can give a preliminary answer in **seconds**.
""")

st.divider()

# ── Section 5: The prediction problem ────────────────────────────────────────
st.header("5. The machine learning framing")

st.markdown("""
We turn this into a **binary classification problem**:

```
Input:   DNA sequence of K. pneumoniae strain  (5.5 million letters)
Output:  Resistant (R) or Susceptible (S) to antibiotic X
```

Trained separately for each antibiotic — because resistance mechanisms are
completely different for each drug class.

One model **cannot** generalise across drug classes. A strain that's resistant
to ciprofloxacin (via *gyrA* mutation) may be fully susceptible to meropenem.
""")

# Visual: mapping from genome to label
fig = go.Figure()
fig.add_annotation(x=0.1, y=0.5, text="🧬 Genome<br>ATGCTTACGG...", showarrow=False,
    font=dict(size=14, color="#cdd6f4"),
    bgcolor="#2d2d44", borderpad=10, bordercolor="#6272a4")
fig.add_annotation(x=0.5, y=0.5, text="⚙️ ML Model<br>(XGBoost +<br>k-mer & gene features)", showarrow=False,
    font=dict(size=13, color="#cdd6f4"),
    bgcolor="#2d2d44", borderpad=10, bordercolor="#6272a4")
fig.add_annotation(x=0.9, y=0.5, text="✗ Resistant<br>or<br>✓ Susceptible", showarrow=False,
    font=dict(size=14, color="#cdd6f4"),
    bgcolor="#2d2d44", borderpad=10, bordercolor="#6272a4")
for x0, x1 in [(0.22, 0.38), (0.62, 0.78)]:
    fig.add_shape(type="line", x0=x0, y0=0.5, x1=x1, y1=0.5,
                  line=dict(color="#e94560", width=2),
                  xref="paper", yref="paper")
fig.update_layout(height=160, xaxis_visible=False, yaxis_visible=False,
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e", margin=dict(t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
