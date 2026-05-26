"""
Page 21 — Strain Comparison: compare two K. pneumoniae genome IDs side-by-side
"""
import sys
import pickle
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np

ROOT      = Path(__file__).parent.parent.parent
ART_DIR   = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import inject_mobile_css
inject_mobile_css()

st.title("⚖️ Strain Comparison")
st.markdown("*Compare two K. pneumoniae genomes side-by-side: resistance profile, gene differences, and clinical implications.*")
st.divider()

ANTIBIOTICS = [
    "ciprofloxacin", "meropenem", "gentamicin", "tetracycline",
    "trimethoprim/sulfamethoxazole", "cefepime",
    "amikacin", "imipenem", "piperacillin/tazobactam", "levofloxacin",
]
DRUG_CLASS = {
    "ciprofloxacin":                "Fluoroquinolone",
    "meropenem":                    "Carbapenem",
    "gentamicin":                   "Aminoglycoside",
    "tetracycline":                 "Tetracycline",
    "trimethoprim/sulfamethoxazole":"Folate inhibitor",
    "cefepime":                     "Cephalosporin",
    "amikacin":                     "Aminoglycoside",
    "imipenem":                     "Carbapenem",
    "piperacillin/tazobactam":      "Beta-lactam/BLI",
    "levofloxacin":                 "Fluoroquinolone",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for ab in ANTIBIOTICS:
        safe = ab.replace("/", "_").replace(" ", "_")
        p = MODEL_DIR / f"{safe}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                models[ab] = pickle.load(f)
    return models

@st.cache_resource(show_spinner=False)
def load_kmer_matrix():
    p = PROC_DIR / "X.csv"
    if not p.exists():
        return pd.DataFrame()
    X = pd.read_csv(p, index_col=0)
    X.index = X.index.astype(str)
    return X

@st.cache_resource(show_spinner=False)
def load_gene_matrix():
    p = PROC_DIR / "gene_matrix.csv"
    if not p.exists():
        return pd.DataFrame()
    g = pd.read_csv(p, index_col=0)
    g.index = g.index.astype(str)
    return g.drop(columns=["__label__"], errors="ignore")

def fetch_genes_api(genome_id: str) -> set:
    url = (f"https://www.bv-brc.org/api/genome_amr/"
           f"?eq(genome_id,{genome_id})&select(gene)&limit(500)")
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        data = r.json()
        return {d.get("gene", "") for d in data if d.get("gene")}
    except Exception:
        return set()

def fetch_genome_metadata(genome_id: str) -> dict:
    url = (f"https://www.bv-brc.org/api/genome/"
           f"?eq(genome_id,{genome_id})&select(genome_name,isolation_country,host_common_name,"
           f"collection_year,mlst,genome_status,contigs)&limit(1)")
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        data = r.json()
        return data[0] if data else {}
    except Exception:
        return {}

def predict_single(genome_id: str, models: dict,
                   X_kmer: pd.DataFrame, X_gene: pd.DataFrame,
                   threshold: float = 70.0) -> pd.DataFrame:
    results = []
    for ab, bundle in models.items():
        features = bundle["features"]
        kmer_cols = [c for c in features if not c.startswith("gene__")]
        gene_cols  = [c for c in features if c.startswith("gene__")]
        gene_bare  = [c.replace("gene__", "") for c in gene_cols]

        kmer_row = (X_kmer.loc[genome_id, kmer_cols]
                    if (genome_id in X_kmer.index and kmer_cols)
                    else pd.Series(0.0, index=kmer_cols))
        if genome_id in X_gene.index and gene_bare:
            gene_row = X_gene.loc[genome_id, [c for c in gene_bare if c in X_gene.columns]]
            gene_row = gene_row.reindex(gene_bare, fill_value=0).rename(lambda x: "gene__" + x)
        else:
            gene_row = pd.Series(0, index=gene_cols)

        x = pd.concat([kmer_row, gene_row]).to_frame().T
        prob_r     = bundle["model"].predict_proba(x)[0][1]
        confidence = max(prob_r, 1 - prob_r) * 100
        verdict = ("Uncertain" if confidence < threshold else
                   "Resistant" if prob_r > 0.5 else "Susceptible")
        results.append({
            "antibiotic":  ab,
            "drug_class":  DRUG_CLASS[ab],
            "prob_r":      round(prob_r * 100, 1),
            "verdict":     verdict,
            "confidence":  round(confidence, 1),
        })
    return pd.DataFrame(results)

# Colour helpers
def verdict_color(v):
    return {"Resistant": "#EF4444", "Susceptible": "#10B981", "Uncertain": "#F59E0B"}.get(v, "#94A3B8")

def verdict_bg(v):
    return {"Resistant": "#FEE2E2", "Susceptible": "#DCFCE7", "Uncertain": "#FEF9C3"}.get(v, "#F1F5F9")

def verdict_tc(v):
    return {"Resistant": "#991B1B", "Susceptible": "#15803D", "Uncertain": "#92400E"}.get(v, "#475569")

# ── Section 1: How strain comparison works ────────────────────────────────────
st.header("1. Why compare strains?")
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
Strain comparison is essential in clinical microbiology to:

- **Detect outbreak clusters** — two patients with near-identical strains may share an infection source
- **Track resistance evolution** — compare a patient's initial isolate with a relapse isolate
- **Inform treatment decisions** — if two strains differ in resistance to key antibiotics, treatment plans diverge
- **Assess therapeutic failure** — a strain collected after treatment failure often has acquired new resistance

**What this tool compares:**
- Predicted resistance to all 10 antibiotics (probability + verdict)
- Concordance/discordance table — where the two strains differ
- Resistance genes present in A but not B (and vice versa)
- Metadata: country, year, MLST type
""")
with col2:
    st.info("""
**Examples of useful comparisons:**

🧫 **Pre/post-treatment** — same patient, before and after carbapenem therapy

🏥 **Hospital cluster** — two ICU patients, same week → check if same clone

🌍 **Travel-imported vs local** — import from high-resistance region vs domestic strain

📊 **Reference vs query** — compare an unknown strain against a known outbreak strain (ST258, ST11)
""")

st.divider()

# ── Section 2: Input ──────────────────────────────────────────────────────────
st.header("2. Enter two genome IDs")

example_pairs = {
    "MDR vs susceptible reference": ("573.12783", "573.65923"),
    "Two carbapenem-resistant strains": ("573.52406", "573.12783"),
    "KPC vs OXA producer": ("573.52406", "573.49554"),
}

col_ex, col_thresh = st.columns([3, 1])
with col_ex:
    example_sel = st.selectbox("Load example pair:", ["— custom —"] + list(example_pairs.keys()))
with col_thresh:
    threshold = st.slider("Confidence threshold", 50, 95, 70, 5)

default_a, default_b = "", ""
if example_sel != "— custom —":
    default_a, default_b = example_pairs[example_sel]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### 🔵 Genome A")
    genome_a = st.text_input("Genome ID (A):", value=default_a, placeholder="e.g. 573.12783", key="gid_a")
with col_b:
    st.markdown("#### 🟠 Genome B")
    genome_b = st.text_input("Genome ID (B):", value=default_b, placeholder="e.g. 573.65923", key="gid_b")

run = st.button("⚖️ Compare Strains", type="primary",
                use_container_width=True,
                disabled=not (genome_a.strip() and genome_b.strip()))

st.divider()

# ── Section 3: Comparison results ────────────────────────────────────────────
if run and genome_a.strip() and genome_b.strip():
    gid_a = genome_a.strip()
    gid_b = genome_b.strip()

    if gid_a == gid_b:
        st.warning("The two genome IDs are identical — please enter different genomes.")
        st.stop()

    with st.spinner("Loading models and data..."):
        models   = load_models()
        X_kmer   = load_kmer_matrix()
        X_gene   = load_gene_matrix()

    if not models:
        st.error("No models found. Run `python src/generate_artifacts.py` first.")
        st.stop()

    with st.spinner(f"Predicting resistance for {gid_a} and {gid_b}..."):
        df_a = predict_single(gid_a, models, X_kmer, X_gene, threshold)
        df_b = predict_single(gid_b, models, X_kmer, X_gene, threshold)

    # Metadata (non-blocking)
    with st.spinner("Fetching genome metadata..."):
        meta_a = fetch_genome_metadata(gid_a)
        meta_b = fetch_genome_metadata(gid_b)
        genes_a = (fetch_genes_api(gid_a)
                   if gid_a not in X_gene.index else
                   set(X_gene.columns[X_gene.loc[gid_a] > 0]))
        genes_b = (fetch_genes_api(gid_b)
                   if gid_b not in X_gene.index else
                   set(X_gene.columns[X_gene.loc[gid_b] > 0]))

    # ── Metadata cards ────────────────────────────────────────────────────────
    st.subheader("Genome metadata")
    meta_col_a, meta_col_b = st.columns(2)

    def _meta_card(meta: dict, label: str, color: str):
        name    = meta.get("genome_name", "Unknown")
        country = meta.get("isolation_country", "Unknown")
        year    = meta.get("collection_year", "Unknown")
        mlst    = meta.get("mlst", "—")
        host    = meta.get("host_common_name", "Unknown")
        return f"""
<div style='background:#F8FAFF; border:2px solid {color}; border-radius:10px;
     padding:1rem 1.2rem;'>
<b style='color:{color}; font-size:1.1rem;'>{label}</b>
<p style='margin:0.5rem 0 0; color:#1E293B; font-size:0.9rem;'><b>{name}</b></p>
<p style='margin:0.2rem 0; color:#64748B; font-size:0.85rem;'>
  📍 {country} &nbsp;|&nbsp; 📅 {year} &nbsp;|&nbsp; 👤 {host}
</p>
<p style='margin:0.2rem 0; color:#4338CA; font-size:0.85rem;'>
  🧬 MLST: {mlst}
</p>
</div>"""

    with meta_col_a:
        st.markdown(_meta_card(meta_a, f"🔵 Genome A ({gid_a})", "#3B82F6"), unsafe_allow_html=True)
    with meta_col_b:
        st.markdown(_meta_card(meta_b, f"🟠 Genome B ({gid_b})", "#F97316"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Head-to-head comparison chart ────────────────────────────────────────
    st.subheader("Resistance probability comparison")

    df_merged = df_a.rename(columns={"prob_r": "prob_a", "verdict": "verdict_a", "confidence": "conf_a"})[
        ["antibiotic", "drug_class", "prob_a", "verdict_a", "conf_a"]
    ].merge(
        df_b.rename(columns={"prob_r": "prob_b", "verdict": "verdict_b", "confidence": "conf_b"})[
            ["antibiotic", "prob_b", "verdict_b", "conf_b"]
        ],
        on="antibiotic"
    )
    df_merged["delta"] = df_merged["prob_a"] - df_merged["prob_b"]
    df_merged["concordant"] = df_merged["verdict_a"] == df_merged["verdict_b"]

    # Grouped bar chart
    ab_sorted = df_merged.sort_values("prob_a", ascending=False)["antibiotic"].tolist()

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name=f"🔵 Genome A ({gid_a})",
        x=df_merged["antibiotic"],
        y=df_merged["prob_a"],
        marker_color=[verdict_color(v) for v in df_merged["verdict_a"]],
        opacity=0.85,
        text=[f"{p:.0f}%" for p in df_merged["prob_a"]],
        textposition="outside",
        customdata=df_merged[["verdict_a", "conf_a"]].values,
        hovertemplate="<b>%{x}</b><br>P(R) = %{y:.1f}%<br>Verdict: %{customdata[0]}<br>Confidence: %{customdata[1]:.1f}%<extra></extra>",
    ))
    fig_cmp.add_trace(go.Bar(
        name=f"🟠 Genome B ({gid_b})",
        x=df_merged["antibiotic"],
        y=df_merged["prob_b"],
        marker_color=[verdict_color(v) for v in df_merged["verdict_b"]],
        opacity=0.5,
        text=[f"{p:.0f}%" for p in df_merged["prob_b"]],
        textposition="outside",
        customdata=df_merged[["verdict_b", "conf_b"]].values,
        hovertemplate="<b>%{x}</b><br>P(R) = %{y:.1f}%<br>Verdict: %{customdata[0]}<br>Confidence: %{customdata[1]:.1f}%<extra></extra>",
    ))
    fig_cmp.add_hline(y=50, line_dash="dot", line_color="#94A3B8",
                      annotation_text="50% threshold")
    fig_cmp.update_layout(
        barmode="group",
        yaxis=dict(title="P(Resistant) %", range=[0, 115]),
        height=420, margin=dict(t=20, b=10, r=20),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B", yaxis_gridcolor="#E2E8F0",
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Concordance table ──────────────────────────────────────────────────────
    st.subheader("Antibiotic-by-antibiotic verdict table")

    n_concordant = df_merged["concordant"].sum()
    n_discordant = (~df_merged["concordant"]).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Concordant verdicts", n_concordant)
    c2.metric("Discordant verdicts", n_discordant)
    similarity_pct = n_concordant / len(df_merged) * 100
    c3.metric("Profile similarity", f"{similarity_pct:.0f}%",
              help="Fraction of antibiotics where both strains share the same verdict")

    rows_html = ""
    for _, r in df_merged.iterrows():
        disc_flag = "⚠️" if not r["concordant"] else ""
        rows_html += f"""
<tr style='background:{"#FFFBEB" if not r["concordant"] else "#FFFFFF"}'>
  <td style='padding:6px 10px; color:#1E293B; font-weight:600;'>{r["antibiotic"]}</td>
  <td style='padding:6px 10px; color:#64748B; font-size:0.85rem;'>{r["drug_class"]}</td>
  <td style='padding:6px 10px; text-align:center;'>
    <span style='background:{verdict_bg(r["verdict_a"])}; color:{verdict_tc(r["verdict_a"])};
     border-radius:4px; padding:2px 8px; font-size:0.85rem; font-weight:600;'>
      {r["verdict_a"]}
    </span>
  </td>
  <td style='padding:6px 10px; text-align:center;'>
    <span style='background:{verdict_bg(r["verdict_b"])}; color:{verdict_tc(r["verdict_b"])};
     border-radius:4px; padding:2px 8px; font-size:0.85rem; font-weight:600;'>
      {r["verdict_b"]}
    </span>
  </td>
  <td style='padding:6px 10px; text-align:center; color:#64748B;'>{disc_flag}</td>
  <td style='padding:6px 10px; text-align:right; font-family:monospace; font-size:0.9rem;
     color:{"#EF4444" if r["delta"] > 10 else "#10B981" if r["delta"] < -10 else "#94A3B8"};'>
     {r["delta"]:+.1f}%
  </td>
</tr>"""

    st.markdown(f"""
<table style='width:100%; border-collapse:collapse; font-size:0.9rem;'>
<thead>
<tr style='background:#F5F3FF;'>
  <th style='padding:8px 10px; text-align:left; color:#4338CA;'>Antibiotic</th>
  <th style='padding:8px 10px; text-align:left; color:#4338CA;'>Drug class</th>
  <th style='padding:8px 10px; text-align:center; color:#3B82F6;'>🔵 Genome A</th>
  <th style='padding:8px 10px; text-align:center; color:#F97316;'>🟠 Genome B</th>
  <th style='padding:8px 10px; text-align:center; color:#4338CA;'>Diff?</th>
  <th style='padding:8px 10px; text-align:right; color:#4338CA;'>Δ P(R)</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
""", unsafe_allow_html=True)
    st.caption("⚠️ = discordant verdicts. Δ P(R) = Genome A minus Genome B probability of resistance.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gene differences ───────────────────────────────────────────────────────
    if genes_a or genes_b:
        st.subheader("Resistance gene differences")

        only_a = sorted(genes_a - genes_b)
        only_b = sorted(genes_b - genes_a)
        shared = sorted(genes_a & genes_b)

        g1, g2, g3 = st.columns(3)
        g1.metric("Genes only in A", len(only_a))
        g2.metric("Genes only in B", len(only_b))
        g3.metric("Shared genes", len(shared))

        gcol_a, gcol_b = st.columns(2)
        with gcol_a:
            st.markdown(f"**🔵 Genes unique to Genome A ({gid_a})**")
            if only_a:
                for g in only_a[:30]:
                    st.markdown(
                        f"<div class='gene-card' style='border-left:3px solid #3B82F6;'>"
                        f"<b style='color:#1D4ED8;'>{g}</b></div>",
                        unsafe_allow_html=True
                    )
                if len(only_a) > 30:
                    st.caption(f"… and {len(only_a)-30} more")
            else:
                st.info("No genes unique to Genome A")

        with gcol_b:
            st.markdown(f"**🟠 Genes unique to Genome B ({gid_b})**")
            if only_b:
                for g in only_b[:30]:
                    st.markdown(
                        f"<div class='gene-card' style='border-left:3px solid #F97316;'>"
                        f"<b style='color:#C2410C;'>{g}</b></div>",
                        unsafe_allow_html=True
                    )
                if len(only_b) > 30:
                    st.caption(f"… and {len(only_b)-30} more")
            else:
                st.info("No genes unique to Genome B")

        if shared:
            with st.expander(f"Shared resistance genes ({len(shared)})", expanded=False):
                cols_shared = st.columns(3)
                for i, g in enumerate(shared):
                    with cols_shared[i % 3]:
                        st.markdown(
                            f"<div class='gene-card'>{g}</div>",
                            unsafe_allow_html=True
                        )

    st.divider()

    # ── Clinical interpretation ────────────────────────────────────────────────
    st.subheader("Clinical interpretation")

    if similarity_pct >= 90:
        st.success(f"""
**{similarity_pct:.0f}% profile similarity** — These strains are nearly identical in resistance.

Possible explanations:
- Same outbreak clone circulating in the same ward
- Patient A's strain transferred to Patient B
- Both acquired from a common environmental source

**Recommendation:** Investigate epidemiological link. Consider WGS SNP analysis to confirm.
""")
    elif similarity_pct >= 70:
        st.info(f"""
**{similarity_pct:.0f}% profile similarity** — Moderate similarity; some key differences.

These strains may be related but have diverged in some resistance elements.
The discordant antibiotics ({n_discordant} drugs) may represent:
- Plasmid gain/loss events
- Chromosomal mutations acquired during treatment
- Different sub-lineages of the same ST

**Recommendation:** Check MLST type and sequence date for epidemiological context.
""")
    else:
        st.warning(f"""
**{similarity_pct:.0f}% profile similarity** — Strains are substantially different.

These are likely **unrelated strains** from different sources. The {n_discordant} discordant
antibiotics indicate distinct resistance gene profiles.

**Recommendation:** Treat as separate cases. No evidence of outbreak connection from resistance profiles alone.
""")

else:
    # Placeholder before running
    st.info("Enter two BV-BRC Genome IDs above and click **Compare Strains** to see the analysis.")
    st.markdown("""
**Example genome IDs to try:**
- `573.12783` — Multi-drug resistant clinical isolate
- `573.65923` — Susceptible reference strain
- `573.52406` — Carbapenem-resistant (KPC-producing) strain
- `573.49554` — Fluoroquinolone-resistant strain
""")
