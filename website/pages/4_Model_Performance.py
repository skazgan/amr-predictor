"""
Page 4 — Model Performance: ROC curves, confusion matrices, AUC comparison
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
st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
inject_mobile_css()
st.title("📈 Model Performance")
st.markdown("*How well do the models actually work? Honest evaluation on held-out test data.*")
st.divider()

summary = json.loads((ART_DIR / "summary.json").read_text())
df_sum  = pd.DataFrame(summary).sort_values("test_auc", ascending=False)

# ── Section 1: What is ROC-AUC ───────────────────────────────────────────────
st.header("1. How we measure performance — ROC-AUC")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
**ROC-AUC** (Area Under the Receiver Operating Characteristic curve) is the
standard metric for binary classifiers. It measures the model's ability to
**rank** resistant strains above susceptible ones.

| AUC | Meaning |
|---|---|
| 0.50 | Random guessing (coin flip) |
| 0.70 | Fair — detects some signal |
| 0.80 | Good — clinically useful |
| 0.90 | Excellent |
| 1.00 | Perfect (impossible in practice) |

**Why not just use accuracy?**
If 70% of strains are susceptible, a model that always predicts "Susceptible"
gets 70% accuracy — but has zero clinical value. AUC is immune to this problem.
""")
with col2:
    # Illustrative ROC curves
    fig = go.Figure()
    import numpy as np
    fpr_rand = [0, 1]
    tpr_rand = [0, 1]
    fig.add_trace(go.Scatter(x=fpr_rand, y=tpr_rand, mode="lines",
        name="Random (AUC=0.50)", line=dict(dash="dot", color="#6272a4")))
    fpr_good = np.linspace(0, 1, 100)
    tpr_good = 1 - (1 - fpr_good) ** 3
    fig.add_trace(go.Scatter(x=list(fpr_good), y=list(tpr_good), mode="lines",
        name="Our model (~AUC=0.83)", line=dict(color="#e94560", width=2)))
    fig.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=260, margin=dict(t=10, b=40),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dot", color="#6272a4"), xref="x", yref="y")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 2: Summary table ──────────────────────────────────────────────────
st.header("2. Results across all 10 antibiotics (Klebsiella pneumoniae)")

st.dataframe(
    df_sum[["antibiotic", "drug_class", "n_genomes",
            "cv_auc", "test_auc", "accuracy", "precision_r", "recall_r", "f1_r"]]
    .rename(columns={
        "antibiotic": "Antibiotic", "drug_class": "Drug class",
        "n_genomes": "Genomes", "cv_auc": "CV AUC",
        "test_auc": "Test AUC", "accuracy": "Accuracy",
        "precision_r": "Precision (R)", "recall_r": "Recall (R)", "f1_r": "F1 (R)",
    })
    .style
    .background_gradient(subset=["Test AUC"], cmap="RdYlGn", vmin=0.6, vmax=1.0)
    .format({"CV AUC": "{:.3f}", "Test AUC": "{:.3f}", "Accuracy": "{:.3f}",
             "Precision (R)": "{:.3f}", "Recall (R)": "{:.3f}", "F1 (R)": "{:.3f}"}),
    use_container_width=True, hide_index=True,
)

st.divider()

# ── Section 3: ROC curves per antibiotic ─────────────────────────────────────
st.header("3. ROC curves — per antibiotic")

COLORS = ["#e94560", "#50fa7b", "#8be9fd", "#ffb86c", "#ff79c6", "#bd93f9"]
fig_roc = go.Figure()

for i, row in df_sum.iterrows():
    ab       = row["antibiotic"]
    safe_ab  = ab.replace("/", "_").replace(" ", "_")
    roc_path = ART_DIR / f"roc_{safe_ab}.json"
    if not roc_path.exists():
        continue
    roc = json.loads(roc_path.read_text())
    color = COLORS[list(df_sum["antibiotic"]).index(ab) % len(COLORS)]
    fig_roc.add_trace(go.Scatter(
        x=roc["fpr"], y=roc["tpr"], mode="lines",
        name=f"{ab} (AUC={roc['auc']:.3f})",
        line=dict(color=color, width=2),
    ))

fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
    line=dict(dash="dot", color="#6272a4"), xref="x", yref="y")
fig_roc.update_layout(
    xaxis_title="False Positive Rate (1 − Specificity)",
    yaxis_title="True Positive Rate (Sensitivity)",
    height=480, margin=dict(t=20, b=40),
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    font_color="#cdd6f4", xaxis_gridcolor="#2d2d44", yaxis_gridcolor="#2d2d44",
    legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
)
st.plotly_chart(fig_roc, use_container_width=True)

st.divider()

# ── Section 4: Confusion matrices ────────────────────────────────────────────
st.header("4. Confusion matrices — what errors does the model make?")

st.markdown("""
A confusion matrix shows where the model gets it right and where it goes wrong.

| | Predicted S | Predicted R |
|---|---|---|
| **Actual S** | ✓ True Negative (correctly called susceptible) | ✗ False Positive (unnecessary concern) |
| **Actual R** | ✗ False Negative (missed resistance — dangerous!) | ✓ True Positive (correctly caught) |

**False Negatives are clinically worse** — a resistant strain incorrectly called susceptible
could lead to treatment failure.
""")

antibiotic_choice = st.selectbox(
    "Select antibiotic:",
    options=[r["antibiotic"] for r in summary],
    index=0,
)

safe_ab  = antibiotic_choice.replace("/", "_").replace(" ", "_")
cm_path  = ART_DIR / f"cm_{safe_ab}.json"
if cm_path.exists():
    cm_data = json.loads(cm_path.read_text())
    tn, fp = cm_data["tn"], cm_data["fp"]
    fn, tp = cm_data["fn"], cm_data["tp"]

    col1, col2 = st.columns([1, 2])
    with col1:
        # Heatmap
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted S", "Predicted R"],
            y=["Actual S", "Actual R"],
            colorscale=[[0, "#1e1e2e"], [1, "#e94560"]],
            showscale=False,
            text=[[str(tn), str(fp)], [str(fn), str(tp)]],
            texttemplate="%{text}",
            textfont=dict(size=24, color="white"),
        ))
        fig_cm.update_layout(
            height=280, margin=dict(t=20, b=10),
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="#cdd6f4",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        report = cm_data.get("report", {})
        total  = tn + fp + fn + tp
        st.markdown(f"""
**{antibiotic_choice.title()} — test set results**

| Metric | Susceptible | Resistant |
|---|---|---|
| Precision | {report.get('Susceptible',{}).get('precision',0):.3f} | {report.get('Resistant',{}).get('precision',0):.3f} |
| Recall    | {report.get('Susceptible',{}).get('recall',0):.3f} | {report.get('Resistant',{}).get('recall',0):.3f} |
| F1-score  | {report.get('Susceptible',{}).get('f1-score',0):.3f} | {report.get('Resistant',{}).get('f1-score',0):.3f} |

**Overall accuracy:** {report.get('accuracy',0):.3f}
**Total test samples:** {total}

> **Recall (Resistant)** = {report.get('Resistant',{}).get('recall',0):.1%} of resistant strains correctly identified.
> A recall below 0.80 means some resistant strains are being missed.
""")

st.divider()

# ── Section 5: Progress over project ─────────────────────────────────────────
st.header("5. How performance improved at each step")

steps = ["k-mers only\n(200 genomes)", "k-mers + selection\n(1000 genomes)",
         "Gene features\n(4000 genomes)", "k-mers + genes\n(calibrated)"]
aucs  = [0.66, 0.67, 0.82, 0.83]

fig_prog = go.Figure(go.Scatter(
    x=steps, y=aucs, mode="lines+markers+text",
    text=[f"{a:.2f}" for a in aucs], textposition="top center",
    marker=dict(size=14, color="#e94560"),
    line=dict(color="#e94560", width=3),
))
fig_prog.add_hline(y=0.80, line_dash="dash", line_color="#50fa7b",
                   annotation_text="Good threshold", annotation_position="right")
fig_prog.update_layout(
    yaxis=dict(range=[0.55, 0.92], title="Mean ROC-AUC"),
    height=300, margin=dict(t=20, b=10, r=120),
    plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
    font_color="#cdd6f4", yaxis_gridcolor="#2d2d44",
)
st.plotly_chart(fig_prog, use_container_width=True)

st.divider()

# ── Section 6: Multi-organism model summary ───────────────────────────────────
st.header("6. Multi-organism model performance")
st.markdown("*26 additional models across E. coli, S. aureus, and A. baumannii — gene presence/absence features only.*")

multi_path = ART_DIR / "multi_org_summary.json"
if multi_path.exists():
    import numpy as np
    multi = json.loads(multi_path.read_text())
    df_multi = pd.DataFrame(multi)

    org_display = {
        "escherichia_coli":       "E. coli",
        "staphylococcus_aureus":  "S. aureus",
        "acinetobacter_baumannii": "A. baumannii",
    }
    org_color = {
        "E. coli":       "#ffb86c",
        "S. aureus":     "#f1fa8c",
        "A. baumannii":  "#8be9fd",
    }
    df_multi["Organism"] = df_multi["organism"].map(org_display)
    df_multi = df_multi.sort_values(["Organism", "test_auc"], ascending=[True, False])

    import plotly.express as px
    fig_mo = px.bar(
        df_multi, x="test_auc", y="antibiotic", color="Organism",
        orientation="h", barmode="group",
        color_discrete_map=org_color,
        labels={"test_auc": "ROC-AUC (20% holdout)", "antibiotic": "Antibiotic"},
    )
    fig_mo.add_vline(x=0.80, line_dash="dash", line_color="#50fa7b",
                     annotation_text="Good (0.80)")
    fig_mo.update_layout(
        height=520, margin=dict(t=20, b=20, r=20),
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", xaxis_gridcolor="#2d2d44",
        xaxis=dict(range=[0.85, 1.01]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig_mo, use_container_width=True)

    # Stats per organism
    cols = st.columns(3)
    for i, (org_key, org_name) in enumerate(org_display.items()):
        sub = df_multi[df_multi["Organism"] == org_name]
        with cols[i]:
            color = org_color[org_name]
            st.markdown(f"""
<div style='background:#1e1e2e; border-left:4px solid {color};
     padding:0.8rem 1rem; border-radius:8px;'>
  <b style='color:#cdd6f4;'>{org_name}</b><br>
  <small style='color:#6272a4;'>{len(sub)} antibiotics modelled</small><br>
  <span style='color:{color}; font-size:1.3rem; font-weight:bold;'>
    AUC {sub["test_auc"].min():.3f}–{sub["test_auc"].max():.3f}
  </span><br>
  <small style='color:#6272a4;'>
    {sub["n_total"].sum():,} total labeled genomes
  </small>
</div>""", unsafe_allow_html=True)
else:
    st.info("Run `python src/train_organisms.py` to generate multi-organism model results.")
