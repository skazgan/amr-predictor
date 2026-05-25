"""
Shared utilities for the AMR Predictor Streamlit app.
"""
import streamlit as st


MOBILE_CSS = """
<style>
/* ── Mobile / responsive overrides ─────────────────────────────────── */

/* Tighter main padding on narrow screens */
@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 0.75rem !important;
    }
    /* Stack columns vertically */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    /* Shrink hero heading */
    .hero h1  { font-size: 1.8rem !important; }
    .hero p   { font-size: 1rem !important; }
    /* Metric cards: smaller value text */
    .metric-card .value { font-size: 1.8rem !important; }
    /* Plotly charts: let them scroll if too wide */
    .stPlotlyChart > div { overflow-x: auto !important; }
    /* DataFrames and tables: horizontal scroll */
    [data-testid="stDataFrame"] > div,
    .stTable > div { overflow-x: auto !important; }
    /* Reduce chart default height */
    .js-plotly-plot .plotly { min-height: unset !important; }
    /* Resistance card divs: stack content */
    div[style*="display:flex"] { flex-wrap: wrap !important; }
    /* Sidebar: ensure it's collapsible */
    [data-testid="stSidebar"] { min-width: 0 !important; }
    /* Expanders: full width */
    .streamlit-expanderHeader { font-size: 0.9rem !important; }
    /* Download buttons side by side even on mobile */
    .stDownloadButton { width: 100% !important; margin-bottom: 0.5rem !important; }
}

@media (max-width: 480px) {
    [data-testid="stMainBlockContainer"] {
        padding: 0.5rem !important;
    }
    .hero h1  { font-size: 1.4rem !important; }
    .metric-card .value { font-size: 1.5rem !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
}

/* ── Shared component styles (all screen sizes) ─────────────────────── */

/* Resistance verdict cards */
.verdict-card {
    background: #181825;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.verdict-label {
    color: #cdd6f4;
    font-weight: bold;
}
.verdict-class {
    color: #6272a4;
    font-size: 0.8rem;
    margin-left: 8px;
}
.verdict-bar-wrap {
    text-align: right;
    min-width: 180px;
}
@media (max-width: 600px) {
    .verdict-bar-wrap { min-width: 100%; }
}

/* Probability bar */
.prob-bar-bg {
    background: #2d2d44;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}
.prob-bar-fill {
    height: 6px;
    border-radius: 4px;
}

/* Info / warning badge */
.mdr-badge {
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: bold;
    margin-bottom: 1rem;
    font-size: 1rem;
}

/* Gene toggle cards */
.gene-card {
    background: #1e1e2e;
    border: 1px solid #2d2d44;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    margin-bottom: 4px;
    font-size: 0.85rem;
    color: #cdd6f4;
}
</style>
"""


def inject_mobile_css():
    """Inject mobile-responsive CSS into the current Streamlit page."""
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
