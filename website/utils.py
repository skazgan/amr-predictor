"""
Shared utilities for the AMR Predictor Streamlit app.
"""
import streamlit as st


MOBILE_CSS = """
<style>
/* ── Global light-pastel theme overrides ────────────────────────────── */

/* Sidebar */
[data-testid="stSidebar"] {
    background: #F5F3FF !important;
    border-right: 1px solid #E0E7FF !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #374151 !important;
}
[data-testid="stSidebarNav"] a {
    color: #374151 !important;
    border-radius: 6px;
}
[data-testid="stSidebarNav"] a:hover {
    background: #EDE9FE !important;
    color: #4338CA !important;
}
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: #EDE9FE !important;
    color: #4338CA !important;
    font-weight: 600;
}

/* Main content area */
[data-testid="stMainBlockContainer"] {
    background: #FFFFFF;
}

/* Dividers */
hr { border-color: #E2E8F0 !important; }

/* Tables */
table { border-collapse: collapse; width: 100%; }
th { background: #F5F3FF !important; color: #4338CA !important; font-weight: 600; }
td, th { border: 1px solid #E2E8F0 !important; padding: 0.5rem 0.75rem; }
tr:hover td { background: #FAFAFF !important; }

/* Code blocks */
code, pre { background: #F8F9FF !important; color: #4338CA !important;
            border: 1px solid #E0E7FF !important; border-radius: 4px; }

/* Streamlit info/warning/error boxes */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* Metric widgets */
[data-testid="metric-container"] {
    background: #FAFAFA;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

/* Selectbox, text_input */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div {
    background: #FFFFFF !important;
    border: 1px solid #C7D2FE !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background: #4F46E5 !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: #FFFFFF !important;
    color: #6366F1 !important;
    border: 1.5px solid #6366F1 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background: #EEF2FF !important;
}

/* ── Mobile / responsive overrides ─────────────────────────────────── */

@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 0.75rem !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .hero h1  { font-size: 1.8rem !important; }
    .hero p   { font-size: 1rem !important; }
    .metric-card .value { font-size: 1.8rem !important; }
    .stPlotlyChart > div { overflow-x: auto !important; }
    [data-testid="stDataFrame"] > div,
    .stTable > div { overflow-x: auto !important; }
    .js-plotly-plot .plotly { min-height: unset !important; }
    div[style*="display:flex"] { flex-wrap: wrap !important; }
    [data-testid="stSidebar"] { min-width: 0 !important; }
    .streamlit-expanderHeader { font-size: 0.9rem !important; }
    .stDownloadButton { width: 100% !important; margin-bottom: 0.5rem !important; }
}

@media (max-width: 480px) {
    [data-testid="stMainBlockContainer"] { padding: 0.5rem !important; }
    .hero h1  { font-size: 1.4rem !important; }
    .metric-card .value { font-size: 1.5rem !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
}

/* ── Shared component styles ────────────────────────────────────────── */

.verdict-card {
    background: #FAFAFA;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    border: 1px solid #F1F5F9;
}
.verdict-label { color: #1E293B; font-weight: bold; }
.verdict-class { color: #64748B; font-size: 0.8rem; margin-left: 8px; }
.verdict-bar-wrap { text-align: right; min-width: 180px; }
@media (max-width: 600px) { .verdict-bar-wrap { min-width: 100%; } }

.prob-bar-bg { background: #EEF2FF; border-radius: 4px; height: 6px; margin-top: 4px; }
.prob-bar-fill { height: 6px; border-radius: 4px; }

.gene-card {
    background: #FFFFFF;
    border: 1px solid #E0E7FF;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    margin-bottom: 4px;
    font-size: 0.85rem;
    color: #1E293B;
}
</style>
"""


def inject_mobile_css():
    """Inject mobile-responsive and light-theme CSS into the current page."""
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
