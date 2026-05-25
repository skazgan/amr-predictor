"""
AMR Predictor — Navigation entrypoint.
Defines the sidebar page list and groups using st.navigation() (Streamlit 1.36+).
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="AMR Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

P = Path(__file__).parent / "pages"

pg = st.navigation(
    {
        "Home": [
            st.Page(str(P / "0_Overview.py"),             title="Home",                     icon="🏠", default=True),
        ],
        "🔬 The Science": [
            st.Page(str(P / "1_The_Biology.py"),          title="The Biology",               icon="🦠"),
            st.Page(str(P / "2_The_Data.py"),             title="The Data",                  icon="📊"),
            st.Page(str(P / "3_Features.py"),             title="Features",                  icon="🔬"),
            st.Page(str(P / "4_Model_Performance.py"),    title="Model Performance",          icon="📈"),
        ],
        "🔮 Predictors": [
            st.Page(str(P / "5_Live_Predictor.py"),       title="Live Predictor",            icon="🔮"),
            st.Page(str(P / "14_Offline_Predictor.py"),   title="Offline Predictor",         icon="⚡"),
            st.Page(str(P / "15_FASTA_Upload.py"),        title="FASTA Upload",              icon="📂"),
        ],
        "🔍 Explainability": [
            st.Page(str(P / "6_Explainability.py"),       title="Explainability (SHAP)",     icon="🔍"),
            st.Page(str(P / "7_Co_Resistance_Network.py"),title="Co-Resistance Network",     icon="🕸️"),
        ],
        "🌍 Epidemiology": [
            st.Page(str(P / "8_Temporal_Drift.py"),       title="Temporal Drift",            icon="📉"),
            st.Page(str(P / "9_Gene_Emergence.py"),       title="Gene Emergence",            icon="🧬"),
            st.Page(str(P / "10_MDR_Over_Time.py"),       title="MDR Over Time",             icon="⏱️"),
            st.Page(str(P / "11_Country_Analysis.py"),    title="Country Analysis",          icon="🌍"),
            st.Page(str(P / "12_Resistance_Forecast.py"), title="Resistance Forecast",       icon="📡"),
            st.Page(str(P / "13_MLST_Lineages.py"),       title="MLST Lineages",             icon="🧫"),
        ],
        "📖 About": [
            st.Page(str(P / "16_Methods_Citations.py"),   title="Methods & Citations",       icon="📖"),
        ],
    }
)

pg.run()
