"""
AMR Predictor — Navigation entrypoint.
Defines the sidebar page list and groups using st.navigation() (Streamlit 1.36+).
"""
import streamlit as st

st.set_page_config(
    page_title="AMR Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use relative paths from this file's directory — st.Page requires paths
# relative to the main script, not absolute paths (Streamlit Cloud restriction).
pg = st.navigation(
    {
        "Home": [
            st.Page("pages/0_Overview.py",             title="Home",                  icon="🏠", default=True),
        ],
        "🔬 The Science": [
            st.Page("pages/1_The_Biology.py",          title="The Biology",           icon="🦠"),
            st.Page("pages/2_The_Data.py",             title="The Data",              icon="📊"),
            st.Page("pages/3_Features.py",             title="Features",              icon="🔬"),
            st.Page("pages/4_Model_Performance.py",    title="Model Performance",     icon="📈"),
        ],
        "🔮 Predictors": [
            st.Page("pages/5_Live_Predictor.py",       title="Live Predictor",        icon="🔮"),
            st.Page("pages/14_Offline_Predictor.py",   title="Offline Predictor",     icon="⚡"),
            st.Page("pages/15_FASTA_Upload.py",        title="FASTA Upload",          icon="📂"),
            st.Page("pages/17_Multi_Organism.py",      title="Multi-Organism",        icon="🦠"),
            st.Page("pages/21_Strain_Comparison.py",   title="Strain Comparison",     icon="⚖️"),
        ],
        "🧫 Clinical Tools": [
            st.Page("pages/18_Antibiogram.py",              title="Antibiogram Heatmap",       icon="🧫"),
            st.Page("pages/19_Treatment_Recommendation.py", title="Treatment Recommendation",  icon="💊"),
            st.Page("pages/20_Outbreak_Detection.py",       title="Outbreak Detection",        icon="🔬"),
        ],
        "🔍 Explainability": [
            st.Page("pages/6_Explainability.py",       title="Explainability (SHAP)", icon="🔍"),
            st.Page("pages/7_Co_Resistance_Network.py",title="Co-Resistance Network", icon="🕸️"),
        ],
        "🌍 Epidemiology": [
            st.Page("pages/8_Temporal_Drift.py",       title="Temporal Drift",        icon="📉"),
            st.Page("pages/9_Gene_Emergence.py",       title="Gene Emergence",        icon="🧬"),
            st.Page("pages/10_MDR_Over_Time.py",       title="MDR Over Time",         icon="⏱️"),
            st.Page("pages/11_Country_Analysis.py",    title="Country Analysis",      icon="🌍"),
            st.Page("pages/12_Resistance_Forecast.py", title="Resistance Forecast",   icon="📡"),
            st.Page("pages/13_MLST_Lineages.py",       title="MLST Lineages",         icon="🧫"),
        ],
        "📖 About": [
            st.Page("pages/16_Methods_Citations.py",   title="Methods & Citations",   icon="📖"),
        ],
    }
)

pg.run()
