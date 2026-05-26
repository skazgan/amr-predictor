"""
Page 19 — Treatment Recommendation: given a resistance profile, suggest active drugs
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
from utils import inject_mobile_css, page_info_expander
inject_mobile_css()
page_info_expander("""
**Treatment tiers** — Antibiotics are ranked by clinical preference, balancing efficacy, safety, and the need to preserve last-resort options:
- 🟢 **First-line** — Preferred option when susceptibility is confirmed; good safety profile.
- 🔵 **Second-line** — Used when first-line drugs fail or are contraindicated.
- 🟣 **Adjunct** — Given alongside another antibiotic to broaden or enhance coverage.
- 🟡 **Reserve** — Effective but with notable toxicity or used to preserve susceptibility (e.g. temocillin).
- 🔴 **Last resort** — Used only when no other options remain (e.g. colistin, fosfomycin IV). Requires specialist input.

**MDR / XDR / PDR** — Multi-Drug Resistant / Extensively Drug Resistant / Pan-Drug Resistant. Classifications based on the number of antibiotic classes to which the isolate is resistant (see MDR Over Time page for full definitions).

**CRKP** — Carbapenem-Resistant *Klebsiella pneumoniae*. Resistance to meropenem/imipenem = loss of the most reliable Gram-negative cover. Triggers infection control alerts and specialist review.

**VRE** — Vancomycin-Resistant *Enterococcus*. Loss of vancomycin in *E. faecium* leaves very few options (linezolid, daptomycin, tedizolid).

**MRSA** — Methicillin-Resistant *Staphylococcus aureus*. Requires glycopeptides (vancomycin, teicoplanin) or newer agents (daptomycin, ceftaroline) instead of standard beta-lactams.

⚠️ **Disclaimer** — Recommendations shown here are population-level defaults based on published guidelines. They do not account for patient-specific factors (renal function, allergies, drug interactions, infection site, PK/PD targets). Always consult local guidelines and a clinical pharmacist or infectious disease specialist.
""")

st.title("💊 Treatment Recommendation")
st.markdown("*Given a resistance profile, which drugs are still active — and which are last resort?*")
st.divider()

# ── Clinical knowledge base ───────────────────────────────────────────────────
# Drug tiers are based on WHO AWaRe classification + IDSA/ESCMID guidelines

TREATMENT_DB = {
    "K. pneumoniae": {
        "note": "Gram-negative rod. ESBL and carbapenem-resistant strains common in hospital settings.",
        "antibiotics": [
            "meropenem", "imipenem", "piperacillin/tazobactam", "cefepime",
            "ceftriaxone", "ciprofloxacin", "levofloxacin",
            "gentamicin", "amikacin", "tetracycline",
            "trimethoprim/sulfamethoxazole",
        ],
        "tiers": {
            "meropenem":                    ("First-line",   "Carbapenem",          False),
            "imipenem":                     ("First-line",   "Carbapenem",          False),
            "piperacillin/tazobactam":      ("Second-line",  "Beta-lactam/BLI",     False),
            "cefepime":                     ("Second-line",  "Cephalosporin",       False),
            "ceftriaxone":                  ("Second-line",  "Cephalosporin",       False),
            "ciprofloxacin":                ("Adjunct",      "Fluoroquinolone",     False),
            "levofloxacin":                 ("Adjunct",      "Fluoroquinolone",     False),
            "gentamicin":                   ("Adjunct",      "Aminoglycoside",      False),
            "amikacin":                     ("Adjunct",      "Aminoglycoside",      False),
            "tetracycline":                 ("Adjunct",      "Tetracycline",        False),
            "trimethoprim/sulfamethoxazole":("Adjunct",      "Folate inhibitor",    False),
        },
        "last_resort": ["colistin", "polymyxin B", "tigecycline", "fosfomycin", "ceftazidime/avibactam"],
        "susceptible_notes": {
            "meropenem":                    "Preferred for serious/severe infections",
            "piperacillin/tazobactam":      "Good for UTI, IAI — if no ESBL",
            "cefepime":                     "Use only if ESBL-negative",
            "ciprofloxacin":                "Good for uncomplicated UTI",
            "amikacin":                     "Preferred aminoglycoside (more stable than gentamicin)",
            "trimethoprim/sulfamethoxazole":"Oral option for uncomplicated UTI",
        },
        "resistance_alert": {
            "meropenem": "⚠️ Carbapenem-resistant K. pneumoniae (CRKP) — notify infection control immediately.",
            "imipenem":  "⚠️ Carbapenem resistance detected — escalate to last-resort agents.",
        },
    },
    "E. coli": {
        "note": "Gram-negative rod. ESBL producers are widespread. UTI is common indication.",
        "antibiotics": [
            "ampicillin", "trimethoprim/sulfamethoxazole", "ciprofloxacin",
            "ceftriaxone", "cefepime", "piperacillin/tazobactam",
            "meropenem", "gentamicin", "amikacin", "tetracycline",
        ],
        "tiers": {
            "ampicillin":                   ("First-line",   "Penicillin",          False),
            "trimethoprim/sulfamethoxazole":("First-line",   "Folate inhibitor",    False),
            "ciprofloxacin":                ("Second-line",  "Fluoroquinolone",     False),
            "ceftriaxone":                  ("Second-line",  "Cephalosporin",       False),
            "cefepime":                     ("Second-line",  "Cephalosporin",       False),
            "piperacillin/tazobactam":      ("Second-line",  "Beta-lactam/BLI",     False),
            "meropenem":                    ("Reserve",      "Carbapenem",          False),
            "gentamicin":                   ("Adjunct",      "Aminoglycoside",      False),
            "amikacin":                     ("Adjunct",      "Aminoglycoside",      False),
            "tetracycline":                 ("Adjunct",      "Tetracycline",        False),
        },
        "last_resort": ["colistin", "polymyxin B", "fosfomycin"],
        "susceptible_notes": {
            "ampicillin":                   "First-line for uncomplicated UTI if susceptible",
            "trimethoprim/sulfamethoxazole":"Oral — preferred for UTI if <20% local resistance",
            "ciprofloxacin":                "Good tissue penetration — also useful for pyelonephritis",
            "meropenem":                    "Reserve for ESBL producers with systemic infection",
            "amikacin":                     "Preferred aminoglycoside — more stable against enzymes",
        },
        "resistance_alert": {
            "meropenem": "⚠️ Carbapenem-resistant E. coli — rare but highly concerning. Notify infection control.",
        },
    },
    "S. aureus": {
        "note": "Gram-positive coccus. MRSA (oxacillin-resistant) requires fundamentally different treatment pathway.",
        "antibiotics": [
            "oxacillin", "ciprofloxacin", "clindamycin",
            "erythromycin", "gentamicin", "tetracycline",
            "trimethoprim/sulfamethoxazole", "vancomycin",
        ],
        "tiers": {
            "oxacillin":                    ("First-line",   "Beta-lactam (MSSA)",  False),
            "vancomycin":                   ("First-line",   "Glycopeptide (MRSA)", False),
            "clindamycin":                  ("Second-line",  "Lincosamide",         False),
            "trimethoprim/sulfamethoxazole":("Second-line",  "Folate inhibitor",    False),
            "ciprofloxacin":                ("Adjunct",      "Fluoroquinolone",     False),
            "gentamicin":                   ("Adjunct",      "Aminoglycoside",      False),
            "erythromycin":                 ("Adjunct",      "Macrolide",           False),
            "tetracycline":                 ("Adjunct",      "Tetracycline",        False),
        },
        "last_resort": ["linezolid", "daptomycin", "ceftaroline", "tedizolid"],
        "susceptible_notes": {
            "oxacillin":                    "Drug of choice for MSSA — excellent clinical outcomes",
            "vancomycin":                   "First-line for MRSA — monitor levels (target AUC 400–600)",
            "clindamycin":                  "Useful for skin/soft tissue — check D-zone inducible resistance",
            "trimethoprim/sulfamethoxazole":"Oral MRSA coverage — useful for community-acquired MRSA",
            "daptomycin":                   "Last resort — do not use for pulmonary infections",
        },
        "resistance_alert": {
            "vancomycin": "⚠️ Vancomycin-resistant S. aureus (VRSA) — extremely rare, notify public health.",
            "oxacillin":  "MRSA confirmed — switch to vancomycin or alternative MRSA-active agent.",
        },
    },
    "A. baumannii": {
        "note": "Gram-negative coccobacillus. Notorious for acquiring pan-resistance. Hospital-associated.",
        "antibiotics": [
            "imipenem", "meropenem", "ciprofloxacin",
            "gentamicin", "amikacin", "tetracycline",
            "trimethoprim/sulfamethoxazole", "colistin",
        ],
        "tiers": {
            "imipenem":                     ("First-line",   "Carbapenem",          False),
            "meropenem":                    ("First-line",   "Carbapenem",          False),
            "ciprofloxacin":                ("Adjunct",      "Fluoroquinolone",     False),
            "gentamicin":                   ("Adjunct",      "Aminoglycoside",      False),
            "amikacin":                     ("Adjunct",      "Aminoglycoside",      False),
            "tetracycline":                 ("Adjunct",      "Tetracycline",        False),
            "trimethoprim/sulfamethoxazole":("Adjunct",      "Folate inhibitor",    False),
            "colistin":                     ("Last resort",  "Polymyxin",           True),
        },
        "last_resort": ["colistin", "polymyxin B", "tigecycline", "sulbactam", "cefiderocol"],
        "susceptible_notes": {
            "imipenem":     "Preferred if susceptible — higher CNS penetration than meropenem",
            "meropenem":    "Good for systemic infections — use with sulbactam for better coverage",
            "colistin":     "Last resort only — nephrotoxic. Use combination therapy.",
            "tigecycline":  "Bacteriostatic — avoid as monotherapy for bacteraemia",
        },
        "resistance_alert": {
            "imipenem":  "⚠️ Carbapenem-resistant A. baumannii (CRAB) — PDR common. Notify infection control.",
            "meropenem": "⚠️ Carbapenem resistance — only colistin/cefiderocol combinations may work.",
            "colistin":  "⚠️ Pan-drug resistant (PDR) A. baumannii — consult infectious disease specialist immediately.",
        },
    },
}

TIER_COLOR = {
    "First-line":  ("#D1FAE5", "#065F46", "#059669"),
    "Second-line": ("#DBEAFE", "#1E40AF", "#3B82F6"),
    "Adjunct":     ("#F5F3FF", "#4338CA", "#6366F1"),
    "Reserve":     ("#FEF9C3", "#713F12", "#D97706"),
    "Last resort": ("#FFF5F5", "#991B1B", "#EF4444"),
}
RESIST_COLOR = ("#FEE2E2", "#991B1B", "#EF4444")
UNKNOWN_COLOR = ("#F1F5F9", "#475569", "#94A3B8")

# ── Section 1: Explanation ────────────────────────────────────────────────────
st.header("1. How does treatment recommendation work?")
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
Clinical microbiologists and infectious disease physicians use a **structured approach**:

1. **Identify the organism** (K. pneumoniae, E. coli, S. aureus, A. baumannii)
2. **Obtain susceptibility testing** (MIC or disc diffusion) — takes 24–72 hours
3. **Match to treatment tier:**
   - **First-line** — most effective, narrowest spectrum
   - **Second-line** — alternatives when first-line fails or patient allergies
   - **Adjunct** — useful in combination or specific indications
   - **Last resort** — reserved for pan-resistant strains; higher toxicity

4. **Flag last-resort options** — colistin, vancomycin (MRSA), linezolid, cefiderocol
   are reserved due to toxicity and to preserve their effectiveness

This tool uses **model predictions** (from the Live Predictor or your manual input)
as a substitute for lab susceptibility results.
""")
with col2:
    st.info("""
**Important clinical note**

This tool provides **educational guidance** based on standard published guidelines
(IDSA, ESCMID, WHO AWaRe). Always consult:
- Local antibiogram for institutional resistance rates
- An infectious disease specialist for complex cases
- Official treatment guidelines for your region

Predictions have uncertainty — always confirm with culture and MIC data.
""")

st.divider()

# ── Section 2: Input ──────────────────────────────────────────────────────────
st.header("2. Enter resistance profile")

org_sel = st.selectbox(
    "Select organism:",
    list(TREATMENT_DB.keys()),
    key="treat_org",
)
db = TREATMENT_DB[org_sel]
st.caption(f"📝 {db['note']}")

st.markdown("**Mark each antibiotic as Resistant (R), Susceptible (S), or Unknown:**")

antibiotics = db["antibiotics"]
resistance_profile = {}

cols_per_row = 3
rows = [antibiotics[i:i+cols_per_row] for i in range(0, len(antibiotics), cols_per_row)]

for row_abs in rows:
    cols = st.columns(cols_per_row)
    for i, ab in enumerate(row_abs):
        with cols[i]:
            tier, drug_class, is_last = db["tiers"].get(ab, ("Adjunct", "Other", False))
            label = ab.replace("/", "/​")  # zero-width space for line break
            last_tag = " 🚨" if is_last else ""
            resistance_profile[ab] = st.radio(
                f"**{label}**{last_tag}",
                options=["Unknown", "Susceptible", "Resistant"],
                horizontal=True,
                key=f"resist_{org_sel}_{ab}",
                index=0,
            )

st.divider()

# ── Section 3: Recommendation ─────────────────────────────────────────────────
st.header("3. Recommendation")

n_resistant   = sum(1 for v in resistance_profile.values() if v == "Resistant")
n_susceptible = sum(1 for v in resistance_profile.values() if v == "Susceptible")
n_unknown     = sum(1 for v in resistance_profile.values() if v == "Unknown")
n_total       = len(resistance_profile)

# Overall assessment
if n_resistant == n_total:
    st.error("🔴 **Pan-resistant profile** — all tested antibiotics are resistant. "
             "Consult infectious disease specialist immediately. Consider last-resort agents.")
elif n_resistant >= n_total * 0.7:
    st.warning("🟠 **Extensively drug-resistant (XDR)** — ≥70% of antibiotics are resistant. "
               "Limited options remain. Consider combination therapy and specialist review.")
elif n_resistant >= 3:
    st.warning("🟡 **Multi-drug resistant (MDR)** — resistant to 3+ antibiotics. "
               "Careful drug selection required.")
else:
    st.success("🟢 **Susceptibility profile acceptable** — viable treatment options exist.")

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Susceptible", n_susceptible, help="Drugs confirmed active")
with metric_col2:
    st.metric("Resistant", n_resistant, help="Drugs to avoid")
with metric_col3:
    st.metric("Unknown", n_unknown, help="Not yet tested")

st.divider()

# Active drugs by tier
st.subheader("Active drug options (susceptible only)")

tiers_order = ["First-line", "Second-line", "Adjunct", "Reserve"]
shown_any = False
for tier in tiers_order:
    tier_drugs = [ab for ab, (t, cls, _) in db["tiers"].items()
                  if t == tier and resistance_profile.get(ab) == "Susceptible"]
    if not tier_drugs:
        continue
    shown_any = True
    bg, tc, bc = TIER_COLOR[tier]
    drugs_html = ""
    for ab in tier_drugs:
        _, cls, _ = db["tiers"][ab]
        note = db.get("susceptible_notes", {}).get(ab, "")
        drugs_html += f"""
<div style='background:#FFFFFF; border:1px solid {bc}; border-left:4px solid {bc};
     border-radius:6px; padding:0.5rem 0.8rem; margin-bottom:6px;'>
  <b style='color:{tc};'>{ab.title()}</b>
  <span style='color:#64748B; font-size:0.8rem; margin-left:8px;'>({cls})</span>
  {'<br><small style="color:#475569;">' + note + '</small>' if note else ''}
</div>"""
    st.markdown(f"""
<div style='background:{bg}; border:1px solid {bc}; border-radius:10px;
     padding:0.8rem 1rem; margin-bottom:1rem;'>
<b style='color:{tc}; font-size:1rem;'>{tier}</b>
{drugs_html}
</div>""", unsafe_allow_html=True)

if not shown_any:
    st.info("No susceptible antibiotics entered yet — mark at least one antibiotic as 'Susceptible' above.")

# Resistant drugs warning
resistant_drugs = [ab for ab, v in resistance_profile.items() if v == "Resistant"]
if resistant_drugs:
    st.subheader("❌ Drugs to avoid (resistant)")
    # Show alerts for key drugs
    for ab in resistant_drugs:
        alert = db.get("resistance_alert", {}).get(ab)
        if alert:
            st.error(alert)
    # Show all resistant drugs as compact badges
    badges = " ".join(
        f"<span style='background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; "
        f"border-radius:4px; padding:2px 8px; font-size:0.85rem; margin:2px; display:inline-block;'>"
        f"{ab}</span>"
        for ab in resistant_drugs
    )
    st.markdown(badges, unsafe_allow_html=True)

# Last resort section
st.subheader("🚨 Last-resort options")
last_resort_drugs = db["last_resort"]
st.markdown(f"""
<div style='background:#FFF5F5; border:2px solid #FCA5A5; border-radius:10px;
     padding:1rem 1.2rem; margin-bottom:1rem;'>
<p style='color:#991B1B; margin:0 0 0.5rem;'>
  <b>Reserved for pan-resistant or extensively resistant strains only.</b>
  These agents have significant toxicity and/or are the last line of defence.
  Overuse accelerates resistance development.
</p>
<p style='margin:0; color:#475569;'>
  {'  ·  '.join(f'<b>{d}</b>' for d in last_resort_drugs)}
</p>
</div>""", unsafe_allow_html=True)

st.divider()

# ── Section 4: Visual summary ─────────────────────────────────────────────────
st.header("4. Visual resistance profile")

if any(v != "Unknown" for v in resistance_profile.values()):
    ab_labels, statuses, colors, tiers_list = [], [], [], []
    for ab, verdict in resistance_profile.items():
        ab_labels.append(ab)
        statuses.append(verdict)
        if verdict == "Resistant":
            colors.append("#EF4444")
        elif verdict == "Susceptible":
            colors.append("#10B981")
        else:
            colors.append("#CBD5E1")
        tiers_list.append(db["tiers"].get(ab, ("?", "?", False))[0])

    df_vis = pd.DataFrame({
        "Antibiotic": ab_labels,
        "Status": statuses,
        "color": colors,
        "Tier": tiers_list,
    })

    fig_vis = go.Figure(go.Bar(
        x=df_vis["Antibiotic"],
        y=[1] * len(df_vis),
        marker_color=df_vis["color"],
        text=df_vis["Status"],
        textposition="inside",
        textfont=dict(color="white", size=12),
        hovertemplate="<b>%{x}</b><br>%{text}<br>Tier: " +
                      df_vis["Tier"] + "<extra></extra>",
    ))
    fig_vis.update_layout(
        yaxis=dict(visible=False, range=[0, 1.4]),
        height=180, margin=dict(t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#1E293B",
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_vis, use_container_width=True)
    st.caption("🟢 Susceptible  🔴 Resistant  ⬜ Unknown")

st.divider()

# ── Section 5: Clinical decision pathway ─────────────────────────────────────
st.header("5. Clinical decision pathway")

st.markdown(f"""
<div style='background:#F8FAFF; border:1px solid #E0E7FF; border-radius:12px; padding:1.2rem 1.5rem;'>
<h4 style='color:#4338CA; margin:0 0 0.8rem;'>For {org_sel}</h4>

**Step 1 — Severity assessment**
- Is this a severe systemic infection (bacteraemia, pneumonia, meningitis)?
- Or a mild infection (uncomplicated UTI, skin/soft tissue)?

**Step 2 — Empiric therapy** (before susceptibility known)
- Use your local antibiogram and institutional guidelines
- Adjust based on patient factors (allergies, renal function, pregnancy)

**Step 3 — De-escalate** once susceptibility results are available
- Narrow spectrum whenever possible
- Switch IV to oral if patient is stable

**Step 4 — Duration**
- Bacteraemia: minimum 14 days (endovascular source: 4–6 weeks)
- UTI uncomplicated: 3–7 days
- Pneumonia: 5–7 days (non-severe)

**Step 5 — Infection control**
- MRSA, CRAB, CRKP, VRE: contact precautions + notify infection control
- Consider screening contacts in outbreak settings
</div>
""", unsafe_allow_html=True)
