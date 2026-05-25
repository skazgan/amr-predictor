"""
Page 16 — Methods & Citations

Detailed methodology writeup covering:
- Dataset and preprocessing
- Feature engineering (k-mers + gene flags)
- Model architecture and ensemble
- Evaluation metrics
- Academic references
"""
import streamlit as st


import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))
from utils import inject_mobile_css
st.set_page_config(page_title="Methods & Citations", page_icon="📖", layout="wide")
inject_mobile_css()
st.title("📖 Methods & Citations")
st.markdown("*Detailed methodology, model architecture, and academic references.*")
st.divider()

# ── Overview ──────────────────────────────────────────────────────────────────
st.header("Overview")
st.markdown("""
This project applies machine learning to predict antibiotic resistance in *Klebsiella pneumoniae*
directly from whole-genome sequences. We trained separate binary classifiers for 10 clinically
important antibiotics, using a soft-voting ensemble of XGBoost and Random Forest models.

All code, trained models, and analysis scripts are open-source at
[github.com/skazgan/amr-predictor](https://github.com/skazgan/amr-predictor).
""")

st.divider()

# ── Dataset ───────────────────────────────────────────────────────────────────
st.header("1. Dataset")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Source:** Bacterial and Viral Bioinformatics Resource Center ([BV-BRC](https://www.bv-brc.org/))

**Collection:**
- 26,105 *K. pneumoniae* genomes downloaded via BV-BRC REST API
- Antibiotic susceptibility testing (AST) results from 16,902 genomes
- Collection years: 2000–2024
- 74 countries represented

**Antibiotics (10 total):**

| Antibiotic | Drug class | Training samples |
|---|---|---|
| Ciprofloxacin | Fluoroquinolone | ~12,000 |
| Meropenem | Carbapenem | ~10,500 |
| Gentamicin | Aminoglycoside | ~9,800 |
| Tetracycline | Tetracycline | ~8,200 |
| TMP/SMX | Folate inhibitor | ~9,100 |
| Cefepime | Cephalosporin | ~7,600 |
| Amikacin | Aminoglycoside | ~8,400 |
| Imipenem | Carbapenem | ~9,200 |
| Piperacillin/Tazobactam | Beta-lactam/inhibitor | ~7,800 |
| Levofloxacin | Fluoroquinolone | ~6,900 |
""")

with col2:
    st.markdown("""
**Preprocessing:**
- Labels binarised: Resistant → 1, Susceptible → 0
- "Intermediate" susceptibility excluded from training (ambiguous breakpoints)
- Only genomes with ≥5 annotated contigs retained (quality filter)
- Class imbalance handled via stratified 5-fold cross-validation

**Label distribution:**

| Antibiotic | % Resistant |
|---|---|
| Ciprofloxacin | ~58% |
| Meropenem | ~29% |
| Gentamicin | ~44% |
| Tetracycline | ~61% |
| TMP/SMX | ~55% |
| Cefepime | ~38% |
| Amikacin | ~21% |
| Imipenem | ~26% |
| Pip/Taz | ~35% |
| Levofloxacin | ~57% |
""")

st.divider()

# ── Feature engineering ───────────────────────────────────────────────────────
st.header("2. Feature Engineering")

tab_kmer, tab_gene = st.tabs(["K-mer Frequency Features", "Resistance Gene Features"])

with tab_kmer:
    st.markdown("""
### 6-mer frequency encoding

A *k*-mer is a substring of length *k* extracted from a DNA sequence by sliding a window
one base at a time. We use *k* = 6, producing up to 4,096 possible hexamers.

**Pipeline:**
1. Load FASTA genome assembly (contigs concatenated)
2. Slide a 6-character window across each contig
3. Count occurrences of each 6-mer (ACGT only; N-bases skipped)
4. Normalise by total valid k-mer count → frequency vector
5. Feature selection via **mutual information** with the resistance label:
   - Retains the top ~256 most informative k-mers per antibiotic
   - Reduces dimensionality from 4,096 → ~256

**Rationale:**
K-mers capture genomic composition without requiring gene annotation.
Resistance mutations (e.g. gyrA Ser83 → Leu in ciprofloxacin resistance) alter local
hexamer frequencies, making k-mers sensitive markers for resistance-associated sequence variants.

**Limitation:**
When gene annotation is unavailable (e.g. raw FASTA uploads), k-mer features alone achieve
AUC ~0.65–0.70, compared to ~0.79–0.88 with full feature sets.
""")

with tab_gene:
    st.markdown("""
### Resistance gene presence/absence flags

**Gene matrix construction:**
1. BV-BRC Specialty Genes database queried for all 26,105 genomes
2. Each genome × gene matrix: binary (1 = gene present, 0 = absent)
3. ~1,600 unique resistance genes identified across the dataset

**Key gene families included:**
- **Beta-lactamases**: CTX-M (ESBL), SHV, TEM, OXA, KPC, NDM, VIM, IMP
- **Quinolone resistance**: gyrA/parC point mutations, qnrA/B/S, aac(6')-Ib-cr
- **Aminoglycoside-modifying enzymes**: aac(3), aac(6'), ant(2''), aph(3')
- **16S methylases** (high-level AMG resistance): armA, rmtB
- **Efflux pumps**: AcrAB-TolC, OqxAB, MdtABC-TolC
- **Carbapenemases**: blaKPC, blaNDM, blaOXA-48, blaVIM, blaIMP

**Combined feature vector:**
```
[k-mer freq 1, ..., k-mer freq N, gene_1, gene_2, ..., gene_M]
  (~256 features)                  (~1,600 features)
```
Total feature dimensions: ~1,856 per sample.
""")

st.divider()

# ── Model architecture ────────────────────────────────────────────────────────
st.header("3. Model Architecture")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
### Soft-Voting Ensemble

We combine two calibrated base learners with a 60/40 weighted soft vote:

```
Input features (k-mers + gene flags)
          │
    ┌─────┴──────┐
    │            │
XGBoost      Random Forest
(calibrated) (calibrated)
    │            │
 P_xgb        P_rf
    │            │
    └──── 0.6 × P_xgb + 0.4 × P_rf ────→ P(Resistant)
```

**Why ensemble?**
- XGBoost captures complex feature interactions well but can overfit noisy labels
- Random Forest is more robust to outliers and provides complementary signal
- Calibration (isotonic regression, 3-fold) ensures probabilities are meaningful, not just rankings

**Prediction thresholds:**
- P(R) ≥ 65% → **Resistant**
- P(R) ≤ 35% → **Susceptible**
- 35% < P(R) < 65% → **Uncertain** (insufficient confidence)

This creates a 3-class output that flags ambiguous predictions rather than forcing
binary decisions where the model lacks confidence.
""")

with col2:
    st.markdown("""
### Hyperparameters

**XGBoost:**
```python
n_estimators = 400
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = (auto)
```

**Random Forest:**
```python
n_estimators = 300
max_features = "sqrt"
min_samples_leaf = 2
class_weight = "balanced"
```

**Calibration:**
```python
CalibratedClassifierCV(
    base_estimator,
    method = "isotonic",
    cv = 3
)
```

**Ensemble weight:**
```python
XGBoost: 60%
Random Forest: 40%
```
""")

st.divider()

# ── Evaluation ────────────────────────────────────────────────────────────────
st.header("4. Evaluation")

st.markdown("""
All models evaluated by **5-fold stratified cross-validation** on held-out test folds.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
### AUC-ROC Results

| Antibiotic | AUC (5-fold CV) | Interpretation |
|---|---|---|
| Ciprofloxacin | **0.882** | Excellent |
| Levofloxacin | **0.879** | Excellent |
| TMP/SMX | **0.856** | Very good |
| Tetracycline | **0.843** | Very good |
| Meropenem | **0.831** | Very good |
| Imipenem | **0.818** | Good |
| Cefepime | **0.812** | Good |
| Gentamicin | **0.802** | Good |
| Amikacin | **0.791** | Good |
| Pip/Tazobactam | **0.785** | Good |

*AUC = 0.5: random; AUC = 1.0: perfect*
""")

with col2:
    st.markdown("""
### Key Observations

**Strong predictors (AUC > 0.85):**
- Fluoroquinolone resistance (Cipro/Levo) — dominated by gyrA/parC mutations
  with very consistent hexamer signatures
- TMP/SMX — sul1/2 + dhfr genes are highly specific markers
- Tetracycline — tet(A/B/C/D) gene presence is near-deterministic

**Harder predictors (AUC 0.78–0.82):**
- Aminoglycosides (Gent/Amik) — resistance mediated by multiple enzyme families
  with overlapping substrates; harder to distinguish
- Carbapenems (Mero/Imi) — carbapenemase diversity (KPC vs NDM vs OXA-48)
  requires distinct gene signatures; some resistance is non-enzymatic (porin loss)
- Pip/Taz — resistance depends on ESBL enzyme type + porin loss combination;
  genotype-phenotype mapping is noisiest here

**Comparison to literature:**
Our results are consistent with Moradigaravand et al. 2018 (AUC 0.74–0.89 for
*K. pneumoniae* using k-mer features). The gene-augmented ensemble achieves
slightly higher AUC for carbapenem resistance compared to k-mer-only models.
""")

st.divider()

# ── MLST analysis ─────────────────────────────────────────────────────────────
st.header("5. MLST & Epidemiological Analyses")

st.markdown("""
In addition to resistance prediction, this project includes four epidemiological analyses
of resistance patterns across time, geography, and bacterial lineages:

| Analysis | Method | Key finding |
|---|---|---|
| **Gene emergence** | Logistic/linear curve fitting (scipy) per gene × year | CTX-M-15 grew from 2% → 48% (2000–2024) |
| **MDR trends** | Annual MDR (3+ drugs) prevalence across merged AST data | MDR peaked ~3% in 2012, now ~1.5% |
| **Country analysis** | BV-BRC `isolation_country` field → per-country resistance profiles | Vietnam 6.9% MDR; Norway <1% meropenem R |
| **Resistance forecast** | Logistic vs linear model selection by R²; bootstrap 80% CI | 4/6 antibiotics projected >50% by 2025 |
| **MLST lineages** | BV-BRC `mlst` field → 1,498 STs; ST-specific resistance heatmaps | ST258/ST512 >30% MDR; ST11 expanding |

All analyses are available as interactive visualisations on pages 9–13 of this app.
""")

st.divider()

# ── Limitations ───────────────────────────────────────────────────────────────
st.header("6. Limitations")

st.markdown("""
- **Organism specificity**: Models are trained on *K. pneumoniae* only. Applying to other
  species (e.g. *E. coli*, *A. baumannii*) will produce unreliable predictions.

- **Data source bias**: Training data comes from BV-BRC, which over-represents hospital
  isolates from high-income countries and research institutions. Rural or community
  isolates may be underrepresented.

- **Novel mechanisms**: The model cannot detect resistance mediated by mechanisms
  absent from the training data (e.g. newly described carbapenemases, novel efflux pump variants).

- **Gene features on raw FASTA**: When uploading a FASTA without BV-BRC gene annotation,
  all gene features are set to zero. This reduces AUC by ~0.1–0.15 vs the full feature set.

- **Phenotypic vs genotypic resistance**: AST results in BV-BRC may use different
  breakpoints across labs and time periods (EUCAST vs CLSI). We applied no breakpoint
  harmonisation — this is a source of label noise.

- **MIC prediction not attempted**: MIC values in BV-BRC have ~9% coverage and
  inconsistent formats (≤1, ≥8, etc.). We restricted to binary R/S classification.
""")

st.divider()

# ── References ────────────────────────────────────────────────────────────────
st.header("7. References & Tools")

st.markdown("""
1. **Olson RD, et al.** (2023). Introducing the Bacterial and Viral Bioinformatics Resource Center (BV-BRC):
   a resource combining PATRIC, IRD and ViPR. *Nucleic Acids Research*, 51(D1), D678–D689.
   [doi:10.1093/nar/gkac1003](https://doi.org/10.1093/nar/gkac1003)

2. **Alcock BP, et al.** (2023). CARD 2023: expanded curation, support for machine learning,
   and resistome prediction at the Comprehensive Antibiotic Resistance Database.
   *Nucleic Acids Research*, 51(D1), D690–D699.
   [doi:10.1093/nar/gkac920](https://doi.org/10.1093/nar/gkac920)

3. **Chen T, Guestrin C.** (2016). XGBoost: A Scalable Tree Boosting System.
   *Proceedings of KDD 2016*, 785–794.
   [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

4. **Lundberg SM, Lee SI.** (2017). A Unified Approach to Interpreting Model Predictions.
   *Advances in Neural Information Processing Systems*, 30.
   [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)

5. **Moradigaravand D, et al.** (2018). Prediction of antibiotic resistance in *Escherichia coli*
   from large-scale pan-genome data. *PLOS Computational Biology*, 14(12), e1006258.
   [doi:10.1371/journal.pcbi.1006258](https://doi.org/10.1371/journal.pcbi.1006258)

6. **Breiman L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
   [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

7. **Platt J.** (1999). Probabilistic Outputs for Support Vector Machines.
   *Advances in Large Margin Classifiers*, MIT Press. *(Basis for isotonic calibration.)*

8. **Wattam AR, et al.** (2017). PATRIC, the Federal Database for Antibiotic-Resistant
   Organisms. *Nucleic Acids Research*, 45(D1), D535–D542.
   [doi:10.1093/nar/gkw1017](https://doi.org/10.1093/nar/gkw1017)

---

**Tools & Libraries:**

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| scikit-learn | 1.5.1 | Model training, calibration, evaluation |
| XGBoost | 2.1.1 | Gradient boosting classifier |
| pandas / numpy | 2.2.2 / 1.26.4 | Data manipulation |
| scipy | 1.13.1 | Curve fitting (forecasting), statistics |
| SHAP | 0.46.0 | Feature importance interpretation |
| Streamlit | 1.37.0 | Interactive web application |
| Plotly | 5.23.0 | Interactive visualisations |
| Biopython | 1.84 | FASTA parsing |
| fpdf2 | 2.8.7 | PDF report generation |

---

*This project was developed for research purposes. It should not be used as the sole basis
for clinical treatment decisions without laboratory confirmation.*
""")
