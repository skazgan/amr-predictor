# 🧬 AMR Predictor — Antibiotic Resistance in *K. pneumoniae*

A machine-learning system that predicts antibiotic resistance in *Klebsiella pneumoniae*
from whole-genome sequencing data — built from scratch using 16,902 public genomes from [BV-BRC](https://www.bv-brc.org/).

**Live demo:** *(deploy link goes here — see Deployment below)*

---

## What it does

Given a bacterial genome, the system predicts whether the strain is **Resistant (R)** or
**Susceptible (S)** to 6 antibiotics, with calibrated probability scores:

| Antibiotic | Drug class | Model AUC |
|---|---|---|
| Ciprofloxacin | Fluoroquinolone | 0.790 |
| Meropenem | Carbapenem | 0.869 |
| Gentamicin | Aminoglycoside | 0.821 |
| Tetracycline | Tetracycline | 0.773 |
| Trimethoprim/sulfamethoxazole | Folate inhibitor | 0.842 |
| Cefepime | Cephalosporin | 0.802 |

---

## Website — 12 interactive pages

| # | Page | What you learn |
|---|---|---|
| 1 | The Biology | DNA, bacteria, antibiotic mechanisms, K. pneumoniae |
| 2 | The Data | BV-BRC source, 16,902 genomes, class balance |
| 3 | Features | K-mer sliding window, resistance gene presence/absence |
| 4 | Model Performance | AUC curves, confusion matrices, progress over time |
| 5 | Live Predictor | Enter any BV-BRC genome ID → instant resistance profile |
| 6 | Explainability | SHAP values, which genes drive each prediction |
| 7 | Co-Resistance Network | φ correlation matrix, MDR plasmid clusters |
| 8 | Temporal Drift | How resistance has changed 2000–2024, model decay |
| 9 | Gene Emergence | Epidemic-style curves for 40 resistance genes |
| 10 | MDR Over Time | Multi-drug resistance trajectory across 24 years |
| 11 | Country Analysis | Global resistance heatmap, 74 countries, geographic hotspots |
| 12 | Resistance Forecast | 2025–2030 projections with confidence intervals |

---

## Key research findings

- **Co-resistance:** Gentamicin × TMP/SMX (φ=+0.544) — strongest linked pair, travel on same plasmids
- **Independence:** Meropenem × Cefepime (φ≈0.002) — both beta-lactams, entirely different resistance mechanisms
- **Temporal drift:** Gentamicin model decays fastest (+0.082 AUC drop on post-2020 data)
- **Geography:** Norway <1% meropenem resistance; Italy 78% — 100× difference in last-resort antibiotic
- **Forecast:** 4 of 6 antibiotics projected to exceed 50% resistance by 2025 if trends continue
- **MDR trend:** Peaked at 3% around 2013, now ~1.5% — chronic plateau, not escalating epidemic

---

## How it works

```
Raw FASTA genome
      │
      ▼
K-mer counting (k=6)          Resistance gene annotation
4,096 6-mer frequencies   +   ~1,600 binary gene flags
      │                              │
      └──────────────────────────────┘
                    │
              Feature selection
            (top 256 k-mers via
           mutual information, MI)
                    │
              XGBoost classifier
            + isotonic calibration
            (5-fold cross-validation)
                    │
            Calibrated probability
            R / S verdict + confidence
```

---

## Project structure

```
amr-predictor/
├── website/
│   ├── Home.py                  # Streamlit entry point
│   └── pages/                   # 12 analysis pages
├── src/
│   ├── download_multi.py        # BV-BRC genome downloader
│   ├── features.py              # K-mer feature extraction
│   ├── gene_features.py         # Resistance gene matrix builder
│   ├── train.py                 # XGBoost training + calibration
│   ├── train_multi.py           # Train one model per antibiotic
│   ├── generate_artifacts.py    # Pre-compute all website JSONs
│   ├── coresistance.py          # φ correlation, MDR, cross-prediction
│   ├── temporal_drift.py        # Year-by-year resistance + model decay
│   ├── gene_emergence.py        # Epidemic growth curves per gene
│   ├── mdr_over_time.py         # MDR trajectory 2000–2024
│   ├── country_analysis.py      # Geographic resistance mapping
│   └── resistance_forecast.py   # 2025–2030 logistic/linear projection
├── models/                      # Saved .pkl model bundles (6 antibiotics)
├── artifacts/                   # Pre-computed JSON artifacts (34 files)
├── data/
│   ├── raw/                     # Downloaded FASTA files (gitignored, 34 GB)
│   └── processed/               # Feature matrices + metadata
└── requirements.txt
```

---

## Run locally

```bash
# 1. Clone
git clone https://github.com/skazgan/amr-predictor.git
cd amr-predictor

# 2. Create environment
conda create -n amr python=3.12
conda activate amr
pip install -r requirements.txt

# 3. Launch website (uses pre-computed artifacts — no data download needed)
streamlit run website/Home.py
```

The website works immediately from the pre-computed `artifacts/` JSONs.
To retrain models or regenerate artifacts, you need the full dataset (~34 GB).

---

## Data source & reproducibility

All genome sequences and resistance labels downloaded from
**BV-BRC** (Bacterial and Viral Bioinformatics Resource Center),
taxon *Klebsiella pneumoniae* (ID 573), genome IDs publicly available.

```python
# Example BV-BRC API query
GET https://www.bv-brc.org/api/genome_amr/
    ?eq(taxon_id,573)
    &eq(antibiotic,ciprofloxacin)
    &select(genome_id,resistant_phenotype)
    &limit(10000)
```

---

## Tech stack

| Component | Library |
|---|---|
| Genome features | Custom k-mer counter + BV-BRC gene API |
| ML model | XGBoost + scikit-learn CalibratedClassifierCV |
| Explainability | SHAP (TreeExplainer) |
| Growth modelling | SciPy curve_fit (logistic), linregress |
| Website | Streamlit 1.37 |
| Charts | Plotly 5.23 |

---

## References

1. **BV-BRC:** Olson RD et al. (2023). Introducing the Bacterial and Viral Bioinformatics Resource Center (BV-BRC). *Nucleic Acids Research*. https://doi.org/10.1093/nar/gkac1003
2. **CARD (resistance genes):** Alcock BP et al. (2023). CARD 2023: expanded curation, support for machine learning, and resistome prediction. *Nucleic Acids Research*. https://doi.org/10.1093/nar/gkac920
3. **XGBoost:** Chen T, Guestrin C (2016). XGBoost: A scalable tree boosting system. *KDD 2016*. https://doi.org/10.1145/2939672.2939785
4. **SHAP:** Lundberg SM, Lee S-I (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
5. **K-mer AMR:** Moradigaravand D et al. (2018). Prediction of antibiotic resistance in Escherichia coli from large-scale pan-genome data. *PLOS Computational Biology*. https://doi.org/10.1371/journal.pcbi.1006258
