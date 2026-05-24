"""
Co-resistance network analysis.

For each genome that appears in multiple antibiotic datasets, build a
resistance profile vector and analyse:
  1. Pairwise co-resistance rates (phi coefficient)
  2. Multi-drug resistance (MDR) prevalence
  3. Genes enriched in MDR vs single-drug-resistant strains
  4. Cross-antibiotic prediction boost

Outputs:
  artifacts/coresistance_matrix.json   — 6x6 phi coefficient matrix
  artifacts/coresistance_counts.json   — raw co-occurrence counts
  artifacts/mdr_genes.json             — genes enriched in MDR strains
  artifacts/mdr_prevalence.json        — MDR prevalence stats
  artifacts/cross_prediction.json      — AUC with/without label features
"""

import json
import sys
import pickle
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
ART_DIR  = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

ANTIBIOTICS = [
    "ciprofloxacin",
    "meropenem",
    "gentamicin",
    "tetracycline",
    "trimethoprim/sulfamethoxazole",
    "cefepime",
]

SHORT = {
    "ciprofloxacin":                "Cipro",
    "meropenem":                    "Mero",
    "gentamicin":                   "Gent",
    "tetracycline":                 "Tet",
    "trimethoprim/sulfamethoxazole":"TMP/SMX",
    "cefepime":                     "Cef",
}


# ── Step 1: Build resistance profile matrix ────────────────────────────────────

def load_labels() -> pd.DataFrame:
    """
    Load labels for all antibiotics and merge into one profile matrix.
    Returns DataFrame: rows=genome_id, cols=antibiotics, values=0/1 (S/R).
    Only includes genomes present in at least 2 antibiotic datasets.
    """
    dfs = []
    for ab in ANTIBIOTICS:
        safe = ab.replace("/","_").replace(" ","_")
        p = PROC_DIR / safe / "metadata.csv"
        if not p.exists():
            p = PROC_DIR / ab / "metadata.csv"
        if not p.exists():
            print(f"  Missing metadata for {ab}")
            continue
        df = pd.read_csv(p)[["genome_id","label"]]
        df["genome_id"] = df["genome_id"].astype(str)
        df = df.drop_duplicates("genome_id")   # remove any duplicate genome IDs
        df = df.rename(columns={"label": ab})
        df[ab] = df[ab].map({"R": 1, "S": 0})
        dfs.append(df.set_index("genome_id"))

    profile = pd.concat(dfs, axis=1, join="outer")  # outer = keep all genomes
    # Only keep genomes with labels for 2+ antibiotics
    profile = profile[profile.notna().sum(axis=1) >= 2]
    print(f"Genomes with 2+ antibiotic labels: {len(profile)}")
    print(f"Coverage per antibiotic:\n{profile.notna().sum().to_string()}\n")
    return profile


# ── Step 2: Pairwise co-resistance (phi coefficient) ──────────────────────────

def phi_coefficient(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Compute the phi coefficient (like Pearson's r but for binary variables)
    and its chi-squared p-value.
    Returns (phi, p_value).
    """
    # Drop rows where either is NaN
    mask = x.notna() & y.notna()
    x, y = x[mask].astype(int), y[mask].astype(int)
    if len(x) < 10:
        return 0.0, 1.0
    ct = pd.crosstab(x, y)
    if ct.shape != (2, 2):
        return 0.0, 1.0
    chi2, p, _, _ = chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    phi = np.sqrt(chi2 / n) * np.sign(
        ct.iloc[1,1] * ct.iloc[0,0] - ct.iloc[1,0] * ct.iloc[0,1]
    )
    return float(phi), float(p)


def build_correlation_matrix(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 6x6 phi coefficient matrix and p-value matrix."""
    abs_list = [ab for ab in ANTIBIOTICS if ab in profile.columns]
    phi_mat  = pd.DataFrame(np.eye(len(abs_list)), index=abs_list, columns=abs_list)
    pval_mat = pd.DataFrame(np.zeros((len(abs_list), len(abs_list))),
                             index=abs_list, columns=abs_list)

    for ab1, ab2 in combinations(abs_list, 2):
        phi, p = phi_coefficient(profile[ab1], profile[ab2])
        phi_mat.loc[ab1, ab2] = phi
        phi_mat.loc[ab2, ab1] = phi
        pval_mat.loc[ab1, ab2] = p
        pval_mat.loc[ab2, ab1] = p
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {SHORT[ab1]:8s} × {SHORT[ab2]:8s}  φ={phi:+.3f}  p={p:.2e} {sig}")

    return phi_mat, pval_mat


# ── Step 3: MDR prevalence and gene enrichment ────────────────────────────────

def mdr_analysis(profile: pd.DataFrame) -> dict:
    """
    Define MDR as resistant to 3+ antibiotics.
    Compute prevalence and return breakdown.
    """
    n_resistant = profile.sum(axis=1)  # how many antibiotics each genome resists
    stats = {
        "total_genomes": int(len(profile)),
        "susceptible_all": int((n_resistant == 0).sum()),
        "single_resistant": int((n_resistant == 1).sum()),
        "double_resistant": int((n_resistant == 2).sum()),
        "mdr_3plus": int((n_resistant >= 3).sum()),
        "mdr_4plus": int((n_resistant >= 4).sum()),
        "mdr_5plus": int((n_resistant >= 5).sum()),
        "pan_resistant": int((n_resistant == 6).sum()),
        "pct_mdr": round((n_resistant >= 3).mean() * 100, 1),
        "mean_resistances": round(float(n_resistant.mean()), 2),
        "distribution": n_resistant.value_counts().sort_index().to_dict(),
    }
    print(f"\nMDR Analysis:")
    print(f"  Total genomes        : {stats['total_genomes']}")
    print(f"  Susceptible to all   : {stats['susceptible_all']}")
    print(f"  MDR (3+ antibiotics) : {stats['mdr_3plus']} ({stats['pct_mdr']}%)")
    print(f"  Pan-resistant (all 6): {stats['pan_resistant']}")
    return stats


def gene_enrichment(profile: pd.DataFrame) -> list[dict]:
    """
    Compare gene presence in MDR (3+) vs non-MDR strains.
    Returns genes most enriched in MDR.
    """
    print("\nLoading gene matrix for enrichment analysis ...")
    gene_path = PROC_DIR / "gene_matrix.csv"
    if not gene_path.exists():
        print("  gene_matrix.csv not found — skipping enrichment.")
        return []

    X_gene = pd.read_csv(gene_path, index_col=0)
    X_gene.index = X_gene.index.astype(str)
    X_gene = X_gene.drop(columns=["__label__"], errors="ignore")

    # MDR labels
    n_resistant = profile.sum(axis=1).reindex(X_gene.index).dropna()
    mdr_label   = (n_resistant >= 3).astype(int)

    # Keep only genomes present in both
    common = X_gene.index.intersection(mdr_label.index)
    X_sub  = X_gene.loc[common]
    y_sub  = mdr_label.loc[common]

    print(f"  Genomes for enrichment: {len(common)} "
          f"(MDR={y_sub.sum()}, non-MDR={(y_sub==0).sum()})")

    # For each gene, compute rate in MDR vs non-MDR
    mdr_idx  = y_sub[y_sub == 1].index
    nmdr_idx = y_sub[y_sub == 0].index

    results = []
    for gene in X_sub.columns:
        rate_mdr  = X_sub.loc[mdr_idx,  gene].mean()
        rate_nmdr = X_sub.loc[nmdr_idx, gene].mean()
        enrichment = rate_mdr - rate_nmdr
        if abs(enrichment) > 0.05 and rate_mdr + rate_nmdr > 0.02:
            results.append({
                "gene":        gene,
                "rate_mdr":    round(float(rate_mdr),  3),
                "rate_nmdr":   round(float(rate_nmdr), 3),
                "enrichment":  round(float(enrichment), 3),
            })

    results = sorted(results, key=lambda x: -abs(x["enrichment"]))[:30]
    print(f"  Top enriched genes found: {len(results)}")
    if results:
        print(f"  Most MDR-enriched: {results[0]['gene']} "
              f"(+{results[0]['enrichment']:.3f})")
    return results


# ── Step 4: Cross-antibiotic prediction boost ─────────────────────────────────

def cross_prediction_boost(profile: pd.DataFrame) -> list[dict]:
    """
    For each antibiotic, test whether knowing resistance to the other 5
    improves prediction AUC over a simple baseline.
    Uses a Random Forest on just the 5 label features (no genomic data).
    This measures pure label-to-label predictability.
    """
    print("\nCross-antibiotic prediction test ...")
    results = []
    abs_list = [ab for ab in ANTIBIOTICS if ab in profile.columns]

    for target in abs_list:
        others = [ab for ab in abs_list if ab != target]
        # Use pairwise dropna — only require target + each other individually
        # then stack them as features (fill missing with -1 = unknown)
        sub = profile[[target] + others].copy()
        sub = sub[sub[target].notna()]   # must have the target label
        if len(sub) < 50:
            continue
        # Fill unknown other-antibiotic labels with -1
        X = sub[others].fillna(-1)
        y = sub[target].astype(int)

        # Baseline: predict majority class
        baseline = max(y.mean(), 1 - y.mean())

        # RF on other labels only
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        scores = cross_validate(rf, X, y, cv=cv, scoring="roc_auc")
        auc = scores["test_score"].mean()

        print(f"  {SHORT[target]:8s}  baseline={baseline:.3f}  "
              f"label-only AUC={auc:.3f}  boost=+{auc-0.5:.3f}")
        results.append({
            "antibiotic": target,
            "baseline_auc": round(baseline, 3),
            "label_only_auc": round(float(auc), 3),
            "boost": round(float(auc - 0.5), 3),
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("CO-RESISTANCE NETWORK ANALYSIS")
    print("="*60)

    # 1. Build profile matrix
    print("\n[1] Building resistance profile matrix ...")
    profile = load_labels()

    # 2. Pairwise correlations
    print("\n[2] Computing pairwise co-resistance (phi coefficient) ...")
    phi_mat, pval_mat = build_correlation_matrix(profile)

    phi_export = {
        "antibiotics": list(phi_mat.columns),
        "short_names": [SHORT[ab] for ab in phi_mat.columns],
        "phi_matrix":  phi_mat.values.tolist(),
        "pval_matrix": pval_mat.values.tolist(),
    }
    (ART_DIR / "coresistance_matrix.json").write_text(json.dumps(phi_export, indent=2))

    # Raw co-occurrence counts
    counts = []
    abs_list = [ab for ab in ANTIBIOTICS if ab in profile.columns]
    for ab1, ab2 in combinations(abs_list, 2):
        mask = profile[[ab1, ab2]].notna().all(axis=1)
        sub  = profile.loc[mask, [ab1, ab2]].astype(int)
        n_both_r  = int(((sub[ab1]==1) & (sub[ab2]==1)).sum())
        n_only_ab1 = int(((sub[ab1]==1) & (sub[ab2]==0)).sum())
        n_only_ab2 = int(((sub[ab1]==0) & (sub[ab2]==1)).sum())
        n_both_s  = int(((sub[ab1]==0) & (sub[ab2]==0)).sum())
        counts.append({
            "ab1": ab1, "ab2": ab2,
            "both_resistant": n_both_r,
            "only_ab1_resistant": n_only_ab1,
            "only_ab2_resistant": n_only_ab2,
            "both_susceptible": n_both_s,
            "total": int(mask.sum()),
        })
    (ART_DIR / "coresistance_counts.json").write_text(json.dumps(counts, indent=2))

    # 3. MDR prevalence
    print("\n[3] MDR prevalence analysis ...")
    mdr_stats = mdr_analysis(profile)
    (ART_DIR / "mdr_prevalence.json").write_text(json.dumps(mdr_stats, indent=2))

    # 4. Gene enrichment
    print("\n[4] Gene enrichment in MDR strains ...")
    mdr_genes = gene_enrichment(profile)
    (ART_DIR / "mdr_genes.json").write_text(json.dumps(mdr_genes, indent=2))

    # 5. Cross-prediction boost
    print("\n[5] Cross-antibiotic prediction boost ...")
    cross_pred = cross_prediction_boost(profile)
    (ART_DIR / "cross_prediction.json").write_text(json.dumps(cross_pred, indent=2))

    # Summary
    print("\n" + "="*60)
    print("DONE — all artifacts saved to artifacts/")
    print("="*60)
    print(f"\nKey findings:")
    phi_vals = [(ANTIBIOTICS[i], ANTIBIOTICS[j], phi_mat.iloc[i,j])
                for i in range(len(abs_list)) for j in range(i+1, len(abs_list))]
    phi_vals.sort(key=lambda x: -abs(x[2]))
    print(f"  Strongest co-resistance: {SHORT[phi_vals[0][0]]} × "
          f"{SHORT[phi_vals[0][1]]}  φ={phi_vals[0][2]:+.3f}")
    print(f"  Weakest co-resistance:   {SHORT[phi_vals[-1][0]]} × "
          f"{SHORT[phi_vals[-1][1]]}  φ={phi_vals[-1][2]:+.3f}")
    print(f"  MDR prevalence (3+):     {mdr_stats['pct_mdr']}%")
    if mdr_genes:
        print(f"  Top MDR gene:            {mdr_genes[0]['gene']}")
    if cross_pred:
        best = max(cross_pred, key=lambda x: x["label_only_auc"])
        print(f"  Best cross-prediction:   {SHORT[best['antibiotic']]} "
              f"predicted from others AUC={best['label_only_auc']:.3f}")


if __name__ == "__main__":
    main()
