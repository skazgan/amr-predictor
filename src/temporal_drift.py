"""
Temporal drift analysis.

Questions we answer:
  1. How does resistance prevalence change year-by-year per antibiotic?
  2. Does our model decay when trained on old data and tested on new?
  3. Which antibiotics are getting worse (more resistant) over time?
  4. Do resistance gene frequencies shift over time?

Steps:
  A. Fetch collection_year for all our genome IDs from BV-BRC
  B. Build yearly resistance profiles per antibiotic
  C. Train on pre-2020, test on 2020+ → measure temporal AUC decay
  D. Track gene frequency trends over time

Outputs:
  artifacts/temporal_years.json          — collection year per genome
  artifacts/temporal_prevalence.json     — yearly resistance rates per antibiotic
  artifacts/temporal_model_decay.json    — AUC: all-data vs temporal split
  artifacts/temporal_gene_trends.json    — gene frequency trend by year
"""

import json
import time
import sys
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from train import train

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

BV_BRC_API = "https://www.bv-brc.org/api"


# ── A. Fetch collection years ─────────────────────────────────────────────────

def fetch_collection_years(genome_ids: list[str],
                           cache_path: Path) -> dict[str, int]:
    """
    Fetch collection_year for a list of genome IDs from BV-BRC.
    Results are cached so we don't re-fetch on re-runs.
    """
    if cache_path.exists():
        print(f"  Loading cached years from {cache_path} ...")
        return json.loads(cache_path.read_text())

    print(f"  Fetching collection years for {len(genome_ids)} genomes ...")
    years = {}
    batch_size = 200
    batches = [genome_ids[i:i+batch_size]
               for i in range(0, len(genome_ids), batch_size)]

    for i, batch in enumerate(batches):
        ids_str = ",".join(batch)
        url = (f"{BV_BRC_API}/genome/"
               f"?in(genome_id,({ids_str}))"
               f"&select(genome_id,collection_year)"
               f"&limit({batch_size})")
        try:
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
            for rec in r.json():
                yr = rec.get("collection_year")
                if yr and isinstance(yr, (int, float)) and 2000 <= int(yr) <= 2030:
                    years[str(rec["genome_id"])] = int(yr)
        except Exception as e:
            print(f"  Batch {i} failed: {e}")
        if i % 10 == 0:
            print(f"  {i+1}/{len(batches)} batches done — {len(years)} years found")
        time.sleep(0.05)

    cache_path.write_text(json.dumps(years))
    print(f"  Done. {len(years)}/{len(genome_ids)} genomes have collection years.")
    return years


# ── B. Yearly resistance prevalence ──────────────────────────────────────────

def yearly_prevalence(ab: str, years: dict) -> list[dict]:
    """
    For one antibiotic, compute % resistant per year.
    Returns list of {year, n_resistant, n_susceptible, pct_resistant}.
    """
    safe = ab.replace("/","_").replace(" ","_")
    p = PROC_DIR / safe / "metadata.csv"
    if not p.exists():
        return []

    meta = pd.read_csv(p)
    meta["genome_id"] = meta["genome_id"].astype(str)
    meta = meta.drop_duplicates("genome_id")
    meta["year"] = meta["genome_id"].map(years)
    meta = meta[meta["year"].notna()].copy()
    meta["year"] = meta["year"].astype(int)

    rows = []
    for yr, grp in meta.groupby("year"):
        n_r = (grp["label"] == "R").sum()
        n_s = (grp["label"] == "S").sum()
        total = n_r + n_s
        if total >= 5:
            rows.append({
                "year": int(yr),
                "n_resistant": int(n_r),
                "n_susceptible": int(n_s),
                "total": int(total),
                "pct_resistant": round(float(n_r / total * 100), 1),
            })
    return sorted(rows, key=lambda x: x["year"])


# ── C. Temporal model decay ───────────────────────────────────────────────────

def temporal_model_decay(ab: str, years: dict,
                         X_kmer: pd.DataFrame,
                         X_gene: pd.DataFrame) -> dict:
    """
    Train on genomes collected before 2020, test on 2020+.
    Compare AUC vs training on random 80/20 split.
    """
    safe = ab.replace("/","_").replace(" ","_")
    p = PROC_DIR / safe / "metadata.csv"
    if not p.exists():
        return {}

    meta = pd.read_csv(p)
    meta["genome_id"] = meta["genome_id"].astype(str)
    meta = meta.drop_duplicates("genome_id")
    meta["year"] = meta["genome_id"].map(years).apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    meta = meta[meta["year"].notna()].copy()

    # Build combined feature matrix
    kmer_cols = [c for c in X_kmer.columns]
    gene_raw  = X_gene.copy()

    def make_X(genome_ids):
        idx = (pd.Index([str(g) for g in genome_ids])
                 .intersection(X_kmer.index)
                 .intersection(X_gene.index))
        return pd.concat([X_kmer.loc[idx], X_gene.loc[idx]], axis=1)

    label_map = dict(zip(meta["genome_id"], meta["label"].map({"R":1,"S":0})))

    # Temporal split: train <2020, test >=2020
    train_ids = meta[meta["year"] < 2020]["genome_id"].tolist()
    test_ids  = meta[meta["year"] >= 2020]["genome_id"].tolist()

    if len(train_ids) < 30 or len(test_ids) < 10:
        return {}

    X_train_t = make_X(train_ids)
    X_test_t  = make_X(test_ids)
    y_train_t = pd.Series({g: label_map[g] for g in X_train_t.index
                           if g in label_map}, name="label")
    y_test_t  = pd.Series({g: label_map[g] for g in X_test_t.index
                           if g in label_map}, name="label")
    X_train_t = X_train_t.loc[y_train_t.index]
    X_test_t  = X_test_t.loc[y_test_t.index]

    if len(y_train_t) < 20 or len(y_test_t) < 5:
        return {}

    # Feature selection on training set
    kmer_c = [c for c in X_train_t.columns if not c.startswith("gene__")]
    gene_c = [c for c in X_train_t.columns if c.startswith("gene__")]
    if len(kmer_c) > 0:
        sel = SelectKBest(mutual_info_classif, k=min(256, len(kmer_c)))
        sel.fit(X_train_t[kmer_c], y_train_t)
        top_k = [kmer_c[i] for i in sel.get_support(indices=True)]
    else:
        top_k = []
    keep = top_k + gene_c

    X_train_t = X_train_t[keep]
    X_test_t  = X_test_t[keep]

    # Train and evaluate — temporal split
    n_feat = min(512, len(keep))
    model_t, _ = train(X_train_t, y_train_t, model="xgb", n_features=n_feat)
    y_prob_t = model_t.predict_proba(X_test_t)[:, 1]
    auc_temporal = roc_auc_score(y_test_t, y_prob_t) if len(set(y_test_t)) > 1 else None

    # Random split baseline (same total size)
    all_ids = train_ids + test_ids
    X_all = make_X(all_ids)
    y_all = pd.Series({g: label_map[g] for g in X_all.index if g in label_map})
    X_all = X_all.loc[y_all.index][keep]

    if len(X_all) > 40:
        test_frac = len(test_ids) / len(all_ids)
        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
            X_all, y_all, test_size=test_frac, stratify=y_all, random_state=42
        )
        model_r, _ = train(X_tr_r, y_tr_r, model="xgb", n_features=n_feat)
        y_prob_r  = model_r.predict_proba(X_te_r)[:, 1]
        auc_random = roc_auc_score(y_te_r, y_prob_r) if len(set(y_te_r)) > 1 else None
    else:
        auc_random = None

    result = {
        "antibiotic":     ab,
        "n_train":        int(len(X_train_t)),
        "n_test":         int(len(X_test_t)),
        "train_years":    "< 2020",
        "test_years":     ">= 2020",
        "auc_temporal":   round(float(auc_temporal), 3) if auc_temporal else None,
        "auc_random":     round(float(auc_random),   3) if auc_random   else None,
        "drift":          round(float(auc_random - auc_temporal), 3)
                          if auc_temporal and auc_random else None,
    }

    decay_str = (f"drift={result['drift']:+.3f}" if result["drift"] else "n/a")
    print(f"  {ab:<35} temporal={result['auc_temporal']}  "
          f"random={result['auc_random']}  {decay_str}")
    return result


# ── D. Gene frequency trends ──────────────────────────────────────────────────

def gene_trends(years: dict, top_n: int = 10) -> list[dict]:
    """
    For the top MDR-enriched genes, track their frequency per year.
    """
    print("\nComputing gene frequency trends ...")
    gene_path = PROC_DIR / "gene_matrix.csv"
    if not gene_path.exists():
        return []

    X_gene = pd.read_csv(gene_path, index_col=0)
    X_gene.index = X_gene.index.astype(str)
    X_gene = X_gene.drop(columns=["__label__"], errors="ignore")

    # Only genomes with known year
    year_series = pd.Series(years)
    year_series = year_series[year_series.index.isin(X_gene.index)]
    year_series = year_series[year_series.astype(int).between(2010, 2024)]
    X_sub = X_gene.loc[year_series.index]

    # Pick top genes from MDR enrichment file
    mdr_genes_path = ART_DIR / "mdr_genes.json"
    if mdr_genes_path.exists():
        mdr_genes = json.loads(mdr_genes_path.read_text())
        focus_genes = [g["gene"] for g in mdr_genes[:top_n]
                       if g["gene"] in X_sub.columns]
    else:
        # Fallback: most variable genes
        focus_genes = X_sub.var().nlargest(top_n).index.tolist()

    results = []
    for gene in focus_genes:
        if gene not in X_sub.columns:
            continue
        gene_yr = pd.DataFrame({"present": X_sub[gene], "year": year_series.astype(int)})
        yearly  = gene_yr.groupby("year")["present"].agg(["mean","count"]).reset_index()
        yearly  = yearly[yearly["count"] >= 5]
        results.append({
            "gene":  gene,
            "trend": [
                {"year": int(r["year"]),
                 "frequency": round(float(r["mean"]) * 100, 1),
                 "n": int(r["count"])}
                for _, r in yearly.iterrows()
            ]
        })
    print(f"  Gene trends computed for {len(results)} genes.")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("TEMPORAL DRIFT ANALYSIS")
    print("="*60)

    # Collect all genome IDs
    all_ids = set()
    for ab in ANTIBIOTICS:
        safe = ab.replace("/","_").replace(" ","_")
        p = PROC_DIR / safe / "metadata.csv"
        if p.exists():
            df = pd.read_csv(p)
            all_ids.update(df["genome_id"].astype(str).tolist())
    print(f"\nTotal unique genomes: {len(all_ids)}")

    # A. Fetch years
    print("\n[A] Fetching collection years ...")
    years = fetch_collection_years(
        list(all_ids),
        cache_path=ART_DIR / "temporal_years.json"
    )

    # Coverage summary
    year_vals = list(years.values())
    print(f"  Year range: {min(year_vals)} – {max(year_vals)}")
    yr_counts = pd.Series(year_vals).value_counts().sort_index()
    print(f"  Genomes per year:\n{yr_counts.to_string()}")

    # B. Yearly prevalence
    print("\n[B] Computing yearly resistance prevalence ...")
    prevalence = {}
    for ab in ANTIBIOTICS:
        rows = yearly_prevalence(ab, years)
        prevalence[ab] = rows
        if rows:
            yr_range = f"{rows[0]['year']}–{rows[-1]['year']}"
            trend    = rows[-1]["pct_resistant"] - rows[0]["pct_resistant"]
            print(f"  {ab:<35} {yr_range}  trend={trend:+.1f}%")
    (ART_DIR / "temporal_prevalence.json").write_text(json.dumps(prevalence, indent=2))

    # C. Model decay
    print("\n[C] Temporal model decay (train <2020, test >=2020) ...")
    print(f"  {'Antibiotic':<35} {'Temporal AUC':>13} {'Random AUC':>11} {'Drift':>7}")
    print("  " + "-"*65)

    # Load shared matrices
    X_kmer = pd.read_csv(PROC_DIR / "X.csv", index_col=0)
    X_kmer.index = X_kmer.index.astype(str)
    gene_raw = pd.read_csv(PROC_DIR / "gene_matrix.csv", index_col=0)
    gene_raw.index = gene_raw.index.astype(str)
    X_gene = gene_raw.drop(columns=["__label__"], errors="ignore")
    X_gene.columns = ["gene__" + c for c in X_gene.columns]

    decay_results = []
    for ab in ANTIBIOTICS:
        res = temporal_model_decay(ab, years, X_kmer, X_gene)
        if res:
            decay_results.append(res)
    (ART_DIR / "temporal_model_decay.json").write_text(json.dumps(decay_results, indent=2))

    # D. Gene trends
    print("\n[D] Gene frequency trends over time ...")
    gene_tr = gene_trends(years)
    (ART_DIR / "temporal_gene_trends.json").write_text(json.dumps(gene_tr, indent=2))

    # Summary
    print("\n" + "="*60)
    print("DONE — artifacts saved to artifacts/")
    print("="*60)
    if decay_results:
        worst = max((r for r in decay_results if r.get("drift")),
                    key=lambda x: x["drift"], default=None)
        best  = min((r for r in decay_results if r.get("drift")),
                    key=lambda x: x["drift"], default=None)
        if worst:
            print(f"  Worst drift : {worst['antibiotic']} ({worst['drift']:+.3f})")
        if best:
            print(f"  Best stable : {best['antibiotic']} ({best['drift']:+.3f})")

    # Fastest-growing resistance
    trends = []
    for ab, rows in prevalence.items():
        if len(rows) >= 3:
            trend = rows[-1]["pct_resistant"] - rows[0]["pct_resistant"]
            trends.append((ab, trend))
    if trends:
        trends.sort(key=lambda x: -x[1])
        print(f"\n  Fastest-rising resistance : {trends[0][0]} (+{trends[0][1]:.1f}%)")
        print(f"  Most stable/declining     : {trends[-1][0]} ({trends[-1][1]:+.1f}%)")


if __name__ == "__main__":
    main()
