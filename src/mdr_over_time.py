"""
MDR over time analysis.

Combines temporal drift + co-resistance data to answer:
  1. How has the fraction of MDR strains (3+ drugs) changed year-by-year?
  2. Is total MDR load increasing, stable, or decreasing?
  3. Which antibiotic combination clusters are driving MDR growth?
  4. Are 2-drug resistant strains graduating to 3-drug over time?

Outputs:
  artifacts/mdr_over_time.json   — yearly MDR prevalence + resistance burden distributions
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import linregress

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


def load_all_labels() -> pd.DataFrame:
    """Load and merge resistance labels for all antibiotics."""
    dfs = []
    for ab in ANTIBIOTICS:
        safe = ab.replace("/", "_").replace(" ", "_")
        p = PROC_DIR / safe / "metadata.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["genome_id"] = df["genome_id"].astype(str)
        df = df.drop_duplicates("genome_id")
        df = df[["genome_id", "label"]].copy()
        df = df.rename(columns={"label": ab})
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="genome_id", how="outer")
    return merged


def main():
    print("=" * 60)
    print("MDR OVER TIME ANALYSIS")
    print("=" * 60)

    # Load collection years
    years_path = ART_DIR / "temporal_years.json"
    if not years_path.exists():
        print("ERROR: Run temporal_drift.py first.")
        sys.exit(1)
    years = json.loads(years_path.read_text())
    year_series = pd.Series({k: int(v) for k, v in years.items()})
    year_series = year_series[year_series.between(2000, 2024)]

    # Load all labels
    print("\nLoading resistance labels ...")
    labels = load_all_labels()
    print(f"  {len(labels)} genomes with at least one antibiotic label")

    # Merge with years
    labels["genome_id"] = labels["genome_id"].astype(str)
    labels["year"] = labels["genome_id"].map(year_series)
    labels = labels[labels["year"].notna()].copy()
    labels["year"] = labels["year"].astype(int)
    print(f"  {len(labels)} genomes with year data")

    # Convert R/S labels to binary (R=1, S=0, missing=NaN)
    for ab in ANTIBIOTICS:
        if ab in labels.columns:
            labels[ab + "_bin"] = labels[ab].map({"R": 1, "S": 0})

    bin_cols = [ab + "_bin" for ab in ANTIBIOTICS if ab + "_bin" in labels.columns]

    # Count drugs resistant per genome (of those tested)
    labels["n_resistant"] = labels[bin_cols].sum(axis=1, min_count=1)
    labels["n_tested"]    = labels[bin_cols].notna().sum(axis=1)
    labels["is_mdr"]      = labels["n_resistant"] >= 3

    print(f"\nOverall MDR summary:")
    total = len(labels)
    mdr_count = labels["is_mdr"].sum()
    print(f"  Total genomes: {total}")
    print(f"  MDR (3+ drugs): {mdr_count} ({100*mdr_count/total:.1f}%)")

    # ── A. MDR prevalence per year ────────────────────────────────────────────
    print("\n[A] MDR prevalence by year ...")
    yearly_mdr = []
    for yr, grp in labels.groupby("year"):
        n_total   = len(grp)
        n_mdr     = grp["is_mdr"].sum()
        n_tested2plus = (grp["n_tested"] >= 2).sum()
        if n_total < 5:
            continue
        yearly_mdr.append({
            "year":        int(yr),
            "n_total":     int(n_total),
            "n_mdr":       int(n_mdr),
            "pct_mdr":     round(float(n_mdr / n_total * 100), 1),
            "mean_drugs_resistant": round(float(grp["n_resistant"].mean()), 3),
        })

    yearly_mdr.sort(key=lambda x: x["year"])
    print(f"  {'Year':>6} {'N genomes':>10} {'N MDR':>8} {'% MDR':>8} {'Avg drugs R':>12}")
    print("  " + "-" * 50)
    for row in yearly_mdr:
        print(f"  {row['year']:>6} {row['n_total']:>10} {row['n_mdr']:>8} "
              f"{row['pct_mdr']:>7.1f}% {row['mean_drugs_resistant']:>12.2f}")

    # Trend in MDR
    if len(yearly_mdr) >= 4:
        yrs  = [r["year"] for r in yearly_mdr]
        pcts = [r["pct_mdr"] for r in yearly_mdr]
        slope, _, r, p, _ = linregress(yrs, pcts)
        print(f"\n  MDR trend: {slope:+.2f}%/yr (R²={r**2:.3f}, p={p:.4f})")

    # ── B. Resistance burden distribution over time ───────────────────────────
    print("\n[B] Resistance burden distribution (0-6 drugs) by year ...")
    burden_by_year = []
    for yr, grp in labels.groupby("year"):
        if len(grp) < 5:
            continue
        row = {"year": int(yr), "n_total": int(len(grp))}
        for n in range(0, 7):
            cnt = int((grp["n_resistant"] == n).sum())
            row[f"n_{n}drugs"] = cnt
            row[f"pct_{n}drugs"] = round(float(cnt / len(grp) * 100), 1)
        burden_by_year.append(row)

    # ── C. Per-drug resistance % over time (for stacked area chart) ──────────
    print("\n[C] Per-antibiotic resistance over time ...")
    per_drug_by_year = {}
    for ab in ANTIBIOTICS:
        bc = ab + "_bin"
        if bc not in labels.columns:
            continue
        rows = []
        for yr, grp in labels.groupby("year"):
            sub = grp[grp[bc].notna()]
            if len(sub) < 5:
                continue
            pct_r = float(sub[bc].mean() * 100)
            rows.append({"year": int(yr), "pct_resistant": round(pct_r, 1), "n": int(len(sub))})
        rows.sort(key=lambda x: x["year"])
        per_drug_by_year[ab] = rows

    # ── D. MDR combination prevalence over time ───────────────────────────────
    print("\n[D] Tracking specific resistance combinations over time ...")
    combos_of_interest = [
        ("Cipro+TMP/SMX+Cef",   ["ciprofloxacin", "trimethoprim/sulfamethoxazole", "cefepime"]),
        ("Gent+TMP/SMX+Cef",    ["gentamicin", "trimethoprim/sulfamethoxazole", "cefepime"]),
        ("Cipro+Gent+TMP/SMX",  ["ciprofloxacin", "gentamicin", "trimethoprim/sulfamethoxazole"]),
        ("Pan-resistant (all 6)",ANTIBIOTICS),
    ]
    combo_trends = {}
    for name, drugs in combos_of_interest:
        bin_cols_c = [d + "_bin" for d in drugs if d + "_bin" in labels.columns]
        if len(bin_cols_c) < len(drugs):
            continue
        rows = []
        for yr, grp in labels.groupby("year"):
            sub = grp[grp[bin_cols_c].notna().all(axis=1)]
            if len(sub) < 5:
                continue
            n_combo = int((sub[bin_cols_c] == 1).all(axis=1).sum())
            rows.append({"year": int(yr), "pct": round(n_combo / len(sub) * 100, 2), "n": len(sub)})
        rows.sort(key=lambda x: x["year"])
        combo_trends[name] = rows
        if rows:
            print(f"  {name}: {rows[0]['pct']:.1f}% → {rows[-1]['pct']:.1f}%")

    # ── Save all results ──────────────────────────────────────────────────────
    out = {
        "yearly_mdr":      yearly_mdr,
        "burden_by_year":  burden_by_year,
        "per_drug_by_year": per_drug_by_year,
        "combo_trends":    combo_trends,
        "antibiotics":     ANTIBIOTICS,
        "short_names":     SHORT,
    }
    (ART_DIR / "mdr_over_time.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved to artifacts/mdr_over_time.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if yearly_mdr:
        first = yearly_mdr[0]
        last  = yearly_mdr[-1]
        change = last["pct_mdr"] - first["pct_mdr"]
        print(f"\n  MDR in {first['year']}: {first['pct_mdr']:.1f}%")
        print(f"  MDR in {last['year']}:  {last['pct_mdr']:.1f}%")
        print(f"  Change:  {change:+.1f}%")
        print(f"\n  {'Rising' if change > 0 else 'Falling'} MDR trend over 24 years")


if __name__ == "__main__":
    main()
