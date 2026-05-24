"""
Gene emergence curve analysis.

For each resistance gene in our dataset, we compute:
  - First year it was detected (emergence year)
  - Year-by-year frequency (% of genomes carrying it)
  - Spread rate (slope of logistic/linear fit)
  - Whether it is still accelerating or plateauing

This is analogous to an epidemic curve — but for resistance genes
spreading through the bacterial population over 2000–2024.

Outputs:
  artifacts/gene_emergence.json   — full emergence data per gene
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
ART_DIR  = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def logistic(x, L, k, x0):
    """Logistic growth curve: L / (1 + exp(-k*(x-x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))


def fit_growth_model(years: list, freqs: list) -> dict:
    """
    Fit both linear and logistic models.
    Return whichever fits better (lower residual), plus key parameters.
    """
    x = np.array(years, dtype=float)
    y = np.array(freqs, dtype=float) / 100.0  # work in 0-1 scale

    result = {"model": "linear", "slope": 0.0, "r2": 0.0,
              "plateau": None, "emergence_speed": "slow"}

    if len(x) < 4:
        return result

    # Linear fit
    slope, intercept, r, p, se = linregress(x, y)
    r2_linear = r ** 2
    result["slope"]    = round(float(slope * 100), 3)   # % per year
    result["r2"]       = round(float(r2_linear), 3)
    result["p_value"]  = round(float(p), 4)

    # Logistic fit (only if enough data and rising trend)
    if slope > 0 and len(x) >= 6:
        try:
            L_init  = max(y) * 1.2
            k_init  = 0.3
            x0_init = float(np.median(x))
            popt, _ = curve_fit(
                logistic, x, y,
                p0=[L_init, k_init, x0_init],
                bounds=([0, 0, 2000], [1.0, 5, 2030]),
                maxfev=5000,
            )
            y_pred     = logistic(x, *popt)
            ss_res     = np.sum((y - y_pred) ** 2)
            ss_tot     = np.sum((y - y.mean()) ** 2)
            r2_logistic = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2_logistic > r2_linear:
                result["model"]     = "logistic"
                result["r2"]        = round(float(r2_logistic), 3)
                result["plateau"]   = round(float(popt[0] * 100), 1)  # L in %
                result["midpoint"]  = round(float(popt[2]), 1)        # inflection year
                result["growth_k"]  = round(float(popt[1]), 3)
        except Exception:
            pass

    # Classify spread speed
    slope_pct = result["slope"]
    result["emergence_speed"] = (
        "rapid"    if slope_pct > 2.0 else
        "moderate" if slope_pct > 0.5 else
        "slow"     if slope_pct > 0   else
        "declining"
    )

    return result


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENE EMERGENCE CURVE ANALYSIS")
    print("=" * 60)

    # Load gene matrix
    gene_path = PROC_DIR / "gene_matrix.csv"
    print("\nLoading gene matrix ...")
    X_gene = pd.read_csv(gene_path, index_col=0)
    X_gene.index = X_gene.index.astype(str)
    X_gene = X_gene.drop(columns=["__label__"], errors="ignore")
    print(f"  {X_gene.shape[0]} genomes × {X_gene.shape[1]} genes")

    # Load collection years (already cached)
    years_path = ART_DIR / "temporal_years.json"
    if not years_path.exists():
        print("ERROR: Run temporal_drift.py first to cache collection years.")
        sys.exit(1)
    years = json.loads(years_path.read_text())
    year_series = pd.Series({k: int(v) for k, v in years.items()})
    year_series = year_series[year_series.between(2000, 2024)]

    # Intersect with gene matrix
    common = X_gene.index.intersection(year_series.index)
    X_sub  = X_gene.loc[common]
    y_yrs  = year_series.loc[common]
    print(f"  Genomes with year + gene data: {len(common)}")

    # Focus on the most prevalent and interesting genes
    # Include: top MDR genes + most variable + clinically known
    mdr_path = ART_DIR / "mdr_genes.json"
    if mdr_path.exists():
        mdr_genes  = [g["gene"] for g in json.loads(mdr_path.read_text())]
    else:
        mdr_genes = []

    # Also include genes with high overall prevalence (>5%)
    prevalence_all = X_sub.mean()
    prevalent = prevalence_all[prevalence_all > 0.05].index.tolist()

    # Union of MDR genes + prevalent genes, capped at 40
    focus = list(dict.fromkeys(mdr_genes + prevalent))[:40]
    focus = [g for g in focus if g in X_sub.columns]
    print(f"  Analysing {len(focus)} genes")

    # ── Per-gene emergence curves ─────────────────────────────────────────────
    results = []
    for gene in focus:
        gene_data = pd.DataFrame({
            "present": X_sub[gene].astype(float),
            "year":    y_yrs,
        })

        yearly = (gene_data
                  .groupby("year")["present"]
                  .agg(["sum","count"])
                  .reset_index()
                  .rename(columns={"sum":"n_with_gene","count":"n_total"}))
        yearly = yearly[yearly["n_total"] >= 5].copy()
        yearly["frequency"] = yearly["n_with_gene"] / yearly["n_total"] * 100

        if len(yearly) < 3:
            continue

        # Emergence year = first year with ≥ 1% frequency
        emerging = yearly[yearly["frequency"] >= 1.0]
        emergence_year = int(emerging["year"].min()) if not emerging.empty else None

        # Fit growth model
        growth = fit_growth_model(
            yearly["year"].tolist(),
            yearly["frequency"].tolist()
        )

        # Peak year and current frequency
        peak_row = yearly.loc[yearly["frequency"].idxmax()]
        last_row = yearly.iloc[-1]

        results.append({
            "gene":            gene,
            "emergence_year":  emergence_year,
            "peak_year":       int(peak_row["year"]),
            "peak_frequency":  round(float(peak_row["frequency"]), 1),
            "current_frequency": round(float(last_row["frequency"]), 1),
            "current_year":    int(last_row["year"]),
            "overall_prevalence": round(float(prevalence_all.get(gene, 0) * 100), 1),
            "growth_model":    growth,
            "yearly": [
                {
                    "year":      int(r["year"]),
                    "frequency": round(float(r["frequency"]), 1),
                    "n_total":   int(r["n_total"]),
                }
                for _, r in yearly.iterrows()
            ],
        })

    # Sort by emergence speed then peak frequency
    results.sort(key=lambda x: (
        -{"rapid":3,"moderate":2,"slow":1,"declining":0}.get(
            x["growth_model"]["emergence_speed"], 0),
        -x["peak_frequency"]
    ))

    (ART_DIR / "gene_emergence.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {len(results)} genes")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Gene':<45} {'Emerged':>8} {'Speed':>10} {'Current%':>10} {'Slope%/yr':>10}")
    print("-" * 85)
    for r in results[:15]:
        print(f"{r['gene'][:44]:<45} "
              f"{str(r['emergence_year'] or '?'):>8} "
              f"{r['growth_model']['emergence_speed']:>10} "
              f"{r['current_frequency']:>9.1f}% "
              f"{r['growth_model']['slope']:>+9.2f}%")

    rapid = [r for r in results if r["growth_model"]["emergence_speed"] == "rapid"]
    declining = [r for r in results if r["growth_model"]["emergence_speed"] == "declining"]
    print(f"\nRapidly spreading genes   : {len(rapid)}")
    print(f"Declining genes           : {len(declining)}")
    if rapid:
        print(f"Fastest spreading         : {rapid[0]['gene'][:50]}")
        print(f"  slope = {rapid[0]['growth_model']['slope']:+.2f}%/yr, "
              f"emerged {rapid[0]['emergence_year']}")


if __name__ == "__main__":
    main()
