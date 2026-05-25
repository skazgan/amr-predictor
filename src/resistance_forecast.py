"""
Resistance forecasting — project future resistance rates.

Uses temporal prevalence data (2000–present) to project forward using:
  1. Linear extrapolation with confidence intervals (simple, transparent)
  2. Logistic saturation model (can't exceed 100%)

For each antibiotic, we output:
  - Predicted % resistant for the next 5 years
  - 80% confidence interval (based on residual variance)
  - Which model was selected (best R²)
  - Whether the forecast suggests the 50% "epidemic threshold" will be crossed

Outputs:
  artifacts/resistance_forecast.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress, t as t_dist

ROOT    = Path(__file__).parent.parent
ART_DIR = ROOT / "artifacts"
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

CURRENT_YEAR   = datetime.now().year
FORECAST_START = CURRENT_YEAR + 1           # start forecasting from next year
FORECAST_END   = FORECAST_START + 5         # 5-year horizon
FORECAST_YEARS = list(range(FORECAST_START, FORECAST_END + 1))


def logistic(x, L, k, x0):
    """Logistic: L / (1 + exp(-k*(x-x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))


def linear_forecast(years, values, forecast_years):
    """
    Fit linear model, extrapolate with prediction intervals.
    Returns: forecast list and model info dict.
    """
    x = np.array(years, dtype=float)
    y = np.array(values, dtype=float)
    n = len(x)

    slope, intercept, r, p, se = linregress(x, y)
    y_pred_in = slope * x + intercept
    residuals  = y - y_pred_in
    mse        = np.sum(residuals**2) / max(n - 2, 1)
    x_mean     = x.mean()
    ss_x       = np.sum((x - x_mean)**2)

    # t-critical for 80% CI (10% each tail)
    t_crit = t_dist.ppf(0.90, df=max(n-2, 1))

    forecasts = []
    for fy in forecast_years:
        pred = slope * fy + intercept
        pred = float(np.clip(pred, 0, 100))
        # Prediction interval: SE * t * sqrt(1 + 1/n + (x*-xmean)^2/ss_x)
        se_pred = np.sqrt(mse * (1 + 1/n + (fy - x_mean)**2 / max(ss_x, 1e-9)))
        margin  = float(t_crit * se_pred)
        forecasts.append({
            "year": fy,
            "predicted": round(pred, 1),
            "lower_80": round(float(np.clip(pred - margin, 0, 100)), 1),
            "upper_80": round(float(np.clip(pred + margin, 0, 100)), 1),
        })

    r2 = float(r ** 2)
    return forecasts, {
        "model":     "linear",
        "slope":     round(float(slope), 3),
        "intercept": round(float(intercept), 3),
        "r2":        round(r2, 3),
        "p_value":   round(float(p), 4),
        "rmse":      round(float(np.sqrt(mse)), 2),
    }


def logistic_forecast(years, values, forecast_years):
    """
    Fit logistic model. Falls back to linear on failure.
    """
    x = np.array(years, dtype=float)
    y = np.array(values, dtype=float)

    try:
        L_init  = min(max(y) * 1.5, 100)
        k_init  = 0.15
        x0_init = float(np.median(x))
        popt, pcov = curve_fit(
            logistic, x, y,
            p0=[L_init, k_init, x0_init],
            bounds=([0, 0.01, 1990], [100, 2, 2040]),
            maxfev=8000,
        )
        L, k, x0 = popt
        y_pred = logistic(x, *popt)
        residuals = y - y_pred
        mse = float(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-9))

        # Bootstrap CI for forecasts
        n_boot = 200
        boot_preds = {fy: [] for fy in forecast_years}
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            idx = rng.choice(len(x), len(x), replace=True)
            try:
                p_boot, _ = curve_fit(
                    logistic, x[idx], y[idx],
                    p0=popt,
                    bounds=([0, 0.01, 1990], [100, 2, 2040]),
                    maxfev=3000,
                )
                for fy in forecast_years:
                    boot_preds[fy].append(float(np.clip(logistic(fy, *p_boot), 0, 100)))
            except Exception:
                pass

        forecasts = []
        for fy in forecast_years:
            pred = float(np.clip(logistic(fy, *popt), 0, 100))
            bp = boot_preds[fy]
            if len(bp) >= 10:
                lower = float(np.percentile(bp, 10))
                upper = float(np.percentile(bp, 90))
            else:
                lower = max(0, pred - 5)
                upper = min(100, pred + 5)
            forecasts.append({
                "year": fy,
                "predicted": round(pred, 1),
                "lower_80": round(lower, 1),
                "upper_80": round(upper, 1),
            })

        return forecasts, {
            "model":    "logistic",
            "L":        round(float(L), 1),
            "k":        round(float(k), 4),
            "x0":       round(float(x0), 1),
            "r2":       round(r2, 3),
            "rmse":     round(float(np.sqrt(mse)), 2),
        }
    except Exception as e:
        return None, None


def main():
    print("=" * 60)
    print(f"RESISTANCE FORECASTING {FORECAST_START}–{FORECAST_END}")
    print("=" * 60)

    # Load historical prevalence
    prev_path = ART_DIR / "temporal_prevalence.json"
    if not prev_path.exists():
        print("ERROR: Run temporal_drift.py first.")
        sys.exit(1)

    prevalence = json.loads(prev_path.read_text())

    results = []
    for ab in ANTIBIOTICS:
        rows = prevalence.get(ab, [])
        if len(rows) < 5:
            print(f"  {ab}: insufficient data ({len(rows)} years)")
            continue

        years  = [r["year"] for r in rows]
        values = [r["pct_resistant"] for r in rows]

        print(f"\n  [{SHORT[ab]}] {ab}")
        print(f"    Historical: {years[0]}–{years[-1]}, "
              f"{values[0]:.1f}% → {values[-1]:.1f}%")

        # Fit both models
        lin_fc, lin_info = linear_forecast(years, values, FORECAST_YEARS)
        log_fc, log_info = logistic_forecast(years, values, FORECAST_YEARS)

        # Select better model (higher R²)
        if log_info and log_info["r2"] > lin_info["r2"] + 0.02:
            best_fc   = log_fc
            best_info = log_info
        else:
            best_fc   = lin_fc
            best_info = lin_info

        # Detect 50% threshold crossing
        threshold_cross = None
        for fc in best_fc:
            if fc["predicted"] >= 50:
                threshold_cross = fc["year"]
                break

        print(f"    Model: {best_info['model']} (R²={best_info['r2']:.3f})")
        print(f"    Forecast {FORECAST_START}: {best_fc[0]['predicted']:.1f}% "
              f"[{best_fc[0]['lower_80']:.1f}–{best_fc[0]['upper_80']:.1f}%]")
        print(f"    Forecast 2030: {best_fc[-1]['predicted']:.1f}% "
              f"[{best_fc[-1]['lower_80']:.1f}–{best_fc[-1]['upper_80']:.1f}%]")
        if threshold_cross:
            print(f"    ⚠️  Will cross 50% threshold around {threshold_cross}")
        else:
            print(f"    Stays below 50% through 2030")

        results.append({
            "antibiotic":       ab,
            "short_name":       SHORT[ab],
            "historical":       [{"year": y, "pct_resistant": v}
                                 for y, v in zip(years, values)],
            "forecast":         best_fc,
            "linear_forecast":  lin_fc,
            "logistic_forecast": log_fc,
            "model_info":       best_info,
            "threshold_50_year": threshold_cross,
            "current_pct":      values[-1],
            "current_year":     years[-1],
        })

    output = {
        "forecast_start": FORECAST_START,
        "forecast_end":   FORECAST_END,
        "generated_year": CURRENT_YEAR,
        "results":        results,
    }
    (ART_DIR / "resistance_forecast.json").write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {len(results)} antibiotics forecasted")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — 2030 PROJECTIONS")
    print("=" * 60)
    print(f"\n{'Antibiotic':<35} {'Current':>9} {'2030 pred':>10} {'CI':>20} {'Model':>10}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["forecast"][-1]["predicted"]):
        fc = r["forecast"][-1]
        print(f"{r['antibiotic'][:34]:<35} {r['current_pct']:>8.1f}% "
              f"{fc['predicted']:>9.1f}% "
              f"[{fc['lower_80']:.1f}–{fc['upper_80']:.1f}%]:>18 "
              f"{r['model_info']['model']:>10}")

    crossings = [r for r in results if r["threshold_50_year"]]
    print(f"\n  Antibiotics projected to exceed 50% resistance: {len(crossings)}")
    for r in crossings:
        print(f"    {r['antibiotic']}: ~{r['threshold_50_year']}")


if __name__ == "__main__":
    main()
