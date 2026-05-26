"""
Fetch MIC (minimum inhibitory concentration) data from BV-BRC and train
ordinal MIC predictors for K. pneumoniae.

MIC prediction goes beyond binary R/S — it predicts the actual concentration
at which bacteria are inhibited, which informs dosing decisions.

Output:
  artifacts/mic_distributions.json  — MIC value distributions per antibiotic
  artifacts/mic_breakpoints.json    — EUCAST/CLSI breakpoints
  models/mic_{antibiotic}.pkl       — XGBoost ordinal classifier per antibiotic

Run:
  python src/train_mic.py --fetch     # fetch only
  python src/train_mic.py --train     # fetch + train
"""
import argparse
import json
import pickle
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
ART_DIR   = ROOT / "artifacts"

BASE_URL = "https://www.bv-brc.org/api"
HEADERS  = {"Accept": "application/json"}

# Target antibiotics (those with enough data and matching gene matrix)
TARGET_ABS = [
    "meropenem", "imipenem", "ciprofloxacin", "gentamicin",
    "amikacin", "piperacillin/tazobactam", "trimethoprim/sulfamethoxazole",
    "ceftazidime", "ceftriaxone", "tetracycline",
]

# EUCAST 2024 clinical breakpoints for K. pneumoniae (mg/L)
# Format: (susceptible_breakpoint, resistant_breakpoint)
# S ≤ breakpoint_s, R > breakpoint_r
EUCAST_BREAKPOINTS = {
    "meropenem":                    (2.0,  8.0),
    "imipenem":                     (2.0,  8.0),
    "ciprofloxacin":                (0.25, 1.0),
    "gentamicin":                   (2.0,  4.0),
    "amikacin":                     (8.0,  16.0),
    "piperacillin/tazobactam":      (8.0,  16.0),
    "trimethoprim/sulfamethoxazole": (2.0,  4.0),
    "ceftazidime":                  (1.0,  4.0),
    "ceftriaxone":                  (1.0,  2.0),
    "tetracycline":                 (1.0,  8.0),
    "levofloxacin":                 (1.0,  2.0),
    "cefepime":                     (1.0,  4.0),
}

# MIC categories (3-class ordinal problem)
# Low = likely susceptible, Intermediate = uncertain/dose-dependent, High = likely resistant
MIC_CATEGORIES = ["Low", "Intermediate", "High"]


def bvbrc_get(endpoint, params, limit=5000, offset=0):
    url = f"{BASE_URL}/{endpoint}/?{params}&limit({limit},{offset})"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return []


def fetch_all_mic(taxon_id: int = 573) -> pd.DataFrame:
    """Fetch all MIC records for K. pneumoniae from BV-BRC."""
    print(f"Fetching MIC data for taxon {taxon_id}...")
    params = (
        f"eq(taxon_id,{taxon_id})"
        f"&eq(evidence,Laboratory%20Method)"
        f"&select(genome_id,antibiotic,resistant_phenotype,measurement,measurement_unit,measurement_sign)"
    )
    all_rows, offset = [], 0
    while True:
        batch = bvbrc_get("genome_amr", params, limit=5000, offset=offset)
        if not batch:
            break
        all_rows.extend(batch)
        print(f"  fetched {len(all_rows):,} records...", end="\r")
        if len(batch) < 5000:
            break
        offset += 5000
        time.sleep(0.3)
    print()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.columns = df.columns.str.lower()

    # Keep only records with actual numeric measurements
    df = df[df["measurement"].notna() & (df["measurement"] != "")].copy()
    df["antibiotic"] = df["antibiotic"].str.lower().str.strip()
    df["mic_value"] = pd.to_numeric(df["measurement"], errors="coerce")
    df = df.dropna(subset=["mic_value"])
    df = df[df["mic_value"] > 0]

    print(f"  Total usable MIC records: {len(df):,}")
    for ab in sorted(df["antibiotic"].unique()):
        sub = df[df["antibiotic"] == ab]
        n_r = (sub["resistant_phenotype"] == "Resistant").sum()
        n_s = (sub["resistant_phenotype"] == "Susceptible").sum()
        if n_r + n_s >= 50:
            print(f"    {ab:<40} n={len(sub):>5,}  R={n_r:>4,}  S={n_s:>4,}  "
                  f"MIC: {sub['mic_value'].min():.3f}–{sub['mic_value'].max():.0f}")
    return df


def build_distributions(df: pd.DataFrame) -> dict:
    """Build MIC distribution statistics per antibiotic for visualization."""
    dist = {}
    for ab in TARGET_ABS:
        sub = df[df["antibiotic"] == ab].copy()
        if len(sub) < 20:
            continue
        log_mics = np.log2(sub["mic_value"].clip(lower=1e-4))
        bp_s, bp_r = EUCAST_BREAKPOINTS.get(ab, (1.0, 4.0))
        dist[ab] = {
            "n": int(len(sub)),
            "n_resistant": int((sub["resistant_phenotype"] == "Resistant").sum()),
            "n_susceptible": int((sub["resistant_phenotype"] == "Susceptible").sum()),
            "mic_values": [round(v, 4) for v in sub["mic_value"].tolist()],
            "mic_labels": sub["resistant_phenotype"].tolist(),
            "log2_mic_values": [round(v, 3) for v in log_mics.tolist()],
            "eucast_s": bp_s,
            "eucast_r": bp_r,
            "mic_50": round(float(sub["mic_value"].median()), 3),
            "mic_90": round(float(sub["mic_value"].quantile(0.9)), 3),
            "mic_min": round(float(sub["mic_value"].min()), 4),
            "mic_max": round(float(sub["mic_value"].max()), 1),
        }
    return dist


def mic_to_category(mic_value: float, bp_s: float, bp_r: float) -> int:
    """Convert MIC value to ordinal category: 0=Low, 1=Intermediate, 2=High."""
    if mic_value <= bp_s:
        return 0  # Low (susceptible range)
    elif mic_value <= bp_r:
        return 1  # Intermediate
    else:
        return 2  # High (resistant range)


def train_mic_models(df: pd.DataFrame, genes: pd.DataFrame) -> list:
    """Train ordinal XGBoost MIC classifiers for each antibiotic."""
    results = []
    mic_model_dir = MODEL_DIR / "mic"
    mic_model_dir.mkdir(parents=True, exist_ok=True)

    for ab in TARGET_ABS:
        safe = ab.replace("/", "_").replace(" ", "_")
        model_path = mic_model_dir / f"{safe}.pkl"

        sub = df[df["antibiotic"] == ab].copy()
        if len(sub) < 100:
            print(f"  SKIP {ab}: only {len(sub)} MIC records")
            continue

        bp_s, bp_r = EUCAST_BREAKPOINTS.get(ab, (1.0, 4.0))

        # Create 3-class ordinal target
        sub["mic_category"] = sub["mic_value"].apply(
            lambda v: mic_to_category(v, bp_s, bp_r)
        )

        sub = sub.drop_duplicates("genome_id").set_index("genome_id")
        common = sub.index.intersection(genes.index)

        if len(common) < 50:
            print(f"  SKIP {ab}: only {len(common)} genomes with gene data")
            continue

        X = genes.loc[common].fillna(0).astype(float)
        y = sub.loc[common, "mic_category"].values

        counts = Counter(y)
        print(f"\n  [{ab}]  n={len(y):,}  "
              f"Low={counts.get(0,0):,}  Inter={counts.get(1,0):,}  High={counts.get(2,0):,}")

        if min(counts.values()) < 5:
            print(f"    SKIP: insufficient samples in one category")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X.values, y, test_size=0.2, random_state=42, stratify=y
        )

        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )
        xgb.fit(X_tr, y_tr)
        probs_te = xgb.predict_proba(X_te)

        # Per-class AUC (one-vs-rest)
        from sklearn.metrics import roc_auc_score
        auc_scores = []
        for c in range(3):
            if (y_te == c).sum() > 0 and (y_te != c).sum() > 0:
                auc_scores.append(roc_auc_score(y_te == c, probs_te[:, c]))
        mean_auc = round(np.mean(auc_scores), 3) if auc_scores else 0.0
        print(f"    Mean one-vs-rest AUC: {mean_auc:.3f}")

        # Final model on ALL data
        xgb_final = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )
        xgb_final.fit(X.values, y)

        bundle = {
            "model":       xgb_final,
            "features":    [f"gene__{c}" for c in X.columns],
            "antibiotic":  ab,
            "categories":  MIC_CATEGORIES,
            "bp_s":        bp_s,
            "bp_r":        bp_r,
            "mean_auc":    mean_auc,
            "n_total":     len(y),
        }
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f, protocol=4)
        print(f"    Saved → {model_path}  ({model_path.stat().st_size/1e6:.1f} MB)")

        results.append({"antibiotic": ab, "mean_auc": mean_auc, "n": len(y)})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch",  action="store_true", help="Fetch MIC data from BV-BRC")
    parser.add_argument("--train",  action="store_true", help="Also train MIC models")
    parser.add_argument("--cached", action="store_true", help="Use cached mic_data.csv if available")
    args = parser.parse_args()

    mic_cache = DATA_DIR / "mic_data.csv"

    # ── Fetch ─────────────────────────────────────────────────────────────────
    if args.cached and mic_cache.exists():
        print(f"Using cached MIC data: {mic_cache}")
        df_mic = pd.read_csv(mic_cache, dtype={"genome_id": str})
    elif args.fetch or args.train:
        df_mic = fetch_all_mic(taxon_id=573)
        if df_mic.empty:
            print("No MIC data fetched. Exiting.")
            return
        df_mic.to_csv(mic_cache, index=False)
        print(f"Cached → {mic_cache}")
    else:
        print("No action specified. Use --fetch or --train.")
        return

    # ── Build distribution artifact ───────────────────────────────────────────
    dist = build_distributions(df_mic)
    bp_out = {ab: {"eucast_s": s, "eucast_r": r} for ab, (s, r) in EUCAST_BREAKPOINTS.items()}
    (ART_DIR / "mic_distributions.json").write_text(json.dumps(dist, indent=2))
    (ART_DIR / "mic_breakpoints.json").write_text(json.dumps(bp_out, indent=2))
    print(f"\nSaved MIC distributions for {len(dist)} antibiotics → artifacts/mic_distributions.json")

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.train:
        print("\nLoading gene matrix...")
        gm = pd.read_csv(DATA_DIR / "gene_matrix.csv", index_col=0)
        gm.index = gm.index.astype(str)
        gm = gm.drop(columns=["__label__"], errors="ignore")

        print(f"Gene matrix: {gm.shape}")
        results = train_mic_models(df_mic, gm)

        print(f"\n{'Antibiotic':<40} {'Mean AUC':>10} {'N':>7}")
        print("-" * 60)
        for r in sorted(results, key=lambda x: -x["mean_auc"]):
            print(f"{r['antibiotic']:<40} {r['mean_auc']:>10.3f} {r['n']:>7,}")


if __name__ == "__main__":
    main()
