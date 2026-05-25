"""
Retrain multi-organism models with XGBoost-only (no RF) to keep file sizes small.
Replaces the large SoftVotingEnsemble models with compact CalibratedClassifierCV(XGBoost).

Expected output: each model < 10 MB (vs 250-400 MB with RF).

Usage:
  /opt/anaconda3/envs/amr/bin/python src/retrain_small.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
ART_DIR   = ROOT / "artifacts"

MIN_SAMPLES = 100

ORGANISMS = json.loads((ART_DIR / "organisms.json").read_text())
# Remove K. pneumoniae — already trained separately
ORGANISMS = {k: v for k, v in ORGANISMS.items() if k != "klebsiella_pneumoniae"}


def train_organism(org_name: str, org_info: dict):
    print(f"\n{'='*60}")
    print(f"TRAINING: {org_info['display']}")
    print(f"{'='*60}")

    org_dir   = DATA_DIR / org_name
    model_dir = MODEL_DIR / org_name
    model_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(org_dir / "labels.csv", dtype={"genome_id": str})
    labels["antibiotic"] = labels["antibiotic"].str.lower().str.strip()
    labels = labels[labels["antibiotic"].isin(org_info["antibiotics"])]

    print("  Loading gene matrix...")
    genes = pd.read_csv(org_dir / "genes.csv", index_col=0, dtype={"genome_id": str})
    genes.index = genes.index.astype(str)

    results = []

    for ab in org_info["antibiotics"]:
        safe = ab.replace("/", "_").replace(" ", "_")
        model_path = model_dir / f"{safe}.pkl"

        sub = labels[labels["antibiotic"] == ab].copy()
        sub["y"] = (sub["resistant_phenotype"] == "Resistant").astype(int)
        sub = sub.drop_duplicates("genome_id").set_index("genome_id")

        common = sub.index.intersection(genes.index)
        if len(common) < MIN_SAMPLES:
            print(f"  SKIP {ab}: only {len(common)} samples")
            continue

        X = genes.loc[common].fillna(0).astype(float)
        y = sub.loc[common, "y"].values

        n_r, n_s = int(y.sum()), int((y == 0).sum())
        if n_r < 20 or n_s < 20:
            print(f"  SKIP {ab}: too imbalanced (R={n_r}, S={n_s})")
            continue

        print(f"\n  [{ab}]  n={len(y):,}  R={n_r:,}  S={n_s:,}")

        # 80/20 stratified split for AUC eval
        X_tr, X_te, y_tr, y_te = train_test_split(
            X.values, y, test_size=0.2, random_state=42, stratify=y
        )

        scale = max(n_s / max(n_r, 1), 1)
        xgb = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        model = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, probs)
        print(f"    Holdout AUC: {auc:.3f}")

        # Retrain on ALL data
        model_final = CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        )
        model_final.fit(X.values, y)

        bundle = {
            "model":         model_final,
            "features":      [f"gene__{c}" for c in X.columns],
            "antibiotic":    ab,
            "organism":      org_name,
            "test_auc":      round(auc, 3),
            "n_total":       len(y),
            "n_resistant":   n_r,
            "n_susceptible": n_s,
            "feature_type":  "gene_only",
        }

        with open(model_path, "wb") as f:
            pickle.dump(bundle, f, protocol=4)

        size_mb = model_path.stat().st_size / 1e6
        print(f"    Saved → {model_path}  ({size_mb:.1f} MB)")

        results.append({
            "organism":    org_name,
            "antibiotic":  ab,
            "test_auc":    round(auc, 3),
            "n_total":     len(y),
            "n_resistant": n_r,
        })

    return results


def main():
    all_results = []
    for org_name, org_info in ORGANISMS.items():
        res = train_organism(org_name, org_info)
        all_results.extend(res)

    # Update summary
    summary_path = ART_DIR / "multi_org_summary.json"
    trained_orgs = {r["organism"] for r in all_results}
    existing = [r for r in json.loads(summary_path.read_text())
                if r["organism"] not in trained_orgs]
    existing.extend(all_results)
    summary_path.write_text(json.dumps(existing, indent=2))

    print(f"\n{'='*60}")
    print("RETRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n{'Organism':<28} {'Antibiotic':<38} {'AUC':>6} {'N':>7}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: (x["organism"], -x["test_auc"])):
        print(f"{r['organism']:<28} {r['antibiotic']:<38} {r['test_auc']:>6.3f} {r['n_total']:>7,}")

    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
