"""
Train AMR resistance models for multiple organisms.

For each organism × antibiotic combination:
  - Loads gene presence/absence matrix + AMR labels
  - Trains XGBoost + Random Forest soft-voting ensemble
  - Saves model bundle to models/{organism}/{antibiotic}.pkl

Usage:
  python src/train_organisms.py
  python src/train_organisms.py --organism escherichia_coli
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
ART_DIR   = ROOT / "artifacts"

MIN_SAMPLES = 100   # minimum per class to train


# ── Ensemble (same as train_multi.py) ─────────────────────────────────────────
class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_weight: float = 0.6, n_estimators: int = 300):
        self.xgb_weight    = xgb_weight
        self.n_estimators  = n_estimators
        self.classes_      = np.array([0, 1])

    def fit(self, X, y):
        scale = max((y == 0).sum() / max((y == 1).sum(), 1), 1)
        self.xgb_ = CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        )
        self.rf_ = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=self.n_estimators, max_features="sqrt",
                min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        )
        self.xgb_.fit(X, y)
        self.rf_.fit(X, y)
        return self

    def predict_proba(self, X):
        p_xgb = self.xgb_.predict_proba(X)
        p_rf  = self.rf_.predict_proba(X)
        return self.xgb_weight * p_xgb + (1 - self.xgb_weight) * p_rf

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_organism(org_name: str, org_info: dict):
    print(f"\n{'='*60}")
    print(f"TRAINING: {org_info['display']}")
    print(f"{'='*60}")

    org_dir   = DATA_DIR / org_name
    model_dir = MODEL_DIR / org_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_path = org_dir / "labels.csv"
    if not labels_path.exists():
        print(f"  SKIP: no labels.csv (run fetch_multi_organism.py first)")
        return []

    df_labels = pd.read_csv(labels_path, dtype={"genome_id": str})
    df_labels["antibiotic"] = df_labels["antibiotic"].str.lower().str.strip()
    df_labels = df_labels[df_labels["antibiotic"].isin(org_info["antibiotics"])]

    # Load gene matrix
    genes_path = org_dir / "genes.csv"
    if not genes_path.exists():
        print(f"  SKIP: no genes.csv (run fetch_multi_organism.py first)")
        return []

    print(f"  Loading gene matrix...")
    df_genes = pd.read_csv(genes_path, index_col=0, dtype={"genome_id": str})
    df_genes.index = df_genes.index.astype(str)

    results = []

    for ab in org_info["antibiotics"]:
        safe = ab.replace("/", "_").replace(" ", "_")
        model_path = model_dir / f"{safe}.pkl"

        # Get R/S labels for this antibiotic
        sub = df_labels[df_labels["antibiotic"] == ab].copy()
        sub["y"] = (sub["resistant_phenotype"] == "Resistant").astype(int)
        sub = sub.drop_duplicates("genome_id").set_index("genome_id")

        # Intersect with gene matrix
        common = sub.index.intersection(df_genes.index)
        if len(common) < MIN_SAMPLES:
            print(f"  SKIP {ab}: only {len(common)} samples (need {MIN_SAMPLES})")
            continue

        X = df_genes.loc[common].fillna(0).astype(float)
        y = sub.loc[common, "y"].values

        n_r = y.sum()
        n_s = len(y) - n_r
        if n_r < 20 or n_s < 20:
            print(f"  SKIP {ab}: class too imbalanced (R={n_r}, S={n_s})")
            continue

        print(f"\n  [{ab}]  n={len(y):,}  R={n_r:,}  S={n_s:,}")

        # 5-fold CV AUC
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            oof_probs = cross_val_predict(
                SoftVotingEnsemble(), X, y, cv=cv, method="predict_proba"
            )[:, 1]
            auc = roc_auc_score(y, oof_probs)
            print(f"    CV AUC: {auc:.3f}")
        except Exception as e:
            print(f"    CV failed: {e}")
            auc = 0.0

        # Train final model on all data
        model = SoftVotingEnsemble()
        model.fit(X.values, y)

        bundle = {
            "model":        model,
            "features":     [f"gene__{c}" for c in X.columns],
            "antibiotic":   ab,
            "organism":     org_name,
            "test_auc":     round(auc, 3),
            "n_total":      len(y),
            "n_resistant":  int(n_r),
            "n_susceptible": int(n_s),
            "feature_type": "gene_only",
        }

        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"    Saved → {model_path}")

        results.append({
            "organism":    org_name,
            "antibiotic":  ab,
            "test_auc":    round(auc, 3),
            "n_total":     len(y),
            "n_resistant": int(n_r),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organism", default=None,
                        help="Train a specific organism only (e.g. escherichia_coli)")
    args = parser.parse_args()

    # Load organism registry
    registry_path = ART_DIR / "organisms.json"
    if not registry_path.exists():
        print("ERROR: run fetch_multi_organism.py first to create organisms.json")
        return

    registry = json.loads(registry_path.read_text())

    # Skip K. pneumoniae — already trained separately
    orgs_to_train = {
        k: v for k, v in registry.items()
        if k != "klebsiella_pneumoniae"
        and (args.organism is None or k == args.organism)
    }

    all_results = []
    for org_name, org_info in orgs_to_train.items():
        res = train_organism(org_name, org_info)
        all_results.extend(res)

    # Save multi-organism summary
    summary_path = ART_DIR / "multi_org_summary.json"
    existing = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    # Replace entries for trained organisms
    trained_orgs = {r["organism"] for r in all_results}
    existing = [r for r in existing if r["organism"] not in trained_orgs]
    existing.extend(all_results)
    summary_path.write_text(json.dumps(existing, indent=2))

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n{'Organism':<30} {'Antibiotic':<35} {'AUC':>6} {'N':>7}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: (x["organism"], -x["test_auc"])):
        print(f"{r['organism']:<30} {r['antibiotic']:<35} {r['test_auc']:>6.3f} {r['n_total']:>7,}")

    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
