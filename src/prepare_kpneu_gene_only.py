"""
Prepare K. pneumoniae data in multi-organism format and train gene-only models.

Uses the existing per-antibiotic metadata files + gene_matrix.csv.
Output: models/klebsiella_pneumoniae/{antibiotic}.pkl

Run:
  python src/prepare_kpneu_gene_only.py
"""
import json
import pickle
import shutil
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

# Antibiotic directory name → canonical antibiotic name
AB_MAP = {
    "amikacin":                       "amikacin",
    "cefepime":                        "cefepime",
    "ciprofloxacin":                   "ciprofloxacin",
    "gentamicin":                      "gentamicin",
    "imipenem":                        "imipenem",
    "levofloxacin":                    "levofloxacin",
    "meropenem":                       "meropenem",
    "piperacillin_tazobactam":         "piperacillin/tazobactam",
    "tetracycline":                    "tetracycline",
    "trimethoprim_sulfamethoxazole":   "trimethoprim/sulfamethoxazole",
}

MIN_SAMPLES = 50


def prepare_labels() -> pd.DataFrame:
    """Merge per-antibiotic metadata files into unified labels.csv format."""
    print("Preparing K. pneumoniae labels...")
    rows = []
    for dir_name, ab_name in AB_MAP.items():
        meta_path = DATA_DIR / dir_name / "metadata.csv"
        if not meta_path.exists():
            print(f"  SKIP {dir_name} — metadata.csv not found")
            continue
        meta = pd.read_csv(meta_path, index_col=0)
        # index = genome_id, columns = genome_name, label (0/1)
        for gid, row in meta.iterrows():
            rows.append({
                "genome_id":           str(gid),
                "antibiotic":          ab_name,
                "resistant_phenotype": "Resistant" if str(row["label"]).strip() in ("1", "R", "Resistant") else "Susceptible",
            })

    df = pd.DataFrame(rows)
    print(f"  Total label rows: {len(df):,}")
    for ab in df["antibiotic"].unique():
        sub = df[df["antibiotic"] == ab]
        n_r = (sub["resistant_phenotype"] == "Resistant").sum()
        n_s = (sub["resistant_phenotype"] == "Susceptible").sum()
        print(f"    {ab:<40} R={n_r:>5,}  S={n_s:>5,}")
    return df


def prepare_genes() -> pd.DataFrame:
    """Load K. pneumoniae gene_matrix.csv — already in the right format."""
    print("\nPreparing K. pneumoniae gene matrix...")
    gm_path = DATA_DIR / "gene_matrix.csv"
    gm = pd.read_csv(gm_path, index_col=0)
    gm.index = gm.index.astype(str)
    # Drop the __label__ column if present
    gm = gm.drop(columns=["__label__"], errors="ignore")
    print(f"  Gene matrix: {gm.shape[0]:,} genomes × {gm.shape[1]:,} genes")
    return gm


def train_kpneu(labels: pd.DataFrame, genes: pd.DataFrame):
    """Train one XGBoost gene-only model per antibiotic."""
    org_name  = "klebsiella_pneumoniae"
    model_dir = MODEL_DIR / org_name
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for ab in sorted(labels["antibiotic"].unique()):
        safe = ab.replace("/", "_").replace(" ", "_")
        model_path = model_dir / f"{safe}.pkl"

        sub = labels[labels["antibiotic"] == ab].copy()
        sub = sub.drop_duplicates("genome_id").set_index("genome_id")
        sub["y"] = (sub["resistant_phenotype"] == "Resistant").astype(int)

        common = sub.index.intersection(genes.index)
        if len(common) < MIN_SAMPLES:
            print(f"\n  SKIP {ab}: only {len(common)} samples with gene data")
            continue

        X = genes.loc[common].fillna(0).astype(float)
        y = sub.loc[common, "y"].values
        n_r = int(y.sum())
        n_s = int((y == 0).sum())

        if n_r < 20 or n_s < 20:
            print(f"\n  SKIP {ab}: too imbalanced (R={n_r}, S={n_s})")
            continue

        print(f"\n  [{ab}]  n={len(y):,}  R={n_r:,}  S={n_s:,}")

        # 80/20 stratified split for AUC evaluation
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

        # Final model on ALL data
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


def update_summary(new_results: list):
    """Merge K. pneumoniae results into multi_org_summary.json."""
    summary_path = ART_DIR / "multi_org_summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    # Remove old K. pneu entries
    existing = [r for r in existing if r["organism"] != "klebsiella_pneumoniae"]
    existing.extend(new_results)
    summary_path.write_text(json.dumps(existing, indent=2))
    print(f"\nUpdated multi_org_summary.json ({len(new_results)} K. pneu entries added)")


def main():
    print("=" * 60)
    print("K. PNEUMONIAE GENE-ONLY MODEL TRAINING")
    print("=" * 60)

    labels = prepare_labels()
    genes  = prepare_genes()

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    results = train_kpneu(labels, genes)

    update_summary(results)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\n{'Antibiotic':<38} {'AUC':>6} {'N':>7}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: -x["test_auc"]):
        print(f"{r['antibiotic']:<38} {r['test_auc']:>6.3f} {r['n_total']:>7,}")


if __name__ == "__main__":
    main()
