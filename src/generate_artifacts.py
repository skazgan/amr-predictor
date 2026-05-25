"""
Retrain all 6 calibrated models and save visualisation artifacts for the website:
  - models/<antibiotic>.pkl           (calibrated model bundle)
  - artifacts/roc_<antibiotic>.json   (ROC curve points)
  - artifacts/cm_<antibiotic>.json    (confusion matrix)
  - artifacts/fi_<antibiotic>.json    (top-20 feature importances)
  - artifacts/summary.json            (AUC table across all antibiotics)
  - artifacts/dataset_stats.json      (class balance, genome counts)

Usage:
    python src/generate_artifacts.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from train import train

ROOT      = Path(__file__).parent.parent
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
ART_DIR   = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

ANTIBIOTICS = [
    "ciprofloxacin",
    "meropenem",
    "gentamicin",
    "tetracycline",
    "trimethoprim/sulfamethoxazole",
    "cefepime",
    "amikacin",
    "imipenem",
    "piperacillin/tazobactam",
    "levofloxacin",
]

DRUG_CLASS = {
    "ciprofloxacin":                "Fluoroquinolone",
    "meropenem":                    "Carbapenem",
    "gentamicin":                   "Aminoglycoside",
    "tetracycline":                 "Tetracycline",
    "trimethoprim/sulfamethoxazole":"Folate inhibitor",
    "cefepime":                     "Cephalosporin",
    "amikacin":                     "Aminoglycoside",
    "imipenem":                     "Carbapenem",
    "piperacillin/tazobactam":      "Beta-lactam/inhibitor",
    "levofloxacin":                 "Fluoroquinolone",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_shared():
    print("Loading shared feature matrices …")
    X_kmer = pd.read_csv(PROC_DIR / "X.csv", index_col=0)
    X_kmer.index = X_kmer.index.astype(str)

    gene_raw = pd.read_csv(PROC_DIR / "gene_matrix.csv", index_col=0)
    gene_raw.index = gene_raw.index.astype(str)
    X_gene = gene_raw.drop(columns=["__label__"], errors="ignore")
    X_gene.columns = ["gene__" + c for c in X_gene.columns]

    print(f"  k-mer {X_kmer.shape}  |  gene {X_gene.shape}")
    return X_kmer, X_gene


def build_combined(X_kmer, X_gene, genome_ids):
    idx = (pd.Index(genome_ids)
             .intersection(X_kmer.index)
             .intersection(X_gene.index))
    return pd.concat([X_kmer.loc[idx], X_gene.loc[idx]], axis=1)


def safe_name(ab):
    return ab.replace("/", "_").replace(" ", "_")


def feature_importances(model_bundle, feature_names):
    """
    Extract feature importances from the calibrated pipeline.
    CalibratedClassifierCV wraps N base estimators — average their importances.
    """
    pipeline = model_bundle["model"]
    calibrated = pipeline.named_steps["clf"]          # CalibratedClassifierCV
    selector   = pipeline.named_steps["select"]

    # selected feature names (after SelectKBest)
    sel_mask      = selector.get_support()
    selected_cols = [feature_names[i] for i, m in enumerate(sel_mask) if m]

    importances = []
    for est in calibrated.calibrated_classifiers_:
        base = est.estimator                          # XGBClassifier
        if hasattr(base, "feature_importances_"):
            importances.append(base.feature_importances_)

    if not importances:
        return []

    mean_imp = np.mean(importances, axis=0)
    pairs    = sorted(zip(selected_cols, mean_imp), key=lambda x: -x[1])
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs[:20]]


# ── Per-antibiotic training + artifact generation ─────────────────────────────

def process_antibiotic(ab, X_kmer, X_gene, summary_rows, dataset_stats):
    print(f"\n{'='*60}")
    print(f"  {ab.upper()}")
    print(f"{'='*60}")

    # Labels
    meta_path = PROC_DIR / safe_name(ab) / "metadata.csv"
    if not meta_path.exists():
        meta_path = PROC_DIR / ab / "metadata.csv"
    if not meta_path.exists():
        print("  No metadata — skipping.")
        return

    meta = pd.read_csv(meta_path)
    meta["genome_id"] = meta["genome_id"].astype(str)
    label_map = dict(zip(meta["genome_id"],
                         meta["label"].map({"R": 1, "S": 0})))

    # Dataset stats
    n_r = sum(v == 1 for v in label_map.values())
    n_s = sum(v == 0 for v in label_map.values())
    dataset_stats.append({
        "antibiotic": ab,
        "drug_class": DRUG_CLASS[ab],
        "n_resistant": n_r,
        "n_susceptible": n_s,
        "total": n_r + n_s,
    })

    # Feature matrix
    genome_ids = list(label_map.keys())
    X = build_combined(X_kmer, X_gene, genome_ids)
    y = pd.Series({gid: label_map[gid] for gid in X.index}, name="label")

    if len(X) < 50:
        print("  Not enough genomes — skipping.")
        return

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Feature selection (k-mers only)
    kmer_cols = [c for c in X_train.columns if not c.startswith("gene__")]
    gene_cols  = [c for c in X_train.columns if c.startswith("gene__")]
    sel = SelectKBest(mutual_info_classif, k=min(256, len(kmer_cols)))
    sel.fit(X_train[kmer_cols], y_train)
    top_kmers = [kmer_cols[i] for i in sel.get_support(indices=True)]
    keep = top_kmers + gene_cols
    X_train, X_test = X_train[keep], X_test[keep]

    # Train calibrated model
    n_feat = min(512, len(keep))
    model, scores = train(X_train, y_train, model="xgb", n_features=n_feat)
    cv_auc = scores["test_roc_auc"].mean()

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"  CV AUC: {cv_auc:.3f}  |  Test AUC: {test_auc:.3f}")

    # ── Save model ─────────────────────────────────────────────────────────
    bundle = {
        "model":      model,
        "antibiotic": ab,
        "features":   keep,
        "cv_auc":     cv_auc,
        "test_auc":   test_auc,
    }
    with open(MODEL_DIR / f"{safe_name(ab)}.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # ── ROC curve ──────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data = {
        "fpr":      [round(float(v), 4) for v in fpr],
        "tpr":      [round(float(v), 4) for v in tpr],
        "auc":      round(test_auc, 4),
        "antibiotic": ab,
    }
    (ART_DIR / f"roc_{safe_name(ab)}.json").write_text(json.dumps(roc_data))

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_data = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "antibiotic": ab,
    }
    report = classification_report(y_test, y_pred,
                                   target_names=["Susceptible", "Resistant"],
                                   output_dict=True)
    cm_data["report"] = report
    (ART_DIR / f"cm_{safe_name(ab)}.json").write_text(json.dumps(cm_data))

    # ── Feature importances ────────────────────────────────────────────────
    fi = feature_importances(bundle, keep)
    (ART_DIR / f"fi_{safe_name(ab)}.json").write_text(json.dumps(fi))

    # ── Summary row ────────────────────────────────────────────────────────
    summary_rows.append({
        "antibiotic":  ab,
        "drug_class":  DRUG_CLASS[ab],
        "n_genomes":   len(X),
        "cv_auc":      round(cv_auc, 3),
        "test_auc":    round(test_auc, 3),
        "accuracy":    round(report["accuracy"], 3),
        "precision_r": round(report["Resistant"]["precision"], 3),
        "recall_r":    round(report["Resistant"]["recall"], 3),
        "f1_r":        round(report["Resistant"]["f1-score"], 3),
    })


def main():
    X_kmer, X_gene = load_shared()

    summary_rows = []
    dataset_stats = []

    for ab in ANTIBIOTICS:
        process_antibiotic(ab, X_kmer, X_gene, summary_rows, dataset_stats)

    # Global summary JSON
    (ART_DIR / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    (ART_DIR / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2))

    print(f"\n{'='*60}")
    print("ALL DONE — artifacts saved to artifacts/")
    print(f"{'='*60}")
    print(f"{'Antibiotic':<35} {'CV AUC':>8} {'Test AUC':>10} {'Accuracy':>10}")
    print("-" * 65)
    for r in sorted(summary_rows, key=lambda x: -x["test_auc"]):
        print(f"{r['antibiotic']:<35} {r['cv_auc']:>8.3f} "
              f"{r['test_auc']:>10.3f} {r['accuracy']:>10.3f}")


if __name__ == "__main__":
    main()
