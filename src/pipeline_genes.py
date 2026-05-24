"""
Pipeline using resistance gene presence/absence features (CARD / BV-BRC).

Usage:
    python src/pipeline_genes.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gene_features import build_gene_matrix
from train import train
from evaluate import plot_roc, plot_confusion

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CACHE = PROC_DIR / "gene_matrix.csv"


def main():
    # ── 1. Build gene presence/absence matrix ─────────────────────────────────
    X, y = build_gene_matrix(
        metadata_path=PROC_DIR / "metadata.csv",
        cache_path=CACHE,
    )

    print(f"\nFeature matrix : {X.shape[0]} genomes × {X.shape[1]} genes")
    print(f"Class balance  : {y.value_counts().to_dict()}")

    # Drop genomes where no genes were detected at all (API returned nothing)
    has_data = X.sum(axis=1) > 0
    n_dropped = (~has_data).sum()
    if n_dropped:
        print(f"Dropping {n_dropped} genomes with zero gene annotations.")
        X, y = X[has_data], y[has_data]

    # ── 2. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain / test split: {len(X_train)} train, {len(X_test)} test")

    # ── 3. Train both models (no feature-count tuning — matrix is already small)
    results = {}
    for model_name in ["rf", "xgb"]:
        n_feat = min(X_train.shape[1], 256)    # cap at 256 or total genes
        print(f"\n{'='*50}")
        model, _ = train(X_train, y_train, model=model_name, n_features=n_feat)

        model_path = MODEL_DIR / f"{model_name}_genes.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[model_name] = (model, y_pred, y_prob, auc)

        print(f"\n--- Hold-out Test Report [{model_name.upper()} / genes] ---")
        print(classification_report(y_test, y_pred, target_names=["Susceptible", "Resistant"]))
        print(f"ROC-AUC : {auc:.3f}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("SUMMARY (gene features)")
    print(f"{'Model':<8} {'Test AUC':>10}")
    for name, (_, _, _, auc) in results.items():
        print(f"{name.upper():<8} {auc:>10.3f}")

    best_name = max(results, key=lambda k: results[k][3])
    _, best_pred, best_prob, best_auc = results[best_name]
    print(f"\nBest model: {best_name.upper()} (AUC {best_auc:.3f})")
    plot_roc(y_test, best_prob, save_path=str(MODEL_DIR / "roc_curve_genes.png"))
    plot_confusion(y_test, best_pred, save_path=str(MODEL_DIR / "confusion_matrix_genes.png"))

    # ── 5. Feature importance — which genes matter most? ─────────────────────
    best_model = results[best_name][0]
    clf = best_model.named_steps["clf"]
    sel = best_model.named_steps["select"]
    selected_genes = X_train.columns[sel.get_support()]

    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=selected_genes)
        print(f"\nTop 15 most predictive resistance genes [{best_name.upper()}]:")
        print(imp.sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
