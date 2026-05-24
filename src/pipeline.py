"""
End-to-end pipeline: features → train → evaluate.

Usage:
    python src/pipeline.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_feature_matrix
from train import train
from evaluate import plot_roc, plot_confusion

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

K = 6


def main():
    # ── 1. Load labels ────────────────────────────────────────────────────────
    meta = pd.read_csv(PROC_DIR / "metadata.csv")
    labels = dict(zip(meta["genome_id"].astype(str), meta["label"]))
    print(f"Loaded metadata: {len(labels)} genomes")

    # ── 2. Extract k-mer features ─────────────────────────────────────────────
    x_path = PROC_DIR / "X.csv"
    y_path = PROC_DIR / "y.csv"

    if x_path.exists() and y_path.exists():
        print("Found cached features — loading from disk.")
        X = pd.read_csv(x_path, index_col=0)
        y = pd.read_csv(y_path, index_col=0).squeeze()
    else:
        print(f"\nExtracting {K}-mer features from {RAW_DIR} ...")
        X, y = build_feature_matrix(RAW_DIR, labels, k=K)
        X.to_csv(x_path)
        y.to_csv(y_path)
        print(f"Features saved to {PROC_DIR}")

    print(f"\nFeature matrix : {X.shape[0]} genomes × {X.shape[1]} features")
    print(f"Class balance  : {y.value_counts().to_dict()}")

    # ── 3. Split first, then train ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain / test split: {len(X_train)} train, {len(X_test)} test")

    results = {}
    for model_name in ["rf", "xgb"]:
        print(f"\n{'='*50}")
        print(f"[{model_name.upper()}] Tuning feature count + training ...")
        model, _ = train(X_train, y_train, model=model_name)  # auto-tunes n_features

        model_path = MODEL_DIR / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[model_name] = (model, y_pred, y_prob, auc)

        print(f"\n--- Hold-out Test Report [{model_name.upper()}] ---")
        print(classification_report(y_test, y_pred, target_names=["Susceptible", "Resistant"]))
        print(f"ROC-AUC : {auc:.3f}")

    # ── 4. Summary + plots ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'Model':<8} {'Test AUC':>10}")
    for name, (_, _, _, auc) in results.items():
        print(f"{name.upper():<8} {auc:>10.3f}")

    best_name = max(results, key=lambda k: results[k][3])
    _, best_pred, best_prob, best_auc = results[best_name]
    print(f"\nBest model: {best_name.upper()} (AUC {best_auc:.3f})")
    plot_roc(y_test, best_prob, save_path=str(MODEL_DIR / "roc_curve.png"))
    plot_confusion(y_test, best_pred, save_path=str(MODEL_DIR / "confusion_matrix.png"))


if __name__ == "__main__":
    main()
