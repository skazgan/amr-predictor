"""
Pipeline combining k-mer counts + resistance gene presence/absence features.

The idea: stack the two feature matrices side-by-side.
Each genome gets a vector of:
  - 4096 k-mer counts  (sequence-level signal)
  - 796  gene flags    (known resistance gene signal)

Usage:
    python src/pipeline_combined.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import train
from evaluate import plot_roc, plot_confusion

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_and_merge() -> tuple[pd.DataFrame, pd.Series]:
    """Load k-mer and gene matrices, intersect on genome_id, concatenate features."""

    # ── k-mer matrix ──────────────────────────────────────────────────────────
    print("Loading k-mer matrix ...")
    X_kmer = pd.read_csv(PROC_DIR / "X.csv", index_col=0)
    y_kmer = pd.read_csv(PROC_DIR / "y.csv", index_col=0).squeeze()
    X_kmer.index = X_kmer.index.astype(str)
    y_kmer.index = y_kmer.index.astype(str)
    print(f"  k-mer : {X_kmer.shape}")

    # ── gene matrix ───────────────────────────────────────────────────────────
    print("Loading gene matrix ...")
    gene_combined = pd.read_csv(PROC_DIR / "gene_matrix.csv", index_col=0)
    gene_combined.index = gene_combined.index.astype(str)
    y_gene = gene_combined["__label__"]
    X_gene = gene_combined.drop(columns=["__label__"])
    # prefix gene columns so names don't clash
    X_gene.columns = ["gene__" + c for c in X_gene.columns]
    print(f"  genes : {X_gene.shape}")

    # ── intersect ─────────────────────────────────────────────────────────────
    common = X_kmer.index.intersection(X_gene.index)
    print(f"\nGenomes in both matrices: {len(common)}")

    X_kmer  = X_kmer.loc[common]
    X_gene  = X_gene.loc[common]
    y       = y_kmer.loc[common]

    # ── concatenate ───────────────────────────────────────────────────────────
    X = pd.concat([X_kmer, X_gene], axis=1)
    print(f"Combined matrix: {X.shape[0]} genomes × {X.shape[1]} features "
          f"({X_kmer.shape[1]} k-mer + {X_gene.shape[1]} gene)")
    return X, y


def main():
    X, y = load_and_merge()
    print(f"Class balance: {y.value_counts().to_dict()}")

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain / test split: {len(X_train)} train, {len(X_test)} test")

    # ── Pre-select within each feature group before training ──────────────────
    # Selecting across the whole 4892-column pool lets k-mers crowd out genes.
    # Instead: keep top 256 k-mers + ALL gene features, then let the model train.
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    kmer_cols = [c for c in X_train.columns if not c.startswith("gene__")]
    gene_cols  = [c for c in X_train.columns if c.startswith("gene__")]

    print(f"\nPer-group feature selection: top 256 k-mers + all {len(gene_cols)} gene features ...")
    selector = SelectKBest(mutual_info_classif, k=256)
    selector.fit(X_train[kmer_cols], y_train)
    top_kmers = [kmer_cols[i] for i in selector.get_support(indices=True)]

    keep = top_kmers + gene_cols
    X_train = X_train[keep]
    X_test  = X_test[keep]
    print(f"Final feature count: {len(keep)} ({len(top_kmers)} k-mer + {len(gene_cols)} gene)")

    # ── Train both models ─────────────────────────────────────────────────────
    results = {}
    for model_name in ["rf", "xgb"]:
        print(f"\n{'='*55}")
        print(f"[{model_name.upper()}] Training on combined features ...")
        model, _ = train(X_train, y_train, model=model_name, n_features=min(len(keep), 512))

        model_path = MODEL_DIR / f"{model_name}_combined.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[model_name] = (model, y_pred, y_prob, auc)

        print(f"\n--- Hold-out Test Report [{model_name.upper()} / combined] ---")
        print(classification_report(y_test, y_pred, target_names=["Susceptible", "Resistant"]))
        print(f"ROC-AUC : {auc:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("SUMMARY — combined features vs individual")
    print(f"{'Feature set':<22} {'Model':<8} {'Test AUC':>10}")
    print(f"{'k-mers only':<22} {'XGB':<8} {'0.661':>10}")
    print(f"{'genes only':<22} {'RF':<8}  {'0.793':>10}")
    for name, (_, _, _, auc) in results.items():
        print(f"{'k-mers + genes':<22} {name.upper():<8} {auc:>10.3f}")

    best_name = max(results, key=lambda k: results[k][3])
    _, best_pred, best_prob, best_auc = results[best_name]
    print(f"\nBest combined model: {best_name.upper()} (AUC {best_auc:.3f})")

    plot_roc(y_test, best_prob,
             save_path=str(MODEL_DIR / "roc_curve_combined.png"))
    plot_confusion(y_test, best_pred,
                   save_path=str(MODEL_DIR / "confusion_matrix_combined.png"))

    # ── Top features ──────────────────────────────────────────────────────────
    best_model = results[best_name][0]
    clf  = best_model.named_steps["clf"]
    sel  = best_model.named_steps["select"]
    cols = X_train.columns[sel.get_support()]

    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=cols)
        gene_imp  = imp[imp.index.str.startswith("gene__")].sort_values(ascending=False)
        kmer_imp  = imp[~imp.index.str.startswith("gene__")].sort_values(ascending=False)
        print(f"\nTop 10 gene features [{best_name.upper()}]:")
        print(gene_imp.head(10).rename(lambda x: x.replace("gene__", "")).to_string())
        print(f"\nTop 5 k-mer features [{best_name.upper()}]:")
        print(kmer_imp.head(5).to_string())


if __name__ == "__main__":
    main()
