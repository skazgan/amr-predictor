"""
Train one combined (k-mer + gene) model per antibiotic.

Reuses the shared gene_matrix.csv and X.csv (k-mer features) already built.
Saves one model file per antibiotic under models/.

Usage:
    python src/train_multi.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import build_feature_matrix
from gene_features import build_gene_matrix
from train import train

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, roc_auc_score

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

ANTIBIOTICS = [
    "ciprofloxacin",
    "meropenem",
    "gentamicin",
    "tetracycline",
    "trimethoprim/sulfamethoxazole",
    "cefepime",
]


def load_shared_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared k-mer and gene matrices (all genomes)."""

    # k-mer matrix
    kmer_path = PROC_DIR / "X.csv"
    print("Loading shared k-mer matrix ...")
    X_kmer = pd.read_csv(kmer_path, index_col=0)
    X_kmer.index = X_kmer.index.astype(str)

    # gene matrix
    gene_path = PROC_DIR / "gene_matrix.csv"
    print("Loading shared gene matrix ...")
    gene_raw = pd.read_csv(gene_path, index_col=0)
    gene_raw.index = gene_raw.index.astype(str)
    X_gene = gene_raw.drop(columns=["__label__"], errors="ignore")
    X_gene.columns = ["gene__" + c for c in X_gene.columns]

    print(f"  k-mer: {X_kmer.shape}  |  gene: {X_gene.shape}")
    return X_kmer, X_gene


def build_combined(X_kmer: pd.DataFrame, X_gene: pd.DataFrame,
                   genome_ids: list[str]) -> pd.DataFrame:
    """Intersect genome IDs and concatenate k-mer + gene features."""
    idx = (pd.Index(genome_ids)
             .intersection(X_kmer.index)
             .intersection(X_gene.index))
    return pd.concat([X_kmer.loc[idx], X_gene.loc[idx]], axis=1)


def train_antibiotic(antibiotic: str,
                     X_kmer: pd.DataFrame,
                     X_gene: pd.DataFrame) -> dict:
    """Train + evaluate model for one antibiotic. Returns result dict."""

    print(f"\n{'='*60}")
    print(f"  ANTIBIOTIC: {antibiotic.upper()}")
    print(f"{'='*60}")

    # ── Labels ────────────────────────────────────────────────────────────────
    safe_dir = antibiotic.replace("/", "_").replace(" ", "_")
    meta_path = PROC_DIR / safe_dir / "metadata.csv"
    if not meta_path.exists():
        print(f"  No metadata found for {antibiotic} — skipping.")
        return {}

    meta = pd.read_csv(meta_path)
    meta["genome_id"] = meta["genome_id"].astype(str)
    label_map = dict(zip(meta["genome_id"], meta["label"].map({"R": 1, "S": 0})))

    # ── Combined feature matrix ────────────────────────────────────────────────
    genome_ids = list(label_map.keys())
    X = build_combined(X_kmer, X_gene, genome_ids)
    y = pd.Series({gid: label_map[gid] for gid in X.index}, name="label")

    n_missing = len(genome_ids) - len(X)
    print(f"  Genomes: {len(X)}  (dropped {n_missing} with no features)")
    print(f"  Class balance: R={y.sum()}  S={(y==0).sum()}")

    if len(X) < 50:
        print("  Not enough genomes — skipping.")
        return {}

    # ── Per-group feature selection ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    kmer_cols = [c for c in X_train.columns if not c.startswith("gene__")]
    gene_cols  = [c for c in X_train.columns if c.startswith("gene__")]

    sel = SelectKBest(mutual_info_classif, k=min(256, len(kmer_cols)))
    sel.fit(X_train[kmer_cols], y_train)
    top_kmers = [kmer_cols[i] for i in sel.get_support(indices=True)]
    keep = top_kmers + gene_cols

    X_train, X_test = X_train[keep], X_test[keep]
    print(f"  Features: {len(keep)} ({len(top_kmers)} k-mer + {len(gene_cols)} gene)")

    # ── Train XGBoost (best model from earlier) ───────────────────────────────
    n_feat = min(512, len(keep))
    model, scores = train(X_train, y_train, model="xgb", n_features=n_feat)

    cv_auc = scores["test_roc_auc"].mean()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"\n  --- Report [{antibiotic}] ---")
    print(classification_report(y_test, y_pred,
                                target_names=["Susceptible", "Resistant"],
                                digits=3))
    print(f"  CV AUC   : {cv_auc:.3f}")
    print(f"  Test AUC : {test_auc:.3f}")

    # ── Save model ────────────────────────────────────────────────────────────
    # sanitise antibiotic name for filename
    safe_name = antibiotic.replace("/", "_").replace(" ", "_")
    model_path = MODEL_DIR / f"{safe_name}.pkl"
    meta_model = {
        "model":      model,
        "antibiotic": antibiotic,
        "features":   keep,
        "cv_auc":     cv_auc,
        "test_auc":   test_auc,
    }
    with open(model_path, "wb") as f:
        pickle.dump(meta_model, f)
    print(f"  Model saved → {model_path}")

    return {"antibiotic": antibiotic, "cv_auc": cv_auc, "test_auc": test_auc,
            "n_genomes": len(X)}


def main():
    X_kmer, X_gene = load_shared_features()

    results = []
    for ab in ANTIBIOTICS:
        res = train_antibiotic(ab, X_kmer, X_gene)
        if res:
            results.append(res)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — all antibiotics")
    print(f"{'='*60}")
    print(f"{'Antibiotic':<35} {'Genomes':>8} {'CV AUC':>8} {'Test AUC':>10}")
    print("-" * 63)
    for r in sorted(results, key=lambda x: -x["test_auc"]):
        print(f"{r['antibiotic']:<35} {r['n_genomes']:>8} "
              f"{r['cv_auc']:>8.3f} {r['test_auc']:>10.3f}")


if __name__ == "__main__":
    main()
