"""
Predict antibiotic resistance for a new bacterial genome.
Runs all available trained models and prints a full resistance profile.

Usage:
    python src/predict.py --fasta path/to/genome.fasta \
                          --genome_id 573.99999

    # single antibiotic
    python src/predict.py --fasta genome.fasta --genome_id 573.x --antibiotic meropenem
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import kmer_counts
from gene_features import fetch_amr_genes

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"

K = 6   # k-mer length used during training


# ── Feature helpers ───────────────────────────────────────────────────────────

def load_kmer_template() -> list[str]:
    X = pd.read_csv(PROC_DIR / "X.csv", index_col=0, nrows=1)
    return list(X.columns)


def load_gene_template() -> list[str]:
    df = pd.read_csv(PROC_DIR / "gene_matrix.csv", index_col=0, nrows=1)
    df = df.drop(columns=["__label__"], errors="ignore")
    return list(df.columns)


def extract_kmer_vector(fasta_path: Path) -> pd.Series:
    template = load_kmer_template()
    seq = ""
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip().upper()
    counts = kmer_counts(seq, K)
    return pd.Series(counts, dtype=float).reindex(template, fill_value=0)


def extract_gene_vector(genome_id: str) -> pd.Series:
    template = load_gene_template()
    genes = fetch_amr_genes(genome_id)
    vec = pd.Series(0, index=template, dtype=int)
    for g in genes:
        if g in vec.index:
            vec[g] = 1
    return vec


def build_input_vector(fasta_path: Path, genome_id: str | None,
                       feature_names: list[str],
                       kmer_vec: pd.Series,
                       gene_vec: pd.Series) -> pd.DataFrame:
    """Assemble the feature vector for one model using its saved feature list."""
    kmer_cols = [c for c in feature_names if not c.startswith("gene__")]
    gene_cols  = [c for c in feature_names if c.startswith("gene__")]

    gene_vec_prefixed = gene_vec.rename(lambda x: "gene__" + x)

    parts = []
    if kmer_cols:
        parts.append(kmer_vec.reindex(kmer_cols, fill_value=0))
    if gene_cols:
        parts.append(gene_vec_prefixed.reindex(gene_cols, fill_value=0))

    row = pd.concat(parts)
    return row.to_frame().T


# ── Core prediction ───────────────────────────────────────────────────────────

def predict_one(model_bundle: dict, X: pd.DataFrame) -> dict:
    model = model_bundle["model"]
    prob  = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    return {
        "pred":  pred,
        "prob_r": prob[1],
        "prob_s": prob[0],
    }


def run_profile(fasta_path: Path, genome_id: str | None,
                antibiotic_filter: str | None = None,
                threshold: float = 70.0) -> None:
    """Run all available models and print a resistance profile."""

    # ── Extract features once, reuse for all models ───────────────────────────
    print("Extracting k-mer features ...")
    kmer_vec = extract_kmer_vector(fasta_path)

    gene_vec = pd.Series(dtype=int)
    if genome_id:
        print(f"Fetching resistance genes for {genome_id} from BV-BRC ...")
        gene_vec = extract_gene_vector(genome_id)
    else:
        print("No --genome_id provided; gene features will be zero.")
        gene_template = load_gene_template()
        gene_vec = pd.Series(0, index=gene_template, dtype=int)

    # ── Find model files ──────────────────────────────────────────────────────
    model_files = sorted(MODEL_DIR.glob("*.pkl"))
    # exclude old-style models that aren't antibiotic bundles
    model_files = [m for m in model_files
                   if not any(m.stem.startswith(p)
                              for p in ["rf_", "xgb_", "rf", "xgb"])]

    if not model_files:
        # fallback: look for any .pkl with antibiotic metadata inside
        model_files = sorted(MODEL_DIR.glob("*.pkl"))

    if antibiotic_filter:
        safe = antibiotic_filter.replace("/", "_").replace(" ", "_")
        model_files = [m for m in model_files if safe in m.stem]

    if not model_files:
        print("No trained models found. Run train_multi.py first.")
        sys.exit(1)

    # ── Run predictions ───────────────────────────────────────────────────────
    rows = []
    for mf in model_files:
        with open(mf, "rb") as f:
            bundle = pickle.load(f)
        if not isinstance(bundle, dict) or "antibiotic" not in bundle:
            continue

        ab       = bundle["antibiotic"]
        features = bundle["features"]

        X = build_input_vector(fasta_path, genome_id, features, kmer_vec, gene_vec)
        res = predict_one(bundle, X)
        rows.append({
            "antibiotic": ab,
            "prediction": "Resistant" if res["pred"] == 1 else "Susceptible",
            "confidence": max(res["prob_r"], res["prob_s"]) * 100,
            "P(Resistant)": res["prob_r"] * 100,
            "P(Susceptible)": res["prob_s"] * 100,
        })

    if not rows:
        print("No valid model bundles found.")
        sys.exit(1)

    # ── Print report ──────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD = threshold   # below this → "Uncertain"

    df = pd.DataFrame(rows).sort_values("antibiotic")

    # Apply threshold: low-confidence predictions become "Uncertain"
    def classify(row):
        if row["confidence"] < CONFIDENCE_THRESHOLD:
            return "Uncertain", "?"
        icon = "✗" if row["prediction"] == "Resistant" else "✓"
        return row["prediction"], icon

    df[["display", "icon"]] = df.apply(
        lambda r: pd.Series(classify(r)), axis=1
    )

    print(f"\n{'─'*68}")
    print(f"  RESISTANCE PROFILE  (confidence threshold: {CONFIDENCE_THRESHOLD:.0f}%)")
    print(f"  Genome : {fasta_path.name}"
          + (f"  (ID: {genome_id})" if genome_id else ""))
    print(f"{'─'*68}")
    print(f"  {'Antibiotic':<35} {'Verdict':<16} {'Confidence':>10}")
    print(f"  {'─'*63}")

    for _, r in df.iterrows():
        if r["display"] == "Uncertain":
            verdict_str = f"~ Uncertain"
        else:
            verdict_str = f"{r['icon']} {r['display']}"
        print(f"  {r['antibiotic']:<35} {verdict_str:<16} {r['confidence']:>8.1f}%")

    n_resistant   = (df["display"] == "Resistant").sum()
    n_susceptible = (df["display"] == "Susceptible").sum()
    n_uncertain   = (df["display"] == "Uncertain").sum()

    print(f"\n  {'─'*63}")
    print(f"  ✗ Resistant   : {n_resistant}/{len(df)} antibiotics")
    print(f"  ✓ Susceptible : {n_susceptible}/{len(df)} antibiotics")
    if n_uncertain:
        print(f"  ~ Uncertain   : {n_uncertain}/{len(df)} antibiotics  "
              f"(confidence < {CONFIDENCE_THRESHOLD:.0f}% — lab confirmation advised)")
    print(f"{'─'*68}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict antibiotic resistance from a bacterial genome FASTA."
    )
    parser.add_argument("--fasta",      required=True, help="Path to genome FASTA")
    parser.add_argument("--genome_id",  default=None,  help="BV-BRC genome ID (for gene lookup)")
    parser.add_argument("--antibiotic", default=None,  help="Single antibiotic (default: all)")
    parser.add_argument("--threshold",  type=float, default=70.0,
                        help="Confidence %% below which prediction is marked Uncertain (default: 70)")
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"ERROR: FASTA file not found: {fasta_path}")
        sys.exit(1)

    run_profile(fasta_path, args.genome_id, args.antibiotic, args.threshold)


if __name__ == "__main__":
    main()
