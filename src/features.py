"""
Feature extraction from bacterial genome FASTA files.
Converts raw DNA sequences into k-mer frequency vectors.
"""

from itertools import product
from collections import Counter
import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm


def all_kmers(k: int) -> list[str]:
    """Return all possible k-mers over the DNA alphabet."""
    return ["".join(p) for p in product("ACGT", repeat=k)]


def kmer_counts(sequence: str, k: int) -> Counter:
    sequence = sequence.upper().replace("N", "")
    return Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def genome_to_kmer_vector(fasta_path: str | Path, k: int = 6) -> np.ndarray:
    """
    Read a FASTA file (one or more contigs) and return a normalized
    k-mer frequency vector.
    """
    total = Counter()
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        total += kmer_counts(str(record.seq), k)

    vocab = all_kmers(k)
    vec = np.array([total.get(km, 0) for km in vocab], dtype=np.float32)

    total_count = vec.sum()
    if total_count > 0:
        vec /= total_count
    return vec


def build_feature_matrix(
    fasta_dir: str | Path,
    labels: dict[str, str],
    k: int = 6,
    label_map: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a feature matrix from all FASTA files in fasta_dir.

    Args:
        fasta_dir: Directory containing one .fasta / .fa file per genome.
        labels:    Dict mapping genome filename stem -> label string (e.g. "R" or "S").
        k:         k-mer length.
        label_map: How to encode label strings as integers. Defaults to {"S": 0, "R": 1}.

    Returns:
        X: DataFrame of shape (n_genomes, 4^k)
        y: Series of integer labels
    """
    if label_map is None:
        label_map = {"S": 0, "R": 1}

    fasta_dir = Path(fasta_dir)
    vocab = all_kmers(k)
    rows, y_vals, index = [], [], []

    fasta_files = sorted(
        f for f in fasta_dir.iterdir() if f.suffix in {".fasta", ".fa", ".fna"}
    )

    for fpath in tqdm(fasta_files, desc="Extracting k-mers"):
        stem = fpath.stem
        if stem not in labels:
            continue
        vec = genome_to_kmer_vector(fpath, k)
        rows.append(vec)
        y_vals.append(label_map[labels[stem]])
        index.append(stem)

    X = pd.DataFrame(rows, columns=vocab, index=index)
    y = pd.Series(y_vals, index=index, name="label")
    return X, y
