"""
Build a gene presence/absence feature matrix using BV-BRC specialty gene annotations.

For each genome we query which antibiotic resistance genes (from CARD, ARDB, PATRIC)
are annotated, then build a binary matrix:

    rows    = genomes
    columns = resistance genes (e.g. "bla_SHV", "aac_6_", "qnrB")
    values  = 1 if the gene is present, 0 if absent

This is far more informative than raw k-mers because every feature is a
biologically known resistance determinant.
"""

import time
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
from tqdm import tqdm

BV_BRC_API = "https://www.bv-brc.org/api"
AMR_SOURCES = {"CARD", "ARDB", "PATRIC"}   # resistance gene databases to include


def fetch_amr_genes(genome_id: str, retries: int = 3) -> list[str]:
    """
    Return a list of resistance gene/product names annotated for this genome.
    Uses CARD, ARDB and PATRIC as sources.

    Note: we fetch all specialty genes without a property filter (spaces in
    query values are tricky with this API) and filter by property + source
    in Python instead.
    """
    url = (
        f"{BV_BRC_API}/sp_gene/"
        f"?eq(genome_id,{genome_id})"
        f"&select(property,gene,product,source)"
        f"&limit(500)"
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
            resp.raise_for_status()
            records = resp.json()
            genes = []
            for r in records:
                # keep only antibiotic resistance entries from our target sources
                if r.get("property") != "Antibiotic Resistance":
                    continue
                if r.get("source") not in AMR_SOURCES:
                    continue
                # prefer gene name; fall back to product description
                name = r.get("gene") or r.get("product") or ""
                name = name.strip()
                if name:
                    genes.append(name)
            return genes
        except Exception:
            time.sleep(2 ** attempt)
    return []


def build_gene_matrix(metadata_path: str | Path,
                      cache_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch AMR genes for every genome in metadata_path and return:
        X  – binary presence/absence DataFrame  (genomes × genes)
        y  – label Series (0 = susceptible, 1 = resistant)
    """
    meta = pd.read_csv(metadata_path)
    meta["genome_id"] = meta["genome_id"].astype(str)

    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        print(f"Loading cached gene matrix from {cache_path}")
        combined = pd.read_csv(cache_path, index_col=0)
        y = combined["__label__"]
        X = combined.drop(columns=["__label__"])
        return X, y

    print(f"Fetching AMR gene annotations for {len(meta)} genomes from BV-BRC ...")
    rows: Dict[str, dict] = {}
    label_map: Dict[str, int] = {}

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Fetching genes"):
        gid = row["genome_id"]
        label_map[gid] = 1 if row["label"] == "R" else 0
        genes = fetch_amr_genes(gid)
        rows[gid] = {g: 1 for g in genes}
        time.sleep(0.05)   # polite rate limiting

    # Build presence/absence matrix
    X = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    X.index.name = "genome_id"

    y = pd.Series(label_map, name="label")
    y.index.name = "genome_id"
    y = y.reindex(X.index)

    print(f"\nGene matrix: {X.shape[0]} genomes × {X.shape[1]} unique genes")
    print(f"Genomes with ≥1 gene: {(X.sum(axis=1) > 0).sum()}")
    print(f"Top 20 most common genes:")
    print(X.sum().sort_values(ascending=False).head(20).to_string())

    if cache_path:
        combined = X.copy()
        combined["__label__"] = y
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_path)
        print(f"\nCached to {cache_path}")

    return X, y


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    X, y = build_gene_matrix(
        metadata_path=ROOT / "data" / "processed" / "metadata.csv",
        cache_path=ROOT / "data" / "processed" / "gene_matrix.csv",
    )
    print(f"\nClass balance: {y.value_counts().to_dict()}")
