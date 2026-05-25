"""
Fetch AMR data for multiple organisms from BV-BRC.

Organisms:
  - Escherichia coli         (taxon_id 562)
  - Staphylococcus aureus    (taxon_id 1280)
  - Acinetobacter baumannii  (taxon_id 470)

For each organism:
  1. Fetch genome IDs with AMR phenotype data
  2. Fetch AMR labels (antibiotic, resistant_phenotype)
  3. Fetch specialty genes (resistance gene presence/absence)
  4. Save to data/processed/{safe_name}/

Output files per organism:
  data/processed/{org}/labels.csv   — genome_id, antibiotic, phenotype
  data/processed/{org}/genes.csv    — genome_id × gene binary matrix
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Organism registry ──────────────────────────────────────────────────────────
ORGANISMS = {
    "escherichia_coli": {
        "display":   "Escherichia coli",
        "taxon_id":  562,
        "antibiotics": [
            "ciprofloxacin", "meropenem", "gentamicin", "tetracycline",
            "trimethoprim/sulfamethoxazole", "cefepime", "amikacin",
            "ampicillin", "ceftriaxone", "piperacillin/tazobactam",
        ],
    },
    "staphylococcus_aureus": {
        "display":   "Staphylococcus aureus",
        "taxon_id":  1280,
        "antibiotics": [
            "oxacillin", "vancomycin", "tetracycline",
            "trimethoprim/sulfamethoxazole", "clindamycin",
            "erythromycin", "ciprofloxacin", "gentamicin",
        ],
    },
    "acinetobacter_baumannii": {
        "display":   "Acinetobacter baumannii",
        "taxon_id":  470,
        "antibiotics": [
            "meropenem", "imipenem", "ciprofloxacin", "gentamicin",
            "amikacin", "colistin", "tetracycline",
            "trimethoprim/sulfamethoxazole",
        ],
    },
}

BASE_URL = "https://www.bv-brc.org/api"
HEADERS  = {"Accept": "application/json"}
BATCH    = 5000


def bvbrc_get(endpoint: str, params: str, limit: int = BATCH, offset: int = 0) -> list:
    url = f"{BASE_URL}/{endpoint}/?{params}&limit({limit},{offset})"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    Retry {attempt+1}/3: {e}")
            time.sleep(2 ** attempt)
    return []


def fetch_all(endpoint: str, params: str) -> list:
    """Paginate through all results."""
    results, offset = [], 0
    while True:
        batch = bvbrc_get(endpoint, params, limit=BATCH, offset=offset)
        if not batch:
            break
        results.extend(batch)
        print(f"    fetched {len(results):,} records...", end="\r")
        if len(batch) < BATCH:
            break
        offset += BATCH
        time.sleep(0.3)
    print()
    return results


def fetch_amr_labels(taxon_id: int) -> pd.DataFrame:
    """Fetch all AMR phenotype records for a taxon."""
    print(f"  Fetching AMR labels for taxon {taxon_id}...")
    params = (
        f"eq(taxon_id,{taxon_id})"
        f"&select(genome_id,antibiotic,resistant_phenotype)"
    )
    rows = fetch_all("genome_amr", params)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.columns = df.columns.str.lower()
    # Keep only R/S
    df = df[df["resistant_phenotype"].isin(["Resistant", "Susceptible"])].copy()
    df["antibiotic"] = df["antibiotic"].str.lower().str.strip()
    return df[["genome_id", "antibiotic", "resistant_phenotype"]]


def fetch_genes(genome_ids: list, taxon_id: int) -> pd.DataFrame:
    """Fetch specialty gene presence for a list of genome IDs."""
    print(f"  Fetching genes for {len(genome_ids):,} genomes...")
    all_rows = []
    batch_size = 200
    for i in range(0, len(genome_ids), batch_size):
        batch = genome_ids[i:i + batch_size]
        id_list = ",".join(batch)
        params = (
            f"in(genome_id,({id_list}))"
            f"&select(genome_id,gene,product)"
            f"&eq(property,Antibiotic Resistance)"
        )
        rows = bvbrc_get("sp_gene", params, limit=10000)
        all_rows.extend(rows)
        if i % 2000 == 0:
            print(f"    genes: {i}/{len(genome_ids)} genomes processed...", end="\r")
    print()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.columns = df.columns.str.lower()

    # Build binary presence/absence matrix
    df["gene"] = df["gene"].fillna(df.get("product", "unknown")).fillna("unknown")
    df["gene"] = df["gene"].str.strip().str.replace(" ", "_")
    df["present"] = 1

    matrix = (df.groupby(["genome_id", "gene"])["present"]
                .max()
                .unstack(fill_value=0))
    return matrix


def process_organism(name: str, info: dict):
    print(f"\n{'='*60}")
    print(f"ORGANISM: {info['display']} (taxon {info['taxon_id']})")
    print(f"{'='*60}")

    out_dir = DATA_DIR / name
    out_dir.mkdir(exist_ok=True)

    # ── 1. AMR labels ─────────────────────────────────────────────────────────
    labels_path = out_dir / "labels.csv"
    if labels_path.exists():
        print(f"  Labels already cached: {labels_path}")
        df_labels = pd.read_csv(labels_path, dtype={"genome_id": str})
    else:
        df_labels = fetch_amr_labels(info["taxon_id"])
        if df_labels.empty:
            print("  ERROR: No AMR data found.")
            return
        df_labels.to_csv(labels_path, index=False)
        print(f"  Saved {len(df_labels):,} label rows → {labels_path}")

    # Filter to target antibiotics
    df_labels = df_labels[df_labels["antibiotic"].isin(info["antibiotics"])]
    genome_ids = df_labels["genome_id"].unique().tolist()
    print(f"  Genomes with target antibiotic data: {len(genome_ids):,}")

    # Print per-antibiotic counts
    for ab in sorted(info["antibiotics"]):
        sub = df_labels[df_labels["antibiotic"] == ab]
        n_r = (sub["resistant_phenotype"] == "Resistant").sum()
        n_s = (sub["resistant_phenotype"] == "Susceptible").sum()
        if n_r + n_s > 0:
            print(f"    {ab:<40} R={n_r:>5,}  S={n_s:>5,}  total={n_r+n_s:>5,}")
        else:
            print(f"    {ab:<40} (no data)")

    # ── 2. Gene matrix ────────────────────────────────────────────────────────
    genes_path = out_dir / "genes.csv"
    if genes_path.exists():
        print(f"  Genes already cached: {genes_path}")
    else:
        gene_matrix = fetch_genes(genome_ids, info["taxon_id"])
        if not gene_matrix.empty:
            gene_matrix.to_csv(genes_path)
            print(f"  Saved gene matrix {gene_matrix.shape} → {genes_path}")
        else:
            print("  WARNING: No gene data retrieved.")

    print(f"  Done: {info['display']}")


def main():
    print("BV-BRC MULTI-ORGANISM DATA FETCH")
    print("Organisms: E. coli, S. aureus, A. baumannii\n")

    for name, info in ORGANISMS.items():
        process_organism(name, info)

    # Save organism registry for the website
    registry = {
        name: {
            "display":     info["display"],
            "taxon_id":    info["taxon_id"],
            "antibiotics": info["antibiotics"],
        }
        for name, info in ORGANISMS.items()
    }
    # Add existing K. pneumoniae entry
    registry["klebsiella_pneumoniae"] = {
        "display":     "Klebsiella pneumoniae",
        "taxon_id":    573,
        "antibiotics": [
            "ciprofloxacin", "meropenem", "gentamicin", "tetracycline",
            "trimethoprim/sulfamethoxazole", "cefepime", "amikacin",
            "imipenem", "piperacillin/tazobactam", "levofloxacin",
        ],
    }

    out = ROOT / "artifacts" / "organisms.json"
    out.write_text(json.dumps(registry, indent=2))
    print(f"\nSaved organism registry → {out}")
    print("\nNext step: run  python src/train_organisms.py")


if __name__ == "__main__":
    main()
