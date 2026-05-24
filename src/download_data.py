"""
Download labeled K. pneumoniae genomes from BV-BRC (formerly PATRIC).

Steps:
  1. Fetch AMR phenotype records (genome_id + resistant_phenotype)
  2. Balance classes (equal R and S)
  3. Download FASTA files for each genome

Usage:
    python src/download_data.py --antibiotic ciprofloxacin \
                                --taxon_id 573 \
                                --n 200 \
                                --out_dir data/raw
"""

import argparse
import time
import sys
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

BV_BRC_API = "https://www.bv-brc.org/api"
FASTA_URL = "https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{gid})&http_accept=application/dna+fasta"


def fetch_amr_labels(taxon_id: int, antibiotic: str, limit: int = 5000) -> pd.DataFrame:
    """Fetch genome IDs and their R/S labels from BV-BRC."""
    url = (
        f"{BV_BRC_API}/genome_amr/"
        f"?eq(antibiotic,{antibiotic})"
        f"&eq(taxon_id,{taxon_id})"
        f"&select(genome_id,resistant_phenotype,genome_name)"
        f"&limit({limit})"
    )
    print(f"Fetching AMR labels from BV-BRC...")
    resp = requests.get(url, headers={"Accept": "application/json"}, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())

    # Keep only clean R/S labels
    df = df[df["resistant_phenotype"].isin(["Resistant", "Susceptible"])].copy()
    df["label"] = df["resistant_phenotype"].map({"Resistant": "R", "Susceptible": "S"})
    df = df.drop_duplicates("genome_id")
    return df


def balance_dataset(df: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    """Sample equal numbers of resistant and susceptible genomes."""
    groups = []
    for label in ["R", "S"]:
        subset = df[df["label"] == label]
        available = len(subset)
        take = min(n_per_class, available)
        print(f"  {label}: {available} available → taking {take}")
        groups.append(subset.sample(take, random_state=42))
    return pd.concat(groups).reset_index(drop=True)


def download_fasta(genome_id: str, out_dir: Path, retries: int = 3) -> bool:
    """Download the FASTA sequence for one genome. Returns True on success."""
    out_path = out_dir / f"{genome_id}.fasta"
    if out_path.exists() and out_path.stat().st_size > 0:
        return True  # already downloaded

    url = FASTA_URL.format(gid=genome_id)
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and resp.text.strip().startswith(">"):
                out_path.write_text(resp.text)
                return True
            time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--antibiotic", default="ciprofloxacin")
    parser.add_argument("--taxon_id", type=int, default=573,
                        help="573 = Klebsiella pneumoniae")
    parser.add_argument("--n", type=int, default=200,
                        help="Total genomes to download (split equally R/S)")
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch labels
    df = fetch_amr_labels(args.taxon_id, args.antibiotic, limit=5000)
    print(f"\nTotal labeled genomes fetched: {len(df)}")
    print(df["label"].value_counts().to_string())

    # 2. Balance
    n_per_class = args.n // 2
    df_balanced = balance_dataset(df, n_per_class)
    print(f"\nBalanced dataset: {len(df_balanced)} genomes")

    # 3. Save metadata
    meta_path = out_dir.parent / "processed" / "metadata.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced[["genome_id", "genome_name", "label"]].to_csv(meta_path, index=False)
    print(f"Metadata saved to {meta_path}")

    # 4. Download FASTAs
    print(f"\nDownloading FASTA files to {out_dir}/ ...")
    failed = []
    for _, row in tqdm(df_balanced.iterrows(), total=len(df_balanced)):
        ok = download_fasta(row["genome_id"], out_dir)
        if not ok:
            failed.append(row["genome_id"])
        time.sleep(0.1)  # polite rate limiting

    print(f"\nDone. {len(df_balanced) - len(failed)}/{len(df_balanced)} genomes downloaded.")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
