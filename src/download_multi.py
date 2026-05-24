"""
Download labels for multiple antibiotics from BV-BRC, then fetch any
FASTA files we don't already have.

For each antibiotic we save:
    data/processed/<antibiotic>/metadata.csv  (genome_id, label)

FASTA files are shared across antibiotics in data/raw/.

Usage:
    python src/download_multi.py
"""

import time
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

BV_BRC_API   = "https://www.bv-brc.org/api"
FASTA_URL    = "https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{gid})&http_accept=application/dna+fasta"
TAXON_ID     = 573        # Klebsiella pneumoniae
N_PER_CLASS  = 2000       # max genomes per R/S class per antibiotic

ANTIBIOTICS = [
    "ciprofloxacin",
    "meropenem",
    "gentamicin",
    "tetracycline",
    "trimethoprim/sulfamethoxazole",
    "cefepime",
]

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_labels(antibiotic: str, limit: int = 10000) -> pd.DataFrame:
    from urllib.parse import quote
    ab_encoded = quote(antibiotic, safe="")
    url = (
        f"{BV_BRC_API}/genome_amr/"
        f"?eq(taxon_id,{TAXON_ID})"
        f"&eq(antibiotic,{ab_encoded})"
        f"&select(genome_id,resistant_phenotype,genome_name)"
        f"&limit({limit})"
    )
    resp = requests.get(url, headers={"Accept": "application/json"}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame(columns=["genome_id", "resistant_phenotype", "genome_name"])
    df = pd.DataFrame(data)
    df = df[df["resistant_phenotype"].isin(["Resistant", "Susceptible"])].copy()
    df["label"] = df["resistant_phenotype"].map({"Resistant": "R", "Susceptible": "S"})
    return df.drop_duplicates("genome_id")


def balance(df: pd.DataFrame, n: int) -> pd.DataFrame:
    parts = []
    for label in ["R", "S"]:
        sub = df[df["label"] == label]
        parts.append(sub.sample(min(n, len(sub)), random_state=42))
    return pd.concat(parts).reset_index(drop=True)


def download_fasta(genome_id: str, retries: int = 3) -> bool:
    out = RAW_DIR / f"{genome_id}.fasta"
    if out.exists() and out.stat().st_size > 0:
        return True
    for attempt in range(retries):
        try:
            resp = requests.get(FASTA_URL.format(gid=genome_id), timeout=60)
            if resp.status_code == 200 and resp.text.strip().startswith(">"):
                out.write_text(resp.text)
                return True
            time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return False


def main():
    all_genome_ids = set()

    # ── 1. Fetch and save labels per antibiotic ───────────────────────────────
    for ab in ANTIBIOTICS:
        out_dir = PROC_DIR / ab
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "metadata.csv"

        if meta_path.exists():
            print(f"[{ab}] labels already downloaded — skipping.")
            df = pd.read_csv(meta_path)
        else:
            print(f"[{ab}] fetching labels ...")
            df_raw = fetch_labels(ab)
            df = balance(df_raw, N_PER_CLASS)
            df[["genome_id", "genome_name", "label"]].to_csv(meta_path, index=False)
            r = (df["label"] == "R").sum()
            s = (df["label"] == "S").sum()
            print(f"  → {len(df)} genomes  (R={r}, S={s})  saved to {meta_path}")

        all_genome_ids.update(df["genome_id"].astype(str).tolist())

    print(f"\nTotal unique genomes across all antibiotics: {len(all_genome_ids)}")

    # ── 2. Download missing FASTA files ───────────────────────────────────────
    already = {p.stem for p in RAW_DIR.glob("*.fasta")}
    to_fetch = [gid for gid in all_genome_ids if gid not in already]
    print(f"Already downloaded: {len(already)}  |  Need to fetch: {len(to_fetch)}")

    if to_fetch:
        failed = []
        for gid in tqdm(to_fetch, desc="Downloading FASTAs"):
            if not download_fasta(gid):
                failed.append(gid)
            time.sleep(0.1)
        print(f"Done. {len(to_fetch)-len(failed)}/{len(to_fetch)} new genomes downloaded.")
        if failed:
            print(f"Failed ({len(failed)}): {failed[:5]}{'...' if len(failed)>5 else ''}")
    else:
        print("All FASTA files already present — nothing to download.")


if __name__ == "__main__":
    main()
