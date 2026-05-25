"""
MLST (Multi-Locus Sequence Typing) analysis.

MLST groups bacteria into sequence types (STs) based on allele combinations
at 7 conserved housekeeping genes. ST = clonal lineage = outbreak cluster.

K. pneumoniae has two major pandemic lineages:
  ST258  — dominant carbapenem-resistant lineage globally
  ST11   — dominant in Asia, especially China
  ST307  — emerging, associated with hypervirulence + MDR

We fetch ST data from BV-BRC genome metadata, then:
  A. Which STs are most common in our dataset?
  B. Which STs have the highest resistance rates per antibiotic?
  C. Which STs are spreading — are outbreak STs rising over time?
  D. ST × resistance gene enrichment (which genes define each lineage)

Outputs:
  artifacts/mlst_analysis.json
"""

import json
import time
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests

ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
ART_DIR  = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

BV_BRC_API = "https://www.bv-brc.org/api"

ANTIBIOTICS = [
    "ciprofloxacin", "meropenem", "gentamicin", "tetracycline",
    "trimethoprim/sulfamethoxazole", "cefepime",
    "amikacin", "imipenem", "piperacillin/tazobactam", "levofloxacin",
]

SHORT = {
    "ciprofloxacin": "Cipro", "meropenem": "Mero",
    "gentamicin": "Gent", "tetracycline": "Tet",
    "trimethoprim/sulfamethoxazole": "TMP/SMX", "cefepime": "Cef",
    "amikacin": "Amik", "imipenem": "Imi",
    "piperacillin/tazobactam": "Pip/Taz", "levofloxacin": "Levo",
}

# Clinically important K. pneumoniae STs
NOTABLE_STs = {
    "258":  "ST258 — dominant carbapenem-resistant lineage (global)",
    "11":   "ST11  — dominant MDR lineage in Asia/China",
    "307":  "ST307 — emerging, hypervirulence + MDR",
    "147":  "ST147 — OXA-48 carbapenemase carrier",
    "15":   "ST15  — ESBL, spreading in Europe",
    "101":  "ST101 — KPC producer",
    "512":  "ST512 — KPC-3 epidemic",
    "45":   "ST45  — aminoglycoside resistance",
}


def fetch_mlst_data(genome_ids: list[str], cache_path: Path) -> dict[str, str]:
    """Fetch sequence_type for each genome from BV-BRC. Cached."""
    if cache_path.exists():
        print(f"  Loading cached MLST data from {cache_path} ...")
        return json.loads(cache_path.read_text())

    print(f"  Fetching MLST data for {len(genome_ids)} genomes ...")
    st_map = {}
    batch_size = 200
    batches = [genome_ids[i:i + batch_size]
               for i in range(0, len(genome_ids), batch_size)]

    for i, batch in enumerate(batches):
        ids_str = ",".join(batch)
        url = (f"{BV_BRC_API}/genome/"
               f"?in(genome_id,({ids_str}))"
               f"&select(genome_id,mlst,sequence_type)"
               f"&limit({batch_size})")
        try:
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
            for rec in r.json():
                # BV-BRC: mlst field is "MLST.klebsiella.258" → extract "258"
                # sequence_type field is "ST258" or "258"
                raw = rec.get("mlst") or rec.get("sequence_type")
                if raw:
                    # "MLST.klebsiella.258" → "258"
                    # "ST258" → "258"
                    # "258"   → "258"
                    st_str = str(raw).split(".")[-1].replace("ST","").replace("st","").strip()
                    if st_str.isdigit():
                        st_map[str(rec["genome_id"])] = st_str
        except Exception as e:
            print(f"  Batch {i} failed: {e}")
        if i % 10 == 0:
            print(f"  {i + 1}/{len(batches)} batches — {len(st_map)} STs found")
        time.sleep(0.05)

    cache_path.write_text(json.dumps(st_map))
    print(f"  Done. {len(st_map)}/{len(genome_ids)} genomes have ST data.")
    return st_map


def load_all_labels() -> pd.DataFrame:
    """Merge resistance labels for all antibiotics."""
    dfs = []
    for ab in ANTIBIOTICS:
        safe = ab.replace("/", "_").replace(" ", "_")
        p = PROC_DIR / safe / "metadata.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["genome_id"] = df["genome_id"].astype(str)
        df = df.drop_duplicates("genome_id")[["genome_id", "label"]].copy()
        df = df.rename(columns={"label": ab})
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="genome_id", how="outer")
    return merged


def main():
    print("=" * 60)
    print("MLST SEQUENCE TYPE ANALYSIS")
    print("=" * 60)

    # Collect all genome IDs
    all_ids = set()
    for ab in ANTIBIOTICS:
        safe = ab.replace("/", "_").replace(" ", "_")
        p = PROC_DIR / safe / "metadata.csv"
        if p.exists():
            df = pd.read_csv(p)
            all_ids.update(df["genome_id"].astype(str).tolist())
    print(f"\nTotal unique genomes: {len(all_ids)}")

    # A. Fetch MLST data
    print("\n[A] Fetching sequence type (ST) data ...")
    st_map = fetch_mlst_data(list(all_ids), ART_DIR / "mlst_data.json")
    print(f"\n  STs found: {len(set(st_map.values()))} unique sequence types")

    # B. Load labels + merge
    print("\n[B] Building ST resistance profiles ...")
    labels = load_all_labels()
    labels["genome_id"] = labels["genome_id"].astype(str)
    labels["ST"] = labels["genome_id"].map(st_map)
    labels = labels[labels["ST"].notna()].copy()
    print(f"  {len(labels)} genomes with ST data")

    # Binary resistance columns
    for ab in ANTIBIOTICS:
        if ab in labels.columns:
            labels[ab + "_bin"] = labels[ab].map({"R": 1, "S": 0})
    bin_cols = [ab + "_bin" for ab in ANTIBIOTICS if ab + "_bin" in labels.columns]
    labels["n_resistant"] = labels[bin_cols].sum(axis=1, min_count=1)
    labels["is_mdr"]      = labels["n_resistant"] >= 3

    # C. Top STs by count
    print("\n[C] Most common sequence types ...")
    st_counts = labels["ST"].value_counts()
    top_sts = st_counts.head(20)
    print(f"  {'ST':>6}  {'N':>6}  {'Notable'}")
    print("  " + "-" * 60)
    for st, n in top_sts.items():
        note = NOTABLE_STs.get(str(st), "")
        print(f"  ST{st:>5}  {n:>6}  {note}")

    # D. Per-ST resistance profiles (STs with >= 15 genomes)
    print("\n[D] Resistance rates by ST ...")
    min_n = 15
    st_profiles = []
    for st, grp in labels.groupby("ST"):
        if len(grp) < min_n:
            continue
        profile = {
            "st":         str(st),
            "n_genomes":  int(len(grp)),
            "pct_mdr":    round(float(grp["is_mdr"].mean() * 100), 1),
            "mean_drugs": round(float(grp["n_resistant"].mean()), 2),
            "notable":    NOTABLE_STs.get(str(st), ""),
        }
        for ab in ANTIBIOTICS:
            bc = ab + "_bin"
            if bc not in grp.columns:
                continue
            sub = grp[grp[bc].notna()]
            if len(sub) >= 5:
                profile[SHORT[ab] + "_pct_R"] = round(float(sub[bc].mean() * 100), 1)
                profile[SHORT[ab] + "_n"]     = int(len(sub))
            else:
                profile[SHORT[ab] + "_pct_R"] = None
        st_profiles.append(profile)

    st_profiles.sort(key=lambda x: -x["n_genomes"])
    print(f"  STs with >= {min_n} genomes: {len(st_profiles)}")
    print(f"\n  {'ST':>6} {'N':>6} {'MDR%':>6} {'Cipro%':>8} {'Mero%':>7}")
    print("  " + "-" * 45)
    for p in st_profiles[:15]:
        cipro = p.get("Cipro_pct_R")
        mero  = p.get("Mero_pct_R")
        print(f"  ST{p['st']:>5} {p['n_genomes']:>6} {p['pct_mdr']:>5.1f}%"
              f" {str(cipro or '?'):>7}% {str(mero or '?'):>6}%")

    # E. ST × year trends (are epidemic STs rising?)
    print("\n[E] ST frequency trends over time ...")
    years_path = ART_DIR / "temporal_years.json"
    st_year_trends = {}
    if years_path.exists():
        year_map = {k: int(v) for k, v in json.loads(years_path.read_text()).items()
                    if 2000 <= int(v) <= 2024}
        labels["year"] = labels["genome_id"].map(year_map)
        all_by_year = labels[labels["year"].notna()].groupby("year").size()

        for st in [p["st"] for p in st_profiles[:10]]:
            sub = labels[(labels["ST"] == st) & labels["year"].notna()]
            if len(sub) < 10:
                continue
            rows = []
            for yr, grp in sub.groupby("year"):
                total_yr = all_by_year.get(yr, 1)
                rows.append({
                    "year": int(yr),
                    "n": int(len(grp)),
                    "pct_of_year": round(float(len(grp) / total_yr * 100), 2),
                })
            rows.sort(key=lambda x: x["year"])
            if len(rows) >= 3:
                st_year_trends[st] = rows
                first, last = rows[0], rows[-1]
                print(f"    ST{st}: {first['year']} ({first['pct_of_year']:.1f}%) → "
                      f"{last['year']} ({last['pct_of_year']:.1f}%)")

    # F. ST × gene enrichment (top MDR STs vs background)
    print("\n[F] Gene enrichment in top MDR STs ...")
    gene_path = PROC_DIR / "gene_matrix.csv"
    st_gene_enrichment = {}
    if gene_path.exists():
        X_gene = pd.read_csv(gene_path, index_col=0)
        X_gene.index = X_gene.index.astype(str)
        X_gene = X_gene.drop(columns=["__label__"], errors="ignore")

        # Top 5 STs by MDR rate (with enough genomes)
        mdr_sts = sorted(
            [p for p in st_profiles if p["n_genomes"] >= 20],
            key=lambda x: -x["pct_mdr"]
        )[:5]

        for p in mdr_sts:
            st = p["st"]
            st_ids = labels[labels["ST"] == st]["genome_id"].tolist()
            other_ids = labels[labels["ST"] != st]["genome_id"].tolist()
            st_genes   = X_gene.loc[X_gene.index.intersection(st_ids)]
            other_genes = X_gene.loc[X_gene.index.intersection(other_ids)]
            if len(st_genes) < 5 or len(other_genes) < 5:
                continue
            enrich = (st_genes.mean() - other_genes.mean()).sort_values(ascending=False)
            top_enriched = [
                {"gene": g, "enrichment": round(float(v), 3)}
                for g, v in enrich.head(10).items()
            ]
            st_gene_enrichment[st] = top_enriched
            top_gene = top_enriched[0]["gene"][:40] if top_enriched else "?"
            print(f"    ST{st} (MDR={p['pct_mdr']:.0f}%): top gene = {top_gene}")

    # Save
    out = {
        "st_profiles":       st_profiles,
        "st_year_trends":    st_year_trends,
        "st_gene_enrichment": st_gene_enrichment,
        "top_sts":           {str(st): int(n) for st, n in top_sts.items()},
        "notable_sts":       NOTABLE_STs,
        "antibiotics":       ANTIBIOTICS,
        "short_names":       SHORT,
        "total_with_st":     len(labels),
        "unique_sts":        len(set(st_map.values())),
    }
    (ART_DIR / "mlst_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved → artifacts/mlst_analysis.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if st_profiles:
        most_mdr = max(st_profiles, key=lambda x: x["pct_mdr"])
        largest  = st_profiles[0]
        print(f"\n  Largest ST:   ST{largest['st']} ({largest['n_genomes']} genomes)")
        print(f"  Highest MDR:  ST{most_mdr['st']} ({most_mdr['pct_mdr']:.1f}% MDR)")
        print(f"  Total STs with data: {len(st_profiles)}")


if __name__ == "__main__":
    main()
