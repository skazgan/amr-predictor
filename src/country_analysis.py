"""
Country-level resistance analysis.

Questions:
  1. Which countries have the highest resistance rates per antibiotic?
  2. Are certain regions hotspots for MDR strains?
  3. Is resistance rising faster in some countries than others?
  4. Which countries contribute most genomes to global databases?

Steps:
  A. Fetch isolation_country for all genome IDs from BV-BRC
  B. Build per-country resistance profiles
  C. Compute geographic MDR burden
  D. Track resistance trends within high-data countries

Outputs:
  artifacts/country_resistance.json  — resistance rates by country
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import linregress

ROOT     = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
ART_DIR  = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

BV_BRC_API = "https://www.bv-brc.org/api"

ANTIBIOTICS = [
    "ciprofloxacin",
    "meropenem",
    "gentamicin",
    "tetracycline",
    "trimethoprim/sulfamethoxazole",
    "cefepime",
]

SHORT = {
    "ciprofloxacin":                "Cipro",
    "meropenem":                    "Mero",
    "gentamicin":                   "Gent",
    "tetracycline":                 "Tet",
    "trimethoprim/sulfamethoxazole":"TMP/SMX",
    "cefepime":                     "Cef",
}

# Country name standardisation (BV-BRC sometimes uses different names)
COUNTRY_NORMALIZE = {
    "USA": "United States",
    "United States of America": "United States",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "People's Republic of China": "China",
    "Republic of Korea": "South Korea",
    "Korea": "South Korea",
}


def fetch_country_data(genome_ids: list[str], cache_path: Path) -> dict[str, str]:
    """Fetch isolation_country for each genome ID from BV-BRC. Cached."""
    if cache_path.exists():
        print(f"  Loading cached countries from {cache_path} ...")
        return json.loads(cache_path.read_text())

    print(f"  Fetching countries for {len(genome_ids)} genomes ...")
    countries = {}
    batch_size = 200
    batches = [genome_ids[i:i+batch_size]
               for i in range(0, len(genome_ids), batch_size)]

    for i, batch in enumerate(batches):
        ids_str = ",".join(batch)
        url = (f"{BV_BRC_API}/genome/"
               f"?in(genome_id,({ids_str}))"
               f"&select(genome_id,isolation_country)"
               f"&limit({batch_size})")
        try:
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
            for rec in r.json():
                country = rec.get("isolation_country")
                if country and isinstance(country, str) and len(country) > 1:
                    gid = str(rec["genome_id"])
                    # Normalize
                    country = COUNTRY_NORMALIZE.get(country, country)
                    countries[gid] = country
        except Exception as e:
            print(f"  Batch {i} failed: {e}")
        if i % 10 == 0:
            print(f"  {i+1}/{len(batches)} batches done — {len(countries)} countries found")
        time.sleep(0.05)

    cache_path.write_text(json.dumps(countries))
    print(f"  Done. {len(countries)}/{len(genome_ids)} genomes have country data.")
    return countries


def load_all_labels() -> pd.DataFrame:
    """Load and merge resistance labels for all antibiotics."""
    dfs = []
    for ab in ANTIBIOTICS:
        safe = ab.replace("/", "_").replace(" ", "_")
        p = PROC_DIR / safe / "metadata.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["genome_id"] = df["genome_id"].astype(str)
        df = df.drop_duplicates("genome_id")
        df = df[["genome_id", "label"]].copy()
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
    print("COUNTRY-LEVEL RESISTANCE ANALYSIS")
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

    # A. Fetch countries
    print("\n[A] Fetching isolation countries ...")
    countries = fetch_country_data(
        list(all_ids),
        cache_path=ART_DIR / "country_data.json"
    )

    # Summary
    country_counts = pd.Series(list(countries.values())).value_counts()
    print(f"\n  Top countries by genome count:")
    for c, n in country_counts.head(15).items():
        print(f"    {c:<30} {n:>5}")

    # B. Load labels + merge with countries
    print("\n[B] Building per-country resistance profiles ...")
    labels = load_all_labels()
    labels["genome_id"] = labels["genome_id"].astype(str)
    labels["country"] = labels["genome_id"].map(countries)
    labels = labels[labels["country"].notna()].copy()
    print(f"  {len(labels)} genomes with country data")

    # Convert to binary
    for ab in ANTIBIOTICS:
        if ab in labels.columns:
            labels[ab + "_bin"] = labels[ab].map({"R": 1, "S": 0})

    bin_cols = [ab + "_bin" for ab in ANTIBIOTICS if ab + "_bin" in labels.columns]
    labels["n_resistant"] = labels[bin_cols].sum(axis=1, min_count=1)
    labels["is_mdr"]      = labels["n_resistant"] >= 3

    # Per-country per-drug resistance rates (only countries with >= 10 genomes)
    min_genomes = 10
    country_profiles = []
    for country, grp in labels.groupby("country"):
        if len(grp) < min_genomes:
            continue
        profile = {
            "country":      country,
            "n_genomes":    int(len(grp)),
            "pct_mdr":      round(float(grp["is_mdr"].mean() * 100), 1),
            "mean_drugs_r": round(float(grp["n_resistant"].mean()), 2),
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
                profile[SHORT[ab] + "_n"]     = 0
        country_profiles.append(profile)

    country_profiles.sort(key=lambda x: -x["n_genomes"])
    print(f"\n  Countries with >= {min_genomes} genomes: {len(country_profiles)}")
    print(f"\n  {'Country':<30} {'N':>6} {'MDR%':>6} {'Cipro%':>8} {'Mero%':>7} {'Gent%':>7}")
    print("  " + "-" * 70)
    for p in country_profiles[:20]:
        cipro = p.get("Cipro_pct_R")
        mero  = p.get("Mero_pct_R")
        gent  = p.get("Gent_pct_R")
        print(f"  {p['country']:<30} {p['n_genomes']:>6} {p['pct_mdr']:>5.1f}% "
              f"{str(cipro or '?'):>7}% {str(mero or '?'):>6}% {str(gent or '?'):>6}%")

    # C. Geographic heatmap data (world map)
    # Compute a single "overall resistance index" per country
    for p in country_profiles:
        r_rates = [p.get(SHORT[ab] + "_pct_R") for ab in ANTIBIOTICS
                   if p.get(SHORT[ab] + "_pct_R") is not None]
        p["mean_resistance_pct"] = round(float(np.mean(r_rates)), 1) if r_rates else None

    # D. Year × Country trends (for top 5 countries)
    print("\n[C] Year × country trends ...")
    years_path = ART_DIR / "temporal_years.json"
    country_year_trends = {}
    if years_path.exists():
        year_map = json.loads(years_path.read_text())
        year_map = {k: int(v) for k, v in year_map.items() if 2000 <= int(v) <= 2024}
        labels["year"] = labels["genome_id"].map(year_map)

        top_countries = [p["country"] for p in country_profiles[:6]]
        for country in top_countries:
            sub = labels[(labels["country"] == country) & labels["year"].notna()].copy()
            sub["year"] = sub["year"].astype(int)
            rows = []
            for yr, grp in sub.groupby("year"):
                if len(grp) < 5:
                    continue
                row = {"year": int(yr), "n": int(len(grp)),
                       "pct_mdr": round(float(grp["is_mdr"].mean() * 100), 1)}
                for ab in ANTIBIOTICS:
                    bc = ab + "_bin"
                    if bc in grp.columns:
                        s = grp[grp[bc].notna()]
                        row[SHORT[ab]] = round(float(s[bc].mean() * 100), 1) if len(s) >= 3 else None
                rows.append(row)
            rows.sort(key=lambda x: x["year"])
            if rows:
                country_year_trends[country] = rows
                print(f"    {country}: {len(rows)} years of data")

    # Save
    out = {
        "country_profiles":    country_profiles,
        "country_year_trends": country_year_trends,
        "antibiotics":         ANTIBIOTICS,
        "short_names":         SHORT,
        "min_genomes":         min_genomes,
    }
    (ART_DIR / "country_resistance.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved to artifacts/country_resistance.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if country_profiles:
        # Highest MDR
        mdr_sorted = sorted([p for p in country_profiles if p["pct_mdr"] is not None],
                            key=lambda x: -x["pct_mdr"])
        print(f"\n  Highest MDR rate: {mdr_sorted[0]['country']} ({mdr_sorted[0]['pct_mdr']:.1f}%)")
        print(f"  Lowest MDR rate:  {mdr_sorted[-1]['country']} ({mdr_sorted[-1]['pct_mdr']:.1f}%)")

        # Most genomes
        print(f"\n  Largest contributor: {country_profiles[0]['country']} "
              f"({country_profiles[0]['n_genomes']} genomes)")


if __name__ == "__main__":
    main()
