"""
Fetch and train models for Pseudomonas aeruginosa and Enterococcus faecium.

Both are WHO critical/high priority pathogens with data on BV-BRC.

Run:
  python src/fetch_new_organisms.py        # fetch data only
  python src/fetch_new_organisms.py --train  # fetch + train

Taxon IDs:
  Pseudomonas aeruginosa  → 287
  Enterococcus faecium    → 1352
"""
import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
ART_DIR   = ROOT / "artifacts"

# ── New organism registry ─────────────────────────────────────────────────────
NEW_ORGANISMS = {
    "pseudomonas_aeruginosa": {
        "display":   "Pseudomonas aeruginosa",
        "taxon_id":  287,
        "antibiotics": [
            "ciprofloxacin",
            "meropenem",
            "piperacillin/tazobactam",
            "ceftazidime",
            "imipenem",
            "gentamicin",
            "amikacin",
            "colistin",
        ],
    },
    "enterococcus_faecium": {
        "display":   "Enterococcus faecium",
        "taxon_id":  1352,
        "antibiotics": [
            "vancomycin",
            "ampicillin",
            "tetracycline",
            "ciprofloxacin",
            "gentamicin",
            "erythromycin",
            "linezolid",
            "daptomycin",
        ],
    },
}

BASE_URL = "https://www.bv-brc.org/api"
HEADERS  = {"Accept": "application/json"}
BATCH    = 5000
MIN_SAMPLES = 100


# ── BV-BRC API helpers ─────────────────────────────────────────────────────────
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
    df = df[df["resistant_phenotype"].isin(["Resistant", "Susceptible"])].copy()
    df["antibiotic"] = df["antibiotic"].str.lower().str.strip()
    return df[["genome_id", "antibiotic", "resistant_phenotype"]]


def fetch_genes(genome_ids: list) -> pd.DataFrame:
    print(f"  Fetching genes for {len(genome_ids):,} genomes...")
    all_rows = []
    batch_size = 200
    for i in range(0, len(genome_ids), batch_size):
        batch = genome_ids[i:i + batch_size]
        id_list = ",".join(str(g) for g in batch)
        params = (
            f"in(genome_id,({id_list}))"
            f"&in(source,(CARD,NDARO))"
            f"&select(genome_id,gene,product,source)"
        )
        rows = bvbrc_get("sp_gene", params, limit=10000)
        all_rows.extend(rows)
        if i % 2000 == 0:
            print(f"    genes: {i}/{len(genome_ids)} genomes...", end="\r")
    print()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.columns = df.columns.str.lower()
    df["gene"] = df["gene"].fillna(df.get("product", "unknown")).fillna("unknown")
    df["gene"] = df["gene"].str.strip().str.replace(" ", "_")
    df["present"] = 1

    matrix = (df.groupby(["genome_id", "gene"])["present"]
                .max()
                .unstack(fill_value=0))
    return matrix


# ── Training ───────────────────────────────────────────────────────────────────
def train_organism(org_name: str, org_info: dict):
    print(f"\n{'='*60}")
    print(f"TRAINING: {org_info['display']}")
    print(f"{'='*60}")

    out_dir   = DATA_DIR / org_name
    model_dir = MODEL_DIR / org_name
    model_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "labels.csv"
    genes_path  = out_dir / "genes.csv"

    if not labels_path.exists() or not genes_path.exists():
        print(f"  ERROR: Run fetch first (missing {labels_path} or {genes_path})")
        return []

    labels = pd.read_csv(labels_path, dtype={"genome_id": str})
    labels["antibiotic"] = labels["antibiotic"].str.lower().str.strip()
    labels = labels[labels["antibiotic"].isin(org_info["antibiotics"])]

    print("  Loading gene matrix...")
    genes = pd.read_csv(genes_path, index_col=0, dtype={"genome_id": str})
    genes.index = genes.index.astype(str)

    results = []

    for ab in org_info["antibiotics"]:
        safe = ab.replace("/", "_").replace(" ", "_")
        model_path = model_dir / f"{safe}.pkl"

        sub = labels[labels["antibiotic"] == ab].copy()
        sub["y"] = (sub["resistant_phenotype"] == "Resistant").astype(int)
        sub = sub.drop_duplicates("genome_id").set_index("genome_id")

        common = sub.index.intersection(genes.index)
        if len(common) < MIN_SAMPLES:
            print(f"  SKIP {ab}: only {len(common)} samples")
            continue

        X  = genes.loc[common].fillna(0).astype(float)
        y  = sub.loc[common, "y"].values
        n_r = int(y.sum())
        n_s = int((y == 0).sum())

        if n_r < 20 or n_s < 20:
            print(f"  SKIP {ab}: too imbalanced (R={n_r}, S={n_s})")
            continue

        print(f"\n  [{ab}]  n={len(y):,}  R={n_r:,}  S={n_s:,}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X.values, y, test_size=0.2, random_state=42, stratify=y
        )

        scale = max(n_s / max(n_r, 1), 1)
        xgb = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        model = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, probs)
        print(f"    Holdout AUC: {auc:.3f}")

        model_final = CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            ),
            method="isotonic", cv=3,
        )
        model_final.fit(X.values, y)

        bundle = {
            "model":         model_final,
            "features":      [f"gene__{c}" for c in X.columns],
            "antibiotic":    ab,
            "organism":      org_name,
            "test_auc":      round(auc, 3),
            "n_total":       len(y),
            "n_resistant":   n_r,
            "n_susceptible": n_s,
            "feature_type":  "gene_only",
        }

        with open(model_path, "wb") as f:
            pickle.dump(bundle, f, protocol=4)

        size_mb = model_path.stat().st_size / 1e6
        print(f"    Saved → {model_path}  ({size_mb:.1f} MB)")

        results.append({
            "organism":    org_name,
            "antibiotic":  ab,
            "test_auc":    round(auc, 3),
            "n_total":     len(y),
            "n_resistant": n_r,
        })

    return results


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Also train models after fetching")
    parser.add_argument("--only", choices=list(NEW_ORGANISMS.keys()), help="Process only one organism")
    args = parser.parse_args()

    orgs = {args.only: NEW_ORGANISMS[args.only]} if args.only else NEW_ORGANISMS

    print("BV-BRC DATA FETCH — NEW ORGANISMS")
    print("P. aeruginosa + E. faecium\n")

    # ── Fetch data ────────────────────────────────────────────────────────────
    for name, info in orgs.items():
        print(f"\n{'='*60}")
        print(f"FETCHING: {info['display']} (taxon {info['taxon_id']})")
        print(f"{'='*60}")

        out_dir = DATA_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        labels_path = out_dir / "labels.csv"
        if labels_path.exists():
            print(f"  Labels cached: {labels_path}")
            df_labels = pd.read_csv(labels_path, dtype={"genome_id": str})
        else:
            df_labels = fetch_amr_labels(info["taxon_id"])
            if df_labels.empty:
                print("  ERROR: No AMR data found.")
                continue
            df_labels.to_csv(labels_path, index=False)
            print(f"  Saved {len(df_labels):,} rows → {labels_path}")

        df_labels = df_labels[df_labels["antibiotic"].isin(info["antibiotics"])]
        genome_ids = df_labels["genome_id"].unique().tolist()
        print(f"  Genomes with target antibiotic data: {len(genome_ids):,}")

        for ab in sorted(info["antibiotics"]):
            sub = df_labels[df_labels["antibiotic"] == ab]
            n_r = (sub["resistant_phenotype"] == "Resistant").sum()
            n_s = (sub["resistant_phenotype"] == "Susceptible").sum()
            if n_r + n_s > 0:
                print(f"    {ab:<40} R={n_r:>5,}  S={n_s:>5,}")
            else:
                print(f"    {ab:<40} (no data)")

        genes_path = out_dir / "genes.csv"
        if genes_path.exists():
            print(f"  Genes cached: {genes_path}")
        else:
            gene_matrix = fetch_genes(genome_ids)
            if not gene_matrix.empty:
                gene_matrix.to_csv(genes_path)
                print(f"  Saved gene matrix {gene_matrix.shape} → {genes_path}")
            else:
                print("  WARNING: No gene data retrieved.")

    # ── Update organisms.json ─────────────────────────────────────────────────
    reg_path = ART_DIR / "organisms.json"
    registry = json.loads(reg_path.read_text())
    for name, info in orgs.items():
        registry[name] = {
            "display":     info["display"],
            "taxon_id":    info["taxon_id"],
            "antibiotics": info["antibiotics"],
        }
    reg_path.write_text(json.dumps(registry, indent=2))
    print(f"\nUpdated organisms.json with {list(orgs.keys())}")

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.train:
        print("\n" + "=" * 60)
        print("TRAINING NEW ORGANISM MODELS")
        print("=" * 60)
        all_results = []
        for name, info in orgs.items():
            results = train_organism(name, info)
            all_results.extend(results)

        # Update summary
        summary_path = ART_DIR / "multi_org_summary.json"
        existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
        trained_orgs = {r["organism"] for r in all_results}
        existing = [r for r in existing if r["organism"] not in trained_orgs]
        existing.extend(all_results)
        summary_path.write_text(json.dumps(existing, indent=2))
        print(f"\nUpdated multi_org_summary.json")

        print(f"\n{'Organism':<28} {'Antibiotic':<38} {'AUC':>6} {'N':>7}")
        print("-" * 80)
        for r in sorted(all_results, key=lambda x: (x["organism"], -x["test_auc"])):
            print(f"{r['organism']:<28} {r['antibiotic']:<38} {r['test_auc']:>6.3f} {r['n_total']:>7,}")
    else:
        print("\nData fetched. Run with --train to train models.")


if __name__ == "__main__":
    main()
