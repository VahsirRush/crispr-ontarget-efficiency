"""Phase 1: build and verify the dataset, then freeze the splits.

Outputs to artifacts/:
  dataset.parquet   - guides + target + split assignments
  X_rs2.npy         - Rule Set 2 hand-crafted feature matrix
  X_rs2_names.json  - feature names
  X_onehot.npy      - (N, 30, 4) sequence tensor
  X_thermo.npy      - thermodynamic/gene-position block for the CNN fusion step
  splits.json       - which genes went where, and why
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import data as D
from crispr import features as F

ART = os.path.join(D.REPO_ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)


def main():
    print("=" * 72)
    print("PHASE 1  DATA PIPELINE")
    print("=" * 72)

    df = D.load_v3(verify=True)
    print(f"\nLoaded FC+RES (V3): {len(df)} rows, {df['Target gene'].nunique()} genes")
    print(f"  unique 30mers : {df['30mer'].nunique()}")
    print(f"  FC  (Doench 2014 flow cytometry) : {(df.dataset=='FC').sum():>5}")
    print(f"  RES (Doench 2016 resistance)     : {(df.dataset=='RES').sum():>5}")
    print(f"  target = {D.TARGET}, range "
          f"[{df[D.TARGET].min():.4f}, {df[D.TARGET].max():.4f}]")

    print("\n-- verifying shipped CSV against raw V1/V2 Excel ------------------")
    v = D.verify_against_xlsx(df)
    for k, val in v.items():
        print(f"  {k}: {val}")
    if v["n_matched"] > 0 and v["max_abs_diff"] < 1e-6:
        print("  OK: shipped CSV target reproduces exactly from the raw source files.")
    else:
        print("  WARNING: reconstruction does not match exactly; see numbers above.")

    # ---- duplicate / leakage audit ---------------------------------------
    dup = df["30mer"].duplicated(keep=False)
    print(f"\n-- duplicate-sequence audit ---------------------------------------")
    print(f"  rows whose 30mer appears more than once: {dup.sum()}")
    if dup.any():
        d = df[dup].groupby("30mer")["Target gene"].nunique()
        print(f"  those sequences span {d.max()} gene(s) at most "
              f"(same guide re-measured under a second drug)")
        print(f"  affected genes: {sorted(df[dup]['Target gene'].unique())}")

    # ---- features ---------------------------------------------------------
    print("\n-- featurizing -----------------------------------------------------")
    X_rs2, rs2_names = F.rule_set_2_features(df)
    print(f"  Rule Set 2 features : {X_rs2.shape}")
    X_oh = F.one_hot(df)
    print(f"  one-hot sequence    : {X_oh.shape}")
    X_th, th_names = F.thermodynamic_features(df)
    print(f"  thermodynamic block : {X_th.shape}  {th_names}")
    assert np.isfinite(X_rs2).all() and np.isfinite(X_th).all(), "non-finite features"
    # Sanity: one-hot must have exactly one base set per position.
    assert (X_oh.sum(axis=2) == 1).all()

    # ---- splits -----------------------------------------------------------
    print("\n-- splits ----------------------------------------------------------")
    gene_split, assign = D.gene_grouped_split(df, test_frac=0.2, cal_frac=0.2)
    df["split_gene"] = gene_split
    df["split_random"] = D.random_split(df, test_frac=0.2, cal_frac=0.2, seed=0)

    print("  gene-held-out split (primary):")
    for s in ["train", "cal", "test"]:
        genes = sorted(str(g) for g, v in assign.items() if v == s)
        n = (gene_split == s).sum()
        print(f"    {s:<6} {n:>5} rows ({n/len(df):5.1%})  genes={genes}")
    print("  random row-level split (leakage contrast):")
    print("   ", df["split_random"].value_counts().to_dict())

    n_logo = sum(1 for _ in D.leave_one_gene_out_folds(df))
    print(f"  leave-one-gene-out folds with >=10 guides: {n_logo}")

    # ---- save -------------------------------------------------------------
    df.to_csv(os.path.join(ART, "dataset.csv"), index=False)
    np.save(os.path.join(ART, "X_rs2.npy"), X_rs2)
    np.save(os.path.join(ART, "X_onehot.npy"), X_oh)
    np.save(os.path.join(ART, "X_thermo.npy"), X_th)
    with open(os.path.join(ART, "X_rs2_names.json"), "w") as f:
        json.dump(rs2_names, f)
    with open(os.path.join(ART, "splits.json"), "w") as f:
        json.dump({
            "strategy": "gene-held-out (primary); leave-one-gene-out CV for the "
                        "paper-comparable benchmark; random split reported only as "
                        "a leakage contrast",
            "gene_assignment": assign,
            "thermo_feature_names": th_names,
            "verification": {k: v for k, v in v.items()},
        }, f, indent=2)
    print(f"\nWrote artifacts to {ART}/")


if __name__ == "__main__":
    main()
