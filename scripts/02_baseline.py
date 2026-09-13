"""Phase 2: GBT baseline, and the Spearman checkpoint against the literature.

Runs three evaluation protocols so the headline number is interpretable:

  LOGO       leave-one-gene-out CV -- Azimuth's default and how Doench 2016
             Fig. 4c reports Spearman. This is THE paper-comparable number.
  GENE-HELD  the frozen train/cal/test gene split, used later for conformal.
  RANDOM     row-level random split, reported only to quantify how much a
             naive split inflates the result.

Also scores the ``predictions`` column shipped in the Azimuth CSV, which is what
the released Rule Set 2 model outputs on its own training data.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import baseline as B
from crispr import data as D
from crispr import evaluate as E

ART = os.path.join(D.REPO_ROOT, "artifacts")


def logo_predictions(make_model, X, y, df):
    """Out-of-fold predictions under leave-one-gene-out."""
    oof = np.full(len(y), np.nan)
    for gene, tr, te in D.leave_one_gene_out_folds(df):
        oof[te] = B.fit_predict(make_model(), X[tr], y[tr], X[te])
    return oof


def main():
    print("=" * 72)
    print("PHASE 2  GRADIENT-BOOSTED-TREE BASELINE (Rule Set 2 reproduction)")
    print("=" * 72)

    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    X = np.load(os.path.join(ART, "X_rs2.npy"))
    y = df[D.TARGET].values
    print(f"\n{len(df)} guides, {X.shape[1]} Rule Set 2 features, "
          f"{df['Target gene'].nunique()} genes")

    results = {}

    # ---- reference: the shipped Azimuth predictions -----------------------
    print("\n-- reference point: Azimuth's own shipped predictions -------------")
    s = E.summarize(df, y, df["azimuth_prediction"].values, "Azimuth shipped predictions")
    results["azimuth_shipped"] = s
    print(E.format_summary(s))
    print("   NOTE: the released V3 model was trained on this same data, so this")
    print("   is an IN-SAMPLE number and an optimistic ceiling, not a benchmark.")

    # ---- LOGO CV ----------------------------------------------------------
    print("\n-- leave-one-gene-out CV (paper protocol) --------------------------")
    for name, ctor in [("Rule Set 2 GBT (Azimuth params)", B.azimuth_gbt),
                       ("LightGBM (same features)", B.lightgbm_gbt)]:
        t0 = time.time()
        oof = logo_predictions(ctor, X, y, df)
        s = E.summarize(df, y, oof, f"LOGO  {name}")
        results[f"logo_{name}"] = s
        print(E.format_summary(s) + f"   [{time.time()-t0:.0f}s]")
        np.save(os.path.join(ART, f"oof_{'azimuth' if 'Azimuth' in name else 'lgbm'}_gbt.npy"), oof)

    pg = results["logo_Rule Set 2 GBT (Azimuth params)"]["_per_gene"]
    print("\n  per-gene Spearman, Rule Set 2 GBT under LOGO "
          "(this is the Fig. 4c quantity):")
    print("  " + pg.to_string(index=False).replace("\n", "\n  "))

    # ---- frozen gene-held-out split --------------------------------------
    print("\n-- frozen gene-held-out split --------------------------------------")
    tr = (df.split_gene == "train").values
    ca = (df.split_gene == "cal").values
    te = (df.split_gene == "test").values
    for name, ctor in [("Rule Set 2 GBT", B.azimuth_gbt), ("LightGBM", B.lightgbm_gbt)]:
        m = ctor()
        # Calibration rows are held out entirely here; they belong to conformal.
        pred = B.fit_predict(m, X[tr], y[tr], X)
        s = E.summarize(df[te], y[te], pred[te], f"GENE-HELD  {name}")
        results[f"geneheld_{name}"] = s
        print(E.format_summary(s))
        np.save(os.path.join(ART, f"pred_geneheld_{'azimuth' if 'Rule' in name else 'lgbm'}.npy"), pred)

    # ---- random split, for contrast only ---------------------------------
    print("\n-- random row-level split (leakage contrast, NOT the headline) -----")
    rtr = (df.split_random == "train").values
    rte = (df.split_random == "test").values
    pred = B.fit_predict(B.azimuth_gbt(), X[rtr], y[rtr], X[rte])
    s = E.summarize(df[rte], y[rte], pred, "RANDOM  Rule Set 2 GBT")
    results["random_azimuth"] = s
    print(E.format_summary(s))

    # ---- checkpoint -------------------------------------------------------
    logo_gbt = results["logo_Rule Set 2 GBT (Azimuth params)"]
    print("\n" + "=" * 72)
    print("CHECKPOINT vs LITERATURE")
    print("=" * 72)
    print(f"  Rule Set 2 GBT, leave-one-gene-out, mean per-gene Spearman:"
          f"  {logo_gbt['spearman_per_gene_mean']:.3f} "
          f"+/- {logo_gbt['spearman_per_gene_sd']:.3f} (sd across genes)")
    print("  Doench 2016 Fig. 4c reports this quantity graphically for the")
    print("  combined FC+RES dataset; boosted regression trees sit in the ~0.4-0.5")
    print("  band with sizeable across-gene spread. The paper gives no single")
    print("  number in text, so the comparison is to that reported range.")
    ok = 0.35 <= logo_gbt["spearman_per_gene_mean"] <= 0.65
    print(f"\n  IN EXPECTED RANGE: {ok}")
    if not ok:
        print("  >>> STOP. Debug the data pipeline before building the CNN. <<<")

    out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
           for k, v in results.items()}
    with open(os.path.join(ART, "baseline_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    pg.to_csv(os.path.join(ART, "baseline_per_gene.csv"), index=False)
    print(f"\nWrote {ART}/baseline_results.json")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
