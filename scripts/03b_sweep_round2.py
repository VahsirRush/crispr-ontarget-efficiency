"""Phase 3b: GBT reference on the identical inner protocol, then a focused round 2.

Round 1 picked conv dropout 0.2 at 0.377 inner CV, but that number is not
comparable to the GBT's 0.515 leave-one-gene-out: the inner protocol trains on 5
genes instead of 16 and holds out much larger gene blocks. So first measure the
GBT under exactly the same 6-gene protocol to establish the real gap, then push
on the regularisation axis that round 1 showed was working.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from crispr import baseline as B
from crispr import data as D
from crispr import evaluate as E

importlib_mod = __import__("importlib")
sweep = importlib_mod.import_module("03a_sweep_cnn")

torch.set_num_threads(4)
ART = os.path.join(D.REPO_ROOT, "artifacts")


def gbt_inner_cv(df, X, y, genes, ctor):
    keep = df["Target gene"].isin(genes).values
    idx_all = np.where(keep)[0]
    scores = []
    for held in genes:
        te = np.where(keep & (df["Target gene"] == held).values)[0]
        if len(te) < 10:
            continue
        tr = np.setdiff1d(idx_all, te)
        p = B.fit_predict(ctor(), X[tr], y[tr], X[te])
        scores.append(E.spearman(y[te], p))
    return float(np.mean(scores)), scores


# Round 1's winner was conv dropout 0.2 at 0.377, against a GBT reference of ~0.505
# on this same protocol. The remaining structural gap is that a plain conv stack has
# no absolute-position information, which is where most of Rule Set 2's signal lives.
ROUND2 = [
    ("conv drop 0.2 (round-1 winner)",   dict(conv_dropout=0.2)),
    ("+ positional embedding",           dict(conv_dropout=0.2, positional=True)),
    ("+ pos emb, conv drop 0.35",        dict(conv_dropout=0.35, positional=True)),
    ("+ pos emb, conv drop 0.35, wd 1e-3",
     dict(conv_dropout=0.35, positional=True, weight_decay=1e-3)),
    ("+ pos emb, narrow 32/64",          dict(conv_dropout=0.2, positional=True,
                                              c1=32, c2=64, attn_dim=32)),
    ("+ pos emb, longer 150 epochs",     dict(conv_dropout=0.2, positional=True,
                                              epochs=150, patience=25)),
]


def main():
    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    seq = np.load(os.path.join(ART, "X_onehot.npy"))
    thermo = np.load(os.path.join(ART, "X_thermo.npy"))
    Xrs2 = np.load(os.path.join(ART, "X_rs2.npy"))
    y = df[D.TARGET].values.astype(np.float32)
    genes = sorted(df[df.split_gene == "train"]["Target gene"].unique())

    print("=" * 72)
    print("PHASE 3b  GBT REFERENCE ON THE INNER PROTOCOL, THEN ROUND 2")
    print("=" * 72)
    print(f"selection genes: {genes}\n")

    print("-- GBT on the identical 6-gene inner CV ----------------------------")
    # Measured in a previous run; recomputed only when --refit-gbt is passed, since
    # it is deterministic and takes a couple of minutes.
    gbt_ref = 0.5051
    if "--refit-gbt" in sys.argv:
        for name, ctor in [("Rule Set 2 GBT", B.azimuth_gbt), ("LightGBM", B.lightgbm_gbt)]:
            m, per = gbt_inner_cv(df, Xrs2, y, genes, ctor)
            print(f"  {name:<34} {m:>7.4f}   per-gene {[f'{x:.3f}' for x in per]}")
            if name == "Rule Set 2 GBT":
                gbt_ref = m
    else:
        print(f"  Rule Set 2 GBT                      {gbt_ref:.4f}   (cached; "
              f"pass --refit-gbt to recompute)")
        print(f"  LightGBM                            0.5328   (cached)")
    print(f"\n  => on this protocol the GBT scores {gbt_ref:.4f}, not its 0.515 LOGO"
          f" number.\n     That is the bar the CNN has to clear here.\n")

    print("-- CNN round 2 ------------------------------------------------------")
    rows = []
    for name, override in ROUND2:
        cfg = {**sweep.BASE, **override}
        t0 = time.time()
        mean, per = sweep.run_config(df, seq, thermo, y, genes, cfg)
        rows.append({"config": name, "mean_spearman": mean, **cfg})
        print(f"  {name:<34} {mean:>7.4f}   per-gene {[f'{x:.3f}' for x in per]}"
              f"  [{time.time()-t0:.0f}s]")

    best = max(rows, key=lambda r: r["mean_spearman"])
    print(f"\nSELECTED: {best['config']}  (inner CV {best['mean_spearman']:.4f}; "
          f"GBT reference {gbt_ref:.4f})")

    with open(os.path.join(ART, "cnn_sweep.json")) as f:
        prev = json.load(f)
    allr = prev["results"] + rows
    best_all = max(allr, key=lambda r: r["mean_spearman"])
    with open(os.path.join(ART, "cnn_sweep.json"), "w") as f:
        json.dump({"results": allr, "selected": best_all,
                   "gbt_reference_same_protocol": gbt_ref}, f, indent=2)
    print(f"Overall selected: {best_all['config']} ({best_all['mean_spearman']:.4f})")


if __name__ == "__main__":
    main()
