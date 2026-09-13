"""Phase 3a: architecture / regularisation selection for the CNN.

The spec's literal 64/128 stack memorises the training genes (train Spearman
~0.99, held-out ~0.21), so some selection is unavoidable. To keep the reported
test number honest, selection happens entirely inside the TRAINING genes:
leave-one-gene-out over the 6 training genes only. The frozen test genes and the
calibration genes are never loaded here.
"""

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These models are tiny; extra threads cost more in synchronisation than they save.
torch.set_num_threads(4)

from crispr import data as D
from crispr import evaluate as E
from crispr import model as M

ART = os.path.join(D.REPO_ROOT, "artifacts")

# Thermodynamic block column order, from features.thermodynamic_features.
TH_NAMES = ["gc_count", "gc_above_10", "gc_below_10", "tm_global",
            "tm_5mer_pam_proximal", "tm_8mer_middle", "tm_5mer_start",
            "percent_peptide", "aa_cut_position", "percent_peptide_lt50"]


def run_config(df, seq, thermo, y, genes, cfg, seeds=2):
    """Leave-one-gene-out over `genes` only. Returns mean per-gene Spearman."""
    keep = df["Target gene"].isin(genes).values
    idx_all = np.where(keep)[0]
    scores, n_used = [], []

    for held in genes:
        te = np.where(keep & (df["Target gene"] == held).values)[0]
        tr = np.array([i for i in idx_all if i not in set(te)])
        if len(te) < 10:
            continue
        # inner split for early stopping, again by gene
        inner_genes = [g for g in genes if g != held]
        sizes = df.iloc[tr]["Target gene"].value_counts()
        val_genes, acc = [], 0
        for g in sorted(inner_genes, key=lambda g: sizes.get(g, 0)):
            if acc + sizes.get(g, 0) <= 0.2 * len(tr) * 1.2:
                val_genes.append(g); acc += sizes.get(g, 0)
        if not val_genes:
            val_genes = [min(inner_genes, key=lambda g: sizes.get(g, 0))]
        vmask = df.iloc[tr]["Target gene"].isin(val_genes).values
        itr, iva = tr[~vmask], tr[vmask]
        if len(itr) < 100 or len(iva) < 30:
            continue

        std = M.Standardizer().fit(thermo[itr])
        th = std.transform(thermo)
        preds = []
        for s in range(seeds):
            m, _, _ = M.train_model(
                seq[itr], th[itr], y[itr], seq[iva], th[iva], y[iva],
                n_channels=seq.shape[2], epochs=cfg["epochs"], lr=cfg["lr"],
                weight_decay=cfg["weight_decay"], patience=cfg["patience"], seed=s,
                conv1_filters=cfg["c1"], conv2_filters=cfg["c2"],
                attn_dim=cfg["attn_dim"], dropout=cfg["dropout"],
                conv_dropout=cfg["conv_dropout"],
                positional=cfg.get("positional", False),
            )
            preds.append(M.predict(m, seq[te], th[te]))
        scores.append(E.spearman(y[te], np.mean(preds, axis=0)))
        n_used.append(len(te))
    return float(np.mean(scores)), scores


BASE = dict(c1=64, c2=128, attn_dim=64, dropout=0.3, conv_dropout=0.0,
            weight_decay=1e-4, lr=1e-3, epochs=80, patience=15, thermo="all")

CONFIGS = [
    ("spec-literal 64/128",            {}),
    ("+ conv dropout 0.2",             dict(conv_dropout=0.2)),
    ("+ weight decay 1e-2",            dict(weight_decay=1e-2)),
    ("narrow 32/64",                   dict(c1=32, c2=64, attn_dim=32)),
    ("narrow + conv drop + wd",        dict(c1=32, c2=64, attn_dim=32,
                                            conv_dropout=0.2, weight_decay=1e-2)),
    ("drop gene-position features",    dict(thermo="seq_only")),
    ("narrow + regd + seq-only thermo", dict(c1=32, c2=64, attn_dim=32,
                                             conv_dropout=0.2, weight_decay=1e-2,
                                             thermo="seq_only")),
    ("very narrow 16/32 + regd",       dict(c1=16, c2=32, attn_dim=32,
                                            conv_dropout=0.3, weight_decay=1e-2,
                                            thermo="seq_only")),
]


def main():
    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    seq = np.load(os.path.join(ART, "X_onehot.npy"))
    thermo_all = np.load(os.path.join(ART, "X_thermo.npy"))
    y = df[D.TARGET].values.astype(np.float32)

    train_genes = sorted(df[df.split_gene == "train"]["Target gene"].unique())
    print("=" * 72)
    print("PHASE 3a  CNN SELECTION (leave-one-gene-out WITHIN training genes)")
    print("=" * 72)
    print(f"selection genes: {train_genes}")
    print(f"rows available : {(df.split_gene=='train').sum()}")
    print("test and calibration genes are not touched here.\n")

    # 'seq_only' drops percent_peptide / aa_cut_position: these are gene-level
    # covariates whose scale differs wildly between genes, so under a gene-held-out
    # split they are a training shortcut that cannot transfer.
    seq_only_cols = [i for i, n in enumerate(TH_NAMES)
                     if n not in ("percent_peptide", "aa_cut_position",
                                  "percent_peptide_lt50")]

    rows = []
    for name, override in CONFIGS:
        cfg = {**BASE, **override}
        th = thermo_all if cfg["thermo"] == "all" else thermo_all[:, seq_only_cols]
        t0 = time.time()
        mean, per = run_config(df, seq, th, y, train_genes, cfg)
        rows.append({"config": name, "mean_spearman": mean, **cfg})
        print(f"  {name:<34} {mean:>7.4f}   per-gene {[f'{x:.3f}' for x in per]}"
              f"  [{time.time()-t0:.0f}s]")

    best = max(rows, key=lambda r: r["mean_spearman"])
    print(f"\nSELECTED: {best['config']}  (inner CV mean Spearman {best['mean_spearman']:.4f})")
    with open(os.path.join(ART, "cnn_sweep.json"), "w") as f:
        json.dump({"results": rows, "selected": best}, f, indent=2)


if __name__ == "__main__":
    main()
