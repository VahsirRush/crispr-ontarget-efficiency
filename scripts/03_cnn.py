"""Phase 3: train and evaluate the CNN + attention model.

Same three protocols as the baseline so the numbers are directly comparable.
Within every training fold an inner *gene-held-out* validation set is carved out
for early stopping, so model selection never sees a gene it will be scored on.

Each fold trains ``--seeds`` models and averages their predictions; a single run
of a small network on ~3k examples is noisy enough that seed variance would
otherwise dominate the comparison against the baseline.

Architecture / regularisation comes from artifacts/cnn_sweep.json, which was
chosen by cross-validation restricted to the training genes (scripts/03a).

Saves the fitted gene-held-out ensemble so Phases 5 and 6 reuse exactly the model
that produced the reported test numbers.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import data as D
from crispr import evaluate as E
from crispr import model as M

torch.set_num_threads(4)
ART = os.path.join(D.REPO_ROOT, "artifacts")

TH_NAMES = ["gc_count", "gc_above_10", "gc_below_10", "tm_global",
            "tm_5mer_pam_proximal", "tm_8mer_middle", "tm_5mer_start",
            "percent_peptide", "aa_cut_position", "percent_peptide_lt50"]
SEQ_ONLY = [i for i, n in enumerate(TH_NAMES)
            if n not in ("percent_peptide", "aa_cut_position", "percent_peptide_lt50")]


def load_config():
    """Architecture chosen by the training-gene-only sweep, with a documented fallback."""
    path = os.path.join(ART, "cnn_sweep.json")
    if os.path.exists(path):
        with open(path) as f:
            sel = json.load(f)["selected"]
        cfg = {k: sel[k] for k in ("c1", "c2", "attn_dim", "dropout", "conv_dropout",
                                   "weight_decay", "lr", "epochs", "patience", "thermo")}
        cfg["positional"] = sel.get("positional", False)
        return cfg, sel["config"]
    return dict(c1=64, c2=128, attn_dim=64, dropout=0.3, conv_dropout=0.35,
                weight_decay=1e-3, lr=1e-3, epochs=80, patience=15,
                thermo="all", positional=True), "fallback (sweep not run)"


def inner_val_split(df, train_idx, frac=0.2, seed=0):
    """Hold out whole genes from the training rows for early stopping.

    Genes are added smallest-first under a shuffled order and never allowed to
    overshoot the quota -- with MED12 at ~35% of all rows, an unguarded version
    hands most of the training data to validation.
    """
    sub = df.iloc[train_idx]
    sizes = sub["Target gene"].value_counts()
    target = frac * len(sub)
    rng = np.random.default_rng(seed)
    order = sorted(rng.permutation(sizes.index), key=lambda g: sizes[g])
    chosen, acc = [], 0
    for g in order:
        if acc + sizes[g] <= target * 1.2:
            chosen.append(g)
            acc += sizes[g]
    if not chosen or acc == len(sub):
        chosen = [order[0]]
    is_val = sub["Target gene"].isin(chosen).values
    return train_idx[~is_val], train_idx[is_val]


def fit_ensemble(df, seq, thermo, y, train_idx, cfg, seeds, verbose=False):
    """Train `seeds` models on train_idx. Returns (models, standardizer, infos)."""
    models, infos = [], []
    std = None
    for s in range(seeds):
        # A different inner validation gene subset per seed, so early stopping
        # is not pinned to one arbitrary choice of held-out genes.
        itr, iva = inner_val_split(df, train_idx, seed=s)
        std = M.Standardizer().fit(thermo[itr])
        th = std.transform(thermo)
        m, best, ep = M.train_model(
            seq[itr], th[itr], y[itr], seq[iva], th[iva], y[iva],
            n_channels=seq.shape[2], epochs=cfg["epochs"], lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], patience=cfg["patience"], seed=s,
            verbose=verbose, conv1_filters=cfg["c1"], conv2_filters=cfg["c2"],
            attn_dim=cfg["attn_dim"], dropout=cfg["dropout"],
            conv_dropout=cfg["conv_dropout"], positional=cfg["positional"],
        )
        models.append(m)
        infos.append({"seed": s, "val_spearman": float(best), "best_epoch": int(ep),
                      "n_train": int(len(itr)), "n_val": int(len(iva))})
    # Refit the standardizer on the full training block for inference.
    std = M.Standardizer().fit(thermo[train_idx])
    return models, std, infos


def ensemble_predict(models, std, seq, thermo, idx):
    th = std.transform(thermo)
    return np.mean([M.predict(m, seq[idx], th[idx]) for m in models], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-logo", action="store_true")
    args = ap.parse_args()

    cfg, cfg_name = load_config()
    print("=" * 72)
    print("PHASE 3  CNN + ATTENTION")
    print("=" * 72)
    print(f"config: {cfg_name}")
    print(f"        {cfg}")

    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    seq = np.load(os.path.join(ART, "X_onehot.npy"))
    thermo_all = np.load(os.path.join(ART, "X_thermo.npy"))
    thermo = thermo_all if cfg["thermo"] == "all" else thermo_all[:, SEQ_ONLY]
    y = df[D.TARGET].values.astype(np.float32)
    print(f"seq {seq.shape}  thermo {thermo.shape}  {args.seeds} seeds/fold\n")

    results = {}
    all_idx = np.arange(len(df))

    # ---- frozen gene-held-out split (also feeds Phases 5 and 6) -----------
    print("-- frozen gene-held-out split --------------------------------------")
    tr = np.where(df.split_gene == "train")[0]
    te = np.where(df.split_gene == "test")[0]
    t0 = time.time()
    models, std, infos = fit_ensemble(df, seq, thermo, y, tr, cfg, args.seeds)
    for i in infos:
        print(f"    seed {i['seed']}: val spearman {i['val_spearman']:.4f} "
              f"@epoch {i['best_epoch']} (train {i['n_train']}, val {i['n_val']})")
    pred_all = ensemble_predict(models, std, seq, thermo, all_idx)
    s = E.summarize(df.iloc[te], y[te], pred_all[te], "GENE-HELD  CNN+attention")
    results["geneheld_cnn"] = s
    print(E.format_summary(s) + f"   [{time.time()-t0:.0f}s]")
    np.save(os.path.join(ART, "pred_geneheld_cnn.npy"), pred_all)
    torch.save({
        "state_dicts": [m.state_dict() for m in models],
        "cfg": cfg, "thermo_mu": std.mu, "thermo_sd": std.sd,
        "n_channels": seq.shape[2], "n_thermo": thermo.shape[1],
    }, os.path.join(ART, "cnn_geneheld.pt"))

    # ---- random split (exchangeable calibration for Phase 6) --------------
    print("\n-- random row-level split (leakage contrast) -----------------------")
    rtr = np.where(df.split_random == "train")[0]
    rte = np.where(df.split_random == "test")[0]
    rmodels, rstd, _ = fit_ensemble(df, seq, thermo, y, rtr, cfg, args.seeds)
    rpred_all = ensemble_predict(rmodels, rstd, seq, thermo, all_idx)
    s = E.summarize(df.iloc[rte], y[rte], rpred_all[rte], "RANDOM  CNN+attention")
    results["random_cnn"] = s
    print(E.format_summary(s))
    np.save(os.path.join(ART, "pred_random_cnn.npy"), rpred_all)

    # ---- LOGO CV -----------------------------------------------------------
    if not args.skip_logo:
        print("\n-- leave-one-gene-out CV (paper protocol) --------------------------")
        oof = np.full(len(y), np.nan)
        t0 = time.time()
        for gene, tr_i, te_i in D.leave_one_gene_out_folds(df):
            ms, sd, _ = fit_ensemble(df, seq, thermo, y, tr_i, cfg, args.seeds)
            oof[te_i] = ensemble_predict(ms, sd, seq, thermo, te_i)
            print(f"  {gene:<10} n={len(te_i):<5} spearman={E.spearman(y[te_i], oof[te_i]):.4f}"
                  f"   [{time.time()-t0:.0f}s]")
        s = E.summarize(df, y, oof, "LOGO  CNN+attention")
        results["logo_cnn"] = s
        print("\n" + E.format_summary(s))
        np.save(os.path.join(ART, "oof_cnn.npy"), oof)
        s["_per_gene"].to_csv(os.path.join(ART, "cnn_per_gene.csv"), index=False)

    # ---- head-to-head ------------------------------------------------------
    with open(os.path.join(ART, "baseline_results.json")) as f:
        base = json.load(f)
    print("\n" + "=" * 72)
    print("CNN vs BASELINE  (mean per-gene Spearman)")
    print("=" * 72)
    print(f"  {'protocol':<22}{'RS2 GBT':>10}{'LightGBM':>12}{'CNN+attn':>12}{'CNN-GBT':>10}")
    for name, a, b, c in [
        ("leave-one-gene-out", base.get("logo_Rule Set 2 GBT (Azimuth params)"),
         base.get("logo_LightGBM (same features)"), results.get("logo_cnn")),
        ("gene-held-out test", base.get("geneheld_Rule Set 2 GBT"),
         base.get("geneheld_LightGBM"), results.get("geneheld_cnn")),
        ("random split", base.get("random_azimuth"), None, results.get("random_cnn")),
    ]:
        g = lambda d: d["spearman_per_gene_mean"] if d else float("nan")
        print(f"  {name:<22}{g(a):>10.4f}{g(b):>12.4f}{g(c):>12.4f}{g(c)-g(a):>+10.4f}")

    out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
           for k, v in results.items()}
    out["_config"] = {"name": cfg_name, **cfg}
    with open(os.path.join(ART, "cnn_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {ART}/cnn_results.json")


if __name__ == "__main__":
    main()
