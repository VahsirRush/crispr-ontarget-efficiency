"""Phase 5: what does the attention layer look at?

Extracts the attention-pooling weights from the gene-held-out ensemble saved by
Phase 3 and plots them against sequence position, with the protospacer, PAM and
seed region marked.

The biological prior being checked: positions ~17-20 of the 20nt protospacer (the
PAM-proximal "seed") dominate Cas9 target recognition, so a model that has learned
real biology should weight them above background. This is a sanity check, not a
pass/fail criterion -- the conv stack has a receptive field wide enough that
position-specific signal can also live in the filters rather than the pooling
weights.

For an independent view of the same question, the GBT's Gini importance is also
aggregated by position; it needs no attention mechanism to be interpretable and so
acts as a control for whatever the attention weights show.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import baseline as B
from crispr import data as D
from crispr import features as Feat
from crispr import model as M

ART = os.path.join(D.REPO_ROOT, "artifacts")
FIGS = os.path.join(D.REPO_ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

TH_NAMES = ["gc_count", "gc_above_10", "gc_below_10", "tm_global",
            "tm_5mer_pam_proximal", "tm_8mer_middle", "tm_5mer_start",
            "percent_peptide", "aa_cut_position", "percent_peptide_lt50"]
SEQ_ONLY = [i for i, n in enumerate(TH_NAMES)
            if n not in ("percent_peptide", "aa_cut_position", "percent_peptide_lt50")]


def shade_regions(ax):
    """Mark the 5' context, protospacer, seed and PAM on a position axis."""
    ax.axvspan(-0.5, 3.5, color="0.92", zorder=0)
    ax.axvspan(23.5, 26.5, color="#ffe0b2", alpha=0.7, zorder=0)
    ax.axvspan(26.5, 29.5, color="0.92", zorder=0)
    ax.axvspan(D.SEED_30MER_INDICES[0] - 0.5, D.SEED_30MER_INDICES[-1] + 0.5,
               color="#c8e6c9", alpha=0.8, zorder=0)


def main():
    print("=" * 72)
    print("PHASE 5  INTERPRETABILITY -- ATTENTION BY SEQUENCE POSITION")
    print("=" * 72)

    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    seq = np.load(os.path.join(ART, "X_onehot.npy"))
    thermo_all = np.load(os.path.join(ART, "X_thermo.npy"))
    y = df[D.TARGET].values.astype(np.float32)

    ckpt = torch.load(os.path.join(ART, "cnn_geneheld.pt"), weights_only=False)
    cfg = ckpt["cfg"]
    thermo = thermo_all if cfg["thermo"] == "all" else thermo_all[:, SEQ_ONLY]
    th = ((thermo - ckpt["thermo_mu"]) / ckpt["thermo_sd"]).astype(np.float32)

    models = []
    for sd in ckpt["state_dicts"]:
        m = M.CrisprCNN(n_channels=ckpt["n_channels"], n_thermo=ckpt["n_thermo"],
                        conv1_filters=cfg["c1"], conv2_filters=cfg["c2"],
                        attn_dim=cfg["attn_dim"], dropout=cfg["dropout"],
                        conv_dropout=cfg["conv_dropout"],
                        positional=cfg.get("positional", False))
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    print(f"loaded {len(models)}-model ensemble from the gene-held-out fit")

    te = np.where(df.split_gene == "test")[0]
    pools, attns = [], []
    for m in models:
        _, pw, aw = M.predict(m, seq[te], th[te], return_attn=True)
        pools.append(pw)
        attns.append(aw)
    pool_w = np.mean(pools, axis=0)          # (n_test, 30)
    attn_w = np.mean(attns, axis=0)          # (n_test, 30, 30)
    print(f"attention extracted on {len(te)} held-out guides")

    # ---- seed-region check -------------------------------------------------
    mean_w = pool_w.mean(axis=0)
    sem_w = pool_w.std(axis=0) / np.sqrt(len(pool_w))
    seed_idx = list(D.SEED_30MER_INDICES)
    proto_idx = list(range(4, 24))
    nonseed_proto = [i for i in proto_idx if i not in seed_idx]
    uniform = 1.0 / 30

    seed_mean = mean_w[seed_idx].mean()
    nonseed_mean = mean_w[nonseed_proto].mean()
    # Paired across guides: each guide contributes its own seed vs non-seed mean.
    per_guide_seed = pool_w[:, seed_idx].mean(axis=1)
    per_guide_rest = pool_w[:, nonseed_proto].mean(axis=1)
    t = stats.wilcoxon(per_guide_seed, per_guide_rest)

    print("\n-- attention-pooling weight by region ------------------------------")
    print(f"  uniform baseline (1/30)            : {uniform:.4f}")
    print(f"  seed, protospacer 17-20            : {seed_mean:.4f}  "
          f"({seed_mean/uniform:.2f}x uniform)")
    print(f"  rest of protospacer, positions 1-16: {nonseed_mean:.4f}  "
          f"({nonseed_mean/uniform:.2f}x uniform)")
    print(f"  PAM (NGG)                          : {mean_w[24:27].mean():.4f}  "
          f"({mean_w[24:27].mean()/uniform:.2f}x uniform)")
    print(f"  5' context (-4..-1)                : {mean_w[0:4].mean():.4f}")
    print(f"  3' context (+1..+3)                : {mean_w[27:30].mean():.4f}")
    print(f"\n  paired Wilcoxon, seed vs rest-of-protospacer: "
          f"statistic={t.statistic:.0f}, p={t.pvalue:.3e}")
    verdict = ("CONSISTENT with the seed-region prior" if seed_mean > nonseed_mean
               else "NOT consistent with the seed-region prior")
    print(f"  => {verdict}")

    top = np.argsort(mean_w)[::-1][:6]
    print(f"\n  highest-weighted positions: "
          f"{[Feat.POSITION_LABELS[i] for i in top]}")

    # ---- GBT positional importance, as an independent control -------------
    print("\n-- control: GBT Gini importance aggregated by position -------------")
    X = np.load(os.path.join(ART, "X_rs2.npy"))
    with open(os.path.join(ART, "X_rs2_names.json")) as f:
        names = json.load(f)
    tr = (df.split_gene == "train").values
    gbt = B.azimuth_gbt()
    gbt.fit(X[tr], y[tr])
    imp = gbt.feature_importances_
    pos_imp = np.zeros(30)
    label_to_pos = {}
    for i, lab in enumerate(Feat.POSITION_LABELS):
        label_to_pos.setdefault(lab, []).append(i)
    for name, v in zip(names, imp):
        if name.startswith("pd1_"):          # order-1 position-dependent only
            lab = name.rsplit("_", 1)[1]
            for p in label_to_pos[lab]:
                pos_imp[p] += v / len(label_to_pos[lab])
    pos_imp = pos_imp / pos_imp.sum()
    print(f"  seed region share : {pos_imp[seed_idx].sum():.3f} "
          f"(4/30 positions = {4/30:.3f} if flat)")
    print(f"  PAM-adjacent N,+1 : {pos_imp[24]:.3f}, {pos_imp[27]:.3f}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 11),
                             gridspec_kw={"height_ratios": [1.1, 1.1, 1.4]})

    ax = axes[0]
    shade_regions(ax)
    ax.bar(range(30), mean_w, yerr=sem_w, color="#1f77b4", zorder=3)
    ax.axhline(uniform, ls="--", c="crimson", lw=1.2, zorder=4,
               label=f"uniform ({uniform:.3f})")
    ax.set_xticks(range(30))
    ax.set_xticklabels(Feat.POSITION_LABELS, fontsize=7.5)
    ax.set_ylabel("mean attention-pooling weight")
    ax.set_title("CNN attention-pooling weight by position "
                 f"(n={len(te)} held-out guides)\n"
                 "green = seed (protospacer 17-20), orange = PAM, grey = flanking context",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 29.5)

    ax = axes[1]
    shade_regions(ax)
    ax.bar(range(30), pos_imp, color="#2ca02c", zorder=3)
    ax.axhline(1 / 30, ls="--", c="crimson", lw=1.2, zorder=4, label="flat (1/30)")
    ax.set_xticks(range(30))
    ax.set_xticklabels(Feat.POSITION_LABELS, fontsize=7.5)
    ax.set_ylabel("normalised Gini importance")
    ax.set_title("Control: Rule Set 2 GBT order-1 nucleotide importance, "
                 "same positions", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 29.5)

    ax = axes[2]
    im = ax.imshow(attn_w.mean(axis=0), cmap="viridis", aspect="auto")
    ax.set_xticks(range(30)); ax.set_xticklabels(Feat.POSITION_LABELS, fontsize=6.5)
    ax.set_yticks(range(30)); ax.set_yticklabels(Feat.POSITION_LABELS, fontsize=6.5)
    ax.set_xlabel("key position (attended to)")
    ax.set_ylabel("query position")
    ax.set_title("Mean single-head self-attention matrix", fontsize=10)
    for k in (D.SEED_30MER_INDICES[0] - 0.5, D.SEED_30MER_INDICES[-1] + 0.5):
        ax.axvline(k, c="w", lw=1.0, ls="--")
    fig.colorbar(im, ax=ax, fraction=0.025)

    fig.tight_layout()
    out = os.path.join(FIGS, "attention_by_position.png")
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")

    pd.DataFrame({
        "index_30mer": range(30),
        "position_label": Feat.POSITION_LABELS,
        "region": (["5' context"] * 4 + ["protospacer"] * 16 + ["seed"] * 4
                   + ["PAM"] * 3 + ["3' context"] * 3),
        "attention_weight": mean_w,
        "attention_sem": sem_w,
        "gbt_positional_importance": pos_imp,
    }).to_csv(os.path.join(ART, "attention_by_position.csv"), index=False)

    with open(os.path.join(ART, "interpretability.json"), "w") as f:
        json.dump({
            "n_guides": int(len(te)),
            "uniform_weight": uniform,
            "seed_mean_weight": float(seed_mean),
            "nonseed_protospacer_mean_weight": float(nonseed_mean),
            "pam_mean_weight": float(mean_w[24:27].mean()),
            "seed_vs_rest_ratio": float(seed_mean / nonseed_mean),
            "wilcoxon_p": float(t.pvalue),
            "consistent_with_seed_prior": bool(seed_mean > nonseed_mean),
            "gbt_seed_importance_share": float(pos_imp[seed_idx].sum()),
            "top_positions": [Feat.POSITION_LABELS[i] for i in top],
        }, f, indent=2)
    print(f"Wrote {ART}/interpretability.json")


if __name__ == "__main__":
    main()
