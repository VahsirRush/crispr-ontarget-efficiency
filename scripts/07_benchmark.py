"""Phase 7: assemble the benchmark table and test whether the differences are real.

With only 18 (gene, drug) evaluation groups and a standard deviation of ~0.08
across them, differences of a few hundredths in mean Spearman are not
distinguishable from noise. Every model pair is therefore compared with a test
paired across gene groups plus a bootstrap interval on the mean difference, so
the table can say "tie" where a tie is what the data supports.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import data as D
from crispr import evaluate as E

ART = os.path.join(D.REPO_ROOT, "artifacts")
FIGS = os.path.join(D.REPO_ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)


def paired_compare(df, y, pred_a, pred_b, name_a, name_b, n_boot=10000, seed=0):
    """Per-gene-group Spearman for each model, then a paired comparison."""
    a = E.per_gene_spearman(df, y, pred_a).set_index(["gene", "drug"])["spearman"]
    b = E.per_gene_spearman(df, y, pred_b).set_index(["gene", "drug"])["spearman"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].values, b.loc[common].values
    d = a - b

    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    w = stats.wilcoxon(a, b) if len(d) > 5 else None
    return {
        "model_a": name_a, "model_b": name_b, "n_groups": len(d),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "wilcoxon_p": float(w.pvalue) if w else float("nan"),
        "significant": bool(lo > 0 or hi < 0),
    }


def main():
    print("=" * 72)
    print("PHASE 7  BENCHMARK TABLE AND SIGNIFICANCE")
    print("=" * 72)

    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    y = df[D.TARGET].values

    with open(os.path.join(ART, "baseline_results.json")) as f:
        base = json.load(f)
    with open(os.path.join(ART, "cnn_results.json")) as f:
        cnn = json.load(f)

    # ---- benchmark table ---------------------------------------------------
    rows = []
    def add(protocol, model, s):
        if s:
            rows.append({
                "protocol": protocol, "model": model, "n": s["n"],
                "spearman_per_gene_mean": s["spearman_per_gene_mean"],
                "spearman_per_gene_sd": s["spearman_per_gene_sd"],
                "spearman_pooled": s["spearman_pooled"], "rmse": s["rmse"],
            })

    add("leave-one-gene-out", "Rule Set 2 GBT (Azimuth params)",
        base.get("logo_Rule Set 2 GBT (Azimuth params)"))
    add("leave-one-gene-out", "LightGBM (Rule Set 2 features)",
        base.get("logo_LightGBM (same features)"))
    add("leave-one-gene-out", "CNN + attention", cnn.get("logo_cnn"))
    add("gene-held-out test", "Rule Set 2 GBT (Azimuth params)",
        base.get("geneheld_Rule Set 2 GBT"))
    add("gene-held-out test", "LightGBM (Rule Set 2 features)",
        base.get("geneheld_LightGBM"))
    add("gene-held-out test", "CNN + attention", cnn.get("geneheld_cnn"))
    add("random split (leaky)", "Rule Set 2 GBT (Azimuth params)",
        base.get("random_azimuth"))
    add("random split (leaky)", "CNN + attention", cnn.get("random_cnn"))
    add("in-sample reference", "Azimuth released model (shipped predictions)",
        base.get("azimuth_shipped"))

    tab = pd.DataFrame(rows)
    print("\n-- benchmark table --------------------------------------------------")
    print(f"  {'protocol':<22}{'model':<46}{'Spearman':>10}{'sd':>8}{'pooled':>9}")
    for _, r in tab.iterrows():
        print(f"  {r['protocol']:<22}{r['model']:<46}"
              f"{r['spearman_per_gene_mean']:>10.4f}{r['spearman_per_gene_sd']:>8.3f}"
              f"{r['spearman_pooled']:>9.4f}")
    tab.to_csv(os.path.join(ART, "benchmark_table.csv"), index=False)

    # ---- significance on the LOGO out-of-fold predictions -----------------
    print("\n-- paired comparisons, leave-one-gene-out OOF -----------------------")
    preds = {
        "Rule Set 2 GBT": np.load(os.path.join(ART, "oof_azimuth_gbt.npy")),
        "LightGBM": np.load(os.path.join(ART, "oof_lgbm_gbt.npy")),
        "CNN + attention": np.load(os.path.join(ART, "oof_cnn.npy")),
    }
    comps = []
    for a, b in [("CNN + attention", "Rule Set 2 GBT"),
                 ("CNN + attention", "LightGBM"),
                 ("LightGBM", "Rule Set 2 GBT")]:
        c = paired_compare(df, y, preds[a], preds[b], a, b)
        comps.append(c)
        verdict = "DIFFERENT" if c["significant"] else "tie (CI spans 0)"
        print(f"  {a} - {b}:")
        print(f"      mean diff {c['mean_diff']:+.4f}  "
              f"95% CI [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]  "
              f"Wilcoxon p={c['wilcoxon_p']:.3f}  -> {verdict}")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    protocols = ["leave-one-gene-out", "gene-held-out test", "random split (leaky)"]
    models = ["Rule Set 2 GBT (Azimuth params)", "LightGBM (Rule Set 2 features)",
              "CNN + attention"]
    colors = {"Rule Set 2 GBT (Azimuth params)": "#7f7f7f",
              "LightGBM (Rule Set 2 features)": "#2ca02c",
              "CNN + attention": "#1f77b4"}
    w = 0.26
    for i, m in enumerate(models):
        xs, vals, errs = [], [], []
        for j, p in enumerate(protocols):
            sub = tab[(tab.protocol == p) & (tab.model == m)]
            if len(sub):
                xs.append(j + (i - 1) * w)
                vals.append(sub.iloc[0]["spearman_per_gene_mean"])
                errs.append(sub.iloc[0]["spearman_per_gene_sd"])
        ax.bar(xs, vals, w, yerr=errs, capsize=3, label=m, color=colors[m])
    ax.axhspan(0.4, 0.6, color="#ffd54f", alpha=0.25, zorder=0,
               label="Doench 2016 Fig. 4c reported band")
    ax.set_xticks(range(len(protocols)))
    ax.set_xticklabels(["leave-one-\ngene-out", "gene-held-out\ntest",
                        "random split\n(leaky)"], fontsize=9)
    ax.set_ylabel("mean per-gene Spearman")
    ax.set_title("Benchmark. Error bars = sd across gene groups,\n"
                 "which is larger than every between-model gap.", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    pg_gbt = E.per_gene_spearman(df, y, preds["Rule Set 2 GBT"]).set_index(["gene", "drug"])
    pg_cnn = E.per_gene_spearman(df, y, preds["CNN + attention"]).set_index(["gene", "drug"])
    common = pg_gbt.index.intersection(pg_cnn.index)
    gx = pg_gbt.loc[common, "spearman"].values
    cy = pg_cnn.loc[common, "spearman"].values
    sizes = pg_gbt.loc[common, "n"].values
    ax.scatter(gx, cy, s=np.sqrt(sizes) * 4, alpha=0.75, color="#1f77b4",
               edgecolor="k", linewidth=0.5)
    lim = [min(gx.min(), cy.min()) - 0.05, max(gx.max(), cy.max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1)
    for (g, d), x0, y0 in zip(common, gx, cy):
        ax.annotate(g, (x0, y0), fontsize=6.5, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Rule Set 2 GBT, per-gene Spearman")
    ax.set_ylabel("CNN + attention, per-gene Spearman")
    ax.set_title("Per-gene agreement under leave-one-gene-out\n"
                 "(marker area ~ number of guides)", fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(FIGS, "benchmark.png")
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")

    with open(os.path.join(ART, "benchmark.json"), "w") as f:
        json.dump({"table": rows, "comparisons": comps}, f, indent=2)
    print(f"Wrote {ART}/benchmark.json")


if __name__ == "__main__":
    main()
