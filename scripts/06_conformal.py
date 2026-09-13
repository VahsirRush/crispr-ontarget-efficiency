"""Phase 6: split conformal prediction intervals and their calibration.

Run under two split types on purpose:

  RANDOM      calibration and test rows are exchangeable, so conformal's
              finite-sample guarantee applies. Empirical coverage should track
              nominal almost exactly. This validates the implementation.
  GENE-HELD   calibration genes and test genes are disjoint, so exchangeability
              fails. Whatever gap opens up here is the honest cost of applying
              conformal to genuinely new genes, which is the deployment setting.

Produces figures/calibration.png with both curves against the diagonal.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import conformal as C
from crispr import data as D

ART = os.path.join(D.REPO_ROOT, "artifacts")
FIGS = os.path.join(D.REPO_ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)


def report(rows, title):
    print(f"\n  {title}")
    print(f"    {'nominal':>9}{'empirical':>11}{'gap':>9}{'half-width':>12}")
    for r in rows:
        if round(r["nominal_coverage"], 2) in (0.5, 0.8, 0.9, 0.95):
            print(f"    {r['nominal_coverage']:>9.2f}{r['empirical_coverage']:>11.3f}"
                  f"{r['empirical_coverage']-r['nominal_coverage']:>+9.3f}"
                  f"{r['mean_width']/2:>12.3f}")
    gaps = [abs(r["empirical_coverage"] - r["nominal_coverage"]) for r in rows]
    print(f"    mean |gap| across all levels: {np.mean(gaps):.4f}   "
          f"max |gap|: {np.max(gaps):.4f}")
    return float(np.mean(gaps)), float(np.max(gaps))


def main():
    print("=" * 72)
    print("PHASE 6  SPLIT CONFORMAL PREDICTION")
    print("=" * 72)

    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    y = df[D.TARGET].values

    settings = {}
    for split_col, pred_file, label in [
        ("split_random", "pred_random_cnn.npy", "RANDOM split (exchangeable)"),
        ("split_gene", "pred_geneheld_cnn.npy", "GENE-HELD split (shifted)"),
    ]:
        path = os.path.join(ART, pred_file)
        if not os.path.exists(path):
            print(f"  missing {pred_file}; run scripts/03_cnn.py first")
            continue
        pred = np.load(path)
        cal = (df[split_col] == "cal").values
        test = (df[split_col] == "test").values
        settings[label] = {
            "rows": C.coverage_curve(y[cal], pred[cal], y[test], pred[test]),
            "n_cal": int(cal.sum()), "n_test": int(test.sum()),
            "cal_mask": cal, "test_mask": test, "pred": pred,
        }

    print("\n-- calibration ------------------------------------------------------")
    summary = {}
    for label, s in settings.items():
        print(f"\n{label}  (n_cal={s['n_cal']}, n_test={s['n_test']})")
        mean_gap, max_gap = report(s["rows"], "marginal split conformal")
        r90 = next(r for r in s["rows"] if round(r["nominal_coverage"], 2) == 0.90)
        summary[label] = {
            "n_cal": s["n_cal"], "n_test": s["n_test"],
            "mean_abs_gap": mean_gap, "max_abs_gap": max_gap,
            "coverage_at_90": r90["empirical_coverage"],
            "half_width_at_90": r90["mean_width"] / 2,
            "curve": s["rows"],
        }
        print(f"    90% nominal -> {r90['empirical_coverage']:.1%} empirical "
              f"coverage, mean interval +/-{r90['mean_width']/2:.3f} "
              f"on a (0,1] target")

    # Group-conditional variant: only meaningful when genes are shared, i.e. the
    # random split. Under gene-held-out there is no shared gene to condition on.
    rnd = settings.get("RANDOM split (exchangeable)")
    if rnd is not None:
        genes = df["Target gene"].values
        mrows = C.mondrian_coverage_curve(
            y[rnd["cal_mask"]], rnd["pred"][rnd["cal_mask"]], genes[rnd["cal_mask"]],
            y[rnd["test_mask"]], rnd["pred"][rnd["test_mask"]], genes[rnd["test_mask"]],
        )
        if mrows:
            print("\nRANDOM split, per-gene (Mondrian) conformal")
            report(mrows, "group-conditional")
            summary["RANDOM split, Mondrian"] = {"curve": mrows}

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")
    styles = {"RANDOM split (exchangeable)": ("#1f77b4", "o"),
              "GENE-HELD split (shifted)": ("#d62728", "s")}
    for label, s in settings.items():
        c, mk = styles[label]
        nom = [r["nominal_coverage"] for r in s["rows"]]
        emp = [r["empirical_coverage"] for r in s["rows"]]
        ax.plot(nom, emp, marker=mk, color=c, ms=4.5, lw=1.6,
                label=f"{label}\n(n_test={s['n_test']})")
    ax.axvline(0.9, color="0.7", lw=0.9, ls=":")
    ax.set_xlabel("nominal coverage $1-\\alpha$")
    ax.set_ylabel("empirical coverage on held-out data")
    ax.set_title("Split conformal calibration, CNN + attention")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    for label, s in settings.items():
        c, mk = styles[label]
        nom = [r["nominal_coverage"] for r in s["rows"]]
        w = [r["mean_width"] for r in s["rows"]]
        ax.plot(nom, w, marker=mk, color=c, ms=4.5, lw=1.6, label=label)
    ax.set_xlabel("nominal coverage $1-\\alpha$")
    ax.set_ylabel("mean interval width")
    ax.set_title("Interval width vs confidence level\n"
                 "(target is a rank in (0, 1], so width 1.0 is uninformative)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(FIGS, "calibration.png")
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")

    with open(os.path.join(ART, "conformal_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {ART}/conformal_results.json")


if __name__ == "__main__":
    main()
