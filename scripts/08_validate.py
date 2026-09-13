"""Phase 8: adversarial validation. Try to break the results rather than confirm them.

Each check is written so that a PASS is informative -- i.e. the check would
actually fail if the corresponding bug were present.

  V1  row accounting against the numbers stated in the paper
  V2  sequence leakage across every split
  V3  feature reproduction, unit-tested against hand-computed values
  V4  integrity of the saved out-of-fold predictions
  V5  label-shuffle negative control (the strongest anti-leakage test)
  V6  independent agreement with the released Azimuth model
  V7  conformal machinery against a synthetic case with a known answer
  V8  headline numbers recomputed from artifacts, independently of the phase scripts
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import baseline as B
from crispr import conformal as C
from crispr import data as D
from crispr import evaluate as E
from crispr import features as F

ART = os.path.join(D.REPO_ROOT, "artifacts")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.split("\n"):
            print(f"         {line}")
    return ok


def main():
    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    y = df[D.TARGET].values
    X = np.load(os.path.join(ART, "X_rs2.npy"))

    # ================================================================== V1
    print("\nV1  ROW ACCOUNTING vs THE PAPER")
    print("-" * 70)
    fc, res = (df.dataset == "FC").sum(), (df.dataset == "RES").sum()
    # Paper: 1,841 sgRNAs in FC (9 genes); 2,549 sgRNAs in RES (8 genes).
    # RES rows exceed 2,549 because MED12 was screened under two drugs.
    res_unique = df[df.dataset == "RES"]["30mer"].nunique()
    med12 = ((df["Target gene"] == "MED12")).sum()
    check("FC rows within 5 of the paper's 1,841",
          abs(fc - 1841) <= 5,
          f"got {fc}; Azimuth drops a handful in the gene-position join\n"
          f"(load_data notes '11 missing guides we were told to ignore')")
    check("RES unique guides == the paper's 2,549",
          res_unique == 2549,
          f"got {res_unique} unique 30mers across {res} rows; "
          f"MED12 contributes {med12} rows under 2 drugs")
    check("gene count == 17", df["Target gene"].nunique() == 17)

    # ================================================================== V2
    print("\nV2  SEQUENCE LEAKAGE ACROSS SPLITS")
    print("-" * 70)
    def overlaps(col):
        parts = {s: set(df[df[col] == s]["30mer"]) for s in ("train", "cal", "test")}
        return {"train/test": len(parts["train"] & parts["test"]),
                "train/cal": len(parts["train"] & parts["cal"]),
                "cal/test": len(parts["cal"] & parts["test"])}

    ov_gene = overlaps("split_gene")
    check("split_gene (the reported protocol): no shared 30mer between splits",
          sum(ov_gene.values()) == 0, f"{ov_gene}")

    # 1,856 rows share a 30mer with another row (MED12 under two drugs, plus CD43).
    # A row-level random split is therefore expected to put copies of the same guide
    # on both sides. Asserting it DOES leak is the honest form of this check: it is
    # the very thing the random split exists to demonstrate.
    ov_rand = overlaps("split_random")
    check("split_random leaks duplicate sequences, as designed to demonstrate",
          sum(ov_rand.values()) > 0,
          f"{ov_rand}  -- this split is reported only as a leakage contrast")

    # The gene-held split must also not share genes.
    gsets = {s: set(df[df.split_gene == s]["Target gene"]) for s in ("train", "cal", "test")}
    check("split_gene: no shared gene between splits",
          not (gsets["train"] & gsets["test"]) and not (gsets["train"] & gsets["cal"])
          and not (gsets["cal"] & gsets["test"]))

    # ================================================================== V3
    print("\nV3  FEATURE REPRODUCTION, UNIT-TESTED")
    print("-" * 70)
    # A 30mer with a hand-checkable composition.
    #      idx 0123   4..................23  242526  272829
    seq = "AAAA" + "GGGGGCCCCCTTTTTAAAAA" + "AGG" + "CTT"
    probe = pd.DataFrame({"30mer": [seq], "Percent Peptide": [10.0],
                          "Amino Acid Cut position": [5.0]})
    check("probe sequence is a valid 30mer with GG at [25:27]",
          len(seq) == 30 and seq[25:27] == "GG", f"seq={seq}")

    gc, gc_names = F.gc_features([seq])
    # protospacer = seq[4:24] = 5xG + 5xC + 5xT + 5xA -> GC count 10
    check("GC count uses the 20mer protospacer only",
          gc[0, 0] == 10, f"got {gc[0,0]}, expected 10 (5 G + 5 C in seq[4:24])")
    check("GC>10 and GC<10 indicators are both 0 at exactly 10",
          gc[0, 1] == 0 and gc[0, 2] == 0, f"got {gc[0,1]}, {gc[0,2]}")

    nggx, nggx_names = F.nggx_features([seq])
    # N = seq[24] = 'A', X = seq[27] = 'C'  -> the "AC" one-hot slot
    expect = nggx_names.index("NGGX_AC")
    check("NGGX one-hot fires on seq[24]+seq[27]",
          nggx[0].sum() == 1 and nggx[0, expect] == 1,
          f"N={seq[24]} X={seq[27]} -> expected NGGX_AC")

    pd1, pd1n, pi1, pi1n = F.nucleotide_features([seq], 1)
    check("order-1 position-dependent: 120 features, one per position",
          pd1.shape[1] == 120 and pd1.sum() == 30)
    counts = {b: seq.count(b) for b in "ATCG"}
    got = {n.split("_")[1]: v for n, v in zip(pi1n, pi1[0])}
    check("order-1 position-independent counts match string counts",
          all(got[b] == counts[b] for b in "ATCG"), f"{got} vs {counts}")

    pd2, pd2n, pi2, pi2n = F.nucleotide_features([seq], 2)
    check("order-2 position-dependent: 29*16 = 464 features",
          pd2.shape[1] == 464 and pd2.sum() == 29)
    check("order-2 counts sum to 29 dinucleotides", pi2[0].sum() == 29)

    oh = F.one_hot(probe)
    idx = F.BASE_INDEX
    check("one-hot matches the sequence exactly",
          all(oh[0, p, idx[b]] == 1 for p, b in enumerate(seq)) and oh[0].sum() == 30)

    tm, tm_names = F.tm_features([seq])
    check("all four Tm values are finite and physically plausible",
          np.all(np.isfinite(tm)) and np.all(tm > -50) and np.all(tm < 150),
          f"{dict(zip(tm_names, np.round(tm[0], 2)))}")
    tm_all, _ = F.tm_features(df["30mer"].tolist())
    check("no non-finite Tm anywhere in the real dataset",
          np.isfinite(tm_all).all(),
          f"ranges: {[f'{tm_all[:,i].min():.1f}..{tm_all[:,i].max():.1f}' for i in range(4)]}")

    Xp, names = F.rule_set_2_features(probe)
    check("full feature matrix width == 630 and names align",
          Xp.shape[1] == 630 == len(names))
    # POSITION_LABELS contains 'G' twice (the GG of the PAM), so pd1 feature names
    # are not unique. Confirm that is the only source of duplication.
    # POSITION_LABELS spells the PAM as N, G, G, so the label 'G' appears twice.
    # That makes exactly 4 order-1 names and 16 order-2 names non-unique. Feature
    # *columns* stay distinct; only the human-readable names collide, which matters
    # solely for the positional aggregation in Phase 5 (handled there by splitting
    # importance across both positions).
    dup = pd.Series(names).duplicated().sum()
    check("repeated feature names are exactly the 20 caused by the PAM's two Gs",
          dup == 20, f"{dup} duplicated names = 4 (order-1) + 16 (order-2)")

    # ================================================================== V4
    print("\nV4  OUT-OF-FOLD PREDICTION INTEGRITY")
    print("-" * 70)
    oofs = {n: np.load(os.path.join(ART, f)) for n, f in [
        ("RS2 GBT", "oof_azimuth_gbt.npy"), ("LightGBM", "oof_lgbm_gbt.npy"),
        ("CNN", "oof_cnn.npy")]}
    for n, o in oofs.items():
        check(f"{n}: every row has an out-of-fold prediction, no NaN",
              len(o) == len(df) and np.isfinite(o).all())
    check("OOF predictions are not degenerate (non-zero variance)",
          all(np.std(o) > 1e-3 for o in oofs.values()),
          f"sds: {({n: round(float(np.std(o)),4) for n, o in oofs.items()})}")
    check("the three models are correlated but not identical",
          0.4 < np.corrcoef(oofs['RS2 GBT'], oofs['CNN'])[0, 1] < 0.99,
          f"GBT-CNN r = {np.corrcoef(oofs['RS2 GBT'], oofs['CNN'])[0,1]:.3f}")

    # ================================================================== V5
    print("\nV5  LABEL-SHUFFLE NEGATIVE CONTROL")
    print("-" * 70)
    print("  If the split or feature pipeline leaked, a model trained on shuffled")
    print("  labels would still score above zero. It must not.")
    rng = np.random.default_rng(0)

    # Shuffle within gene group, which preserves the marginal rank distribution
    # and so is the strictest version of this test.
    y_shuf = y.copy()
    for (_, _), g in df.groupby(["Target gene", "drug"]):
        i = g.index.values
        y_shuf[i] = rng.permutation(y_shuf[i])

    tr = (df.split_gene == "train").values
    te = (df.split_gene == "test").values
    p_shuf = B.fit_predict(B.azimuth_gbt(), X[tr], y_shuf[tr], X[te])
    s_shuf = E.summarize(df[te], y_shuf[te], p_shuf, "shuffled")["spearman_per_gene_mean"]
    p_real = B.fit_predict(B.azimuth_gbt(), X[tr], y[tr], X[te])
    s_real = E.summarize(df[te], y[te], p_real, "real")["spearman_per_gene_mean"]
    check("GBT on shuffled labels collapses to ~0",
          abs(s_shuf) < 0.06,
          f"shuffled = {s_shuf:+.4f}  vs  real = {s_real:+.4f}")
    check("real signal is far above the shuffled control",
          s_real - abs(s_shuf) > 0.3, f"gap = {s_real - abs(s_shuf):.4f}")

    # ================================================================== V6
    print("\nV6  AGREEMENT WITH THE RELEASED AZIMUTH MODEL")
    print("-" * 70)
    print("  Our features + Azimuth's hyperparameters, trained on everything, should")
    print("  closely track the shipped predictions. A weak match would mean the")
    print("  feature reproduction is wrong.")
    m = B.azimuth_gbt()
    m.fit(X, y)
    mine = m.predict(X)
    r = stats.spearmanr(mine, df["azimuth_prediction"]).statistic
    check("in-sample refit agrees with the released model (Spearman > 0.85)",
          r > 0.85, f"Spearman vs shipped predictions = {r:.4f}")
    r_self = stats.spearmanr(mine, y).statistic
    check("in-sample fit reaches a similar ceiling to the released model",
          abs(r_self - 0.7148) < 0.15,
          f"ours in-sample {r_self:.4f} vs shipped {0.7148:.4f}")

    # ================================================================== V7
    print("\nV7  CONFORMAL MACHINERY ON A CASE WITH A KNOWN ANSWER")
    print("-" * 70)
    # Exchangeable synthetic data: coverage must hit nominal within sampling error.
    g = np.random.default_rng(1)
    n = 4000
    yt = g.normal(0, 1, n)
    pr = yt + g.normal(0, 0.5, n)
    cal, tst = slice(0, n // 2), slice(n // 2, n)
    rows = C.coverage_curve(yt[cal], pr[cal], yt[tst], pr[tst],
                            alphas=[0.05, 0.1, 0.2], clip=(None, None))
    gaps = [abs(r_["empirical_coverage"] - r_["nominal_coverage"]) for r_ in rows]
    check("synthetic exchangeable data: coverage tracks nominal within 2%",
          max(gaps) < 0.02,
          "  ".join(f"{r_['nominal_coverage']:.2f}->{r_['empirical_coverage']:.3f}"
                    for r_ in rows))
    # The finite-sample correction must be the ceil((n+1)(1-a))/n order statistic.
    s = np.arange(1, 101, dtype=float)
    check("quantile uses the ceil((n+1)(1-alpha)) order statistic",
          C.conformal_quantile(s, 0.1) == 91.0 and C.conformal_quantile(s, 0.05) == 96.0,
          f"alpha=0.10 -> {C.conformal_quantile(s,0.1)} (expect 91), "
          f"alpha=0.05 -> {C.conformal_quantile(s,0.05)} (expect 96)")
    # alphas ascend, so nominal coverage descends; empirical must descend with it.
    check("coverage decreases as alpha increases",
          all(rows[i]["empirical_coverage"] >= rows[i + 1]["empirical_coverage"] - 1e-9
              for i in range(len(rows) - 1)),
          "  ".join(f"a={r_['alpha']}: {r_['empirical_coverage']:.3f}" for r_ in rows))

    # Conformal calibration/test sets must be disjoint in the real runs.
    for col in ["split_random", "split_gene"]:
        c_ = set(df[df[col] == "cal"].index)
        t_ = set(df[df[col] == "test"].index)
        check(f"{col}: conformal calibration and test rows are disjoint",
              len(c_ & t_) == 0)

    # Does the duplicate-sequence leak in the random split change the conformal
    # conclusion? Re-run coverage on only those test rows whose 30mer the model
    # never saw during training.
    pred_r = np.load(os.path.join(ART, "pred_random_cnn.npy"))
    seen = set(df[df.split_random == "train"]["30mer"])
    cal_m = (df.split_random == "cal").values
    test_m = (df.split_random == "test").values
    clean = test_m & ~df["30mer"].isin(seen).values
    r_all = C.coverage_curve(y[cal_m], pred_r[cal_m], y[test_m], pred_r[test_m],
                             alphas=[0.1])[0]["empirical_coverage"]
    r_clean = C.coverage_curve(y[cal_m], pred_r[cal_m], y[clean], pred_r[clean],
                               alphas=[0.1])[0]["empirical_coverage"]
    check("random-split 90% coverage holds after removing leaked test rows",
          abs(r_clean - 0.90) < 0.03,
          f"all test rows: {r_all:.3f} (n={test_m.sum()})   "
          f"unseen sequences only: {r_clean:.3f} (n={clean.sum()})")

    # ================================================================== V8
    print("\nV8  HEADLINE NUMBERS RECOMPUTED FROM ARTIFACTS")
    print("-" * 70)
    with open(os.path.join(ART, "benchmark.json")) as f:
        bench = {(r["protocol"], r["model"]): r for r in json.load(f)["table"]}
    recomputed = {
        ("leave-one-gene-out", "Rule Set 2 GBT (Azimuth params)"): oofs["RS2 GBT"],
        ("leave-one-gene-out", "LightGBM (Rule Set 2 features)"): oofs["LightGBM"],
        ("leave-one-gene-out", "CNN + attention"): oofs["CNN"],
    }
    for key, pred in recomputed.items():
        got = E.summarize(df, y, pred, "")["spearman_per_gene_mean"]
        want = bench[key]["spearman_per_gene_mean"]
        check(f"{key[1]} LOGO reproduces from saved OOF",
              abs(got - want) < 1e-9, f"{got:.6f} vs reported {want:.6f}")

    gh = np.load(os.path.join(ART, "pred_geneheld_cnn.npy"))
    got = E.summarize(df[te], y[te], gh[te], "")["spearman_per_gene_mean"]
    check("CNN gene-held-out test reproduces from saved predictions",
          abs(got - 0.4950) < 5e-4, f"{got:.4f} vs reported 0.4950")

    baseline_logo = E.summarize(df, y, oofs["RS2 GBT"], "")["spearman_per_gene_mean"]
    check("baseline sits in the Doench 2016 Fig. 4c band (0.4-0.6)",
          0.4 <= baseline_logo <= 0.6, f"{baseline_logo:.4f}")

    # ================================================================== summary
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 70)
    print(f"VALIDATION SUMMARY: {n_pass}/{len(RESULTS)} checks passed")
    print("=" * 70)
    failed = [n for n, ok, _ in RESULTS if not ok]
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
    with open(os.path.join(ART, "validation.json"), "w") as f:
        json.dump([{"check": n, "passed": ok, "detail": d} for n, ok, d in RESULTS],
                  f, indent=2)
    return not failed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
