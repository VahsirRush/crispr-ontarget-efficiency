"""Split conformal prediction intervals.

Standard split (inductive) conformal regression: fit on the training set, compute
nonconformity scores on a disjoint calibration set, and take the appropriate
empirical quantile as a half-width.

For a miscoverage level alpha, the quantile index uses the finite-sample
correction ceil((n+1)(1-alpha))/n. With that correction and *exchangeable*
calibration and test data, the interval is guaranteed to cover with probability
at least 1-alpha -- no assumption about the model being correct.

The exchangeability caveat matters here. Under the gene-held-out split, the
calibration guides and the test guides come from different genes, so they are not
exchangeable and the guarantee does not formally hold. That is exactly why
``coverage_curve`` is run under both split types: the random split shows the
method is implemented correctly, and the gene-held-out split shows what the
distribution shift costs in practice.
"""

from __future__ import annotations

import numpy as np


def absolute_residual_scores(y_true, y_pred) -> np.ndarray:
    return np.abs(np.asarray(y_true) - np.asarray(y_pred))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of the calibration scores."""
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        # Not enough calibration points to certify this level; widest possible.
        return float(np.max(scores))
    return float(np.sort(scores)[k - 1])


def predict_interval(y_pred, q: float, lo: float | None = None, hi: float | None = None):
    """Symmetric interval y_pred +/- q, optionally clipped to the target's range.

    Clipping to the known support only ever shrinks the interval toward values the
    target can actually take, so empirical coverage cannot go down from it.
    """
    y_pred = np.asarray(y_pred)
    low, high = y_pred - q, y_pred + q
    if lo is not None:
        low = np.maximum(low, lo)
    if hi is not None:
        high = np.minimum(high, hi)
    return low, high


def coverage(y_true, low, high) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= low) & (y_true <= high)))


def coverage_curve(
    y_cal, pred_cal, y_test, pred_test,
    alphas=None, clip=(0.0, 1.0),
):
    """Empirical coverage and mean interval width across nominal levels."""
    if alphas is None:
        alphas = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = absolute_residual_scores(y_cal, pred_cal)
    rows = []
    for a in alphas:
        q = conformal_quantile(scores, a)
        low, high = predict_interval(pred_test, q, *clip)
        rows.append({
            "alpha": float(a),
            "nominal_coverage": float(1 - a),
            "q": q,
            "empirical_coverage": coverage(y_test, low, high),
            "mean_width": float(np.mean(high - low)),
            "n_cal": len(scores),
            "n_test": len(y_test),
        })
    return rows


def mondrian_coverage_curve(
    y_cal, pred_cal, cal_groups, y_test, pred_test, test_groups,
    alphas=None, clip=(0.0, 1.0),
):
    """Group-conditional ("Mondrian") conformal: a separate quantile per group.

    Only usable when calibration and test share group labels. Under the random
    split the groups are genes present in both; under the gene-held-out split they
    are not, which is itself the point.
    """
    if alphas is None:
        alphas = np.round(np.arange(0.05, 0.96, 0.05), 2)
    cal_groups = np.asarray(cal_groups)
    test_groups = np.asarray(test_groups)
    shared = set(np.unique(cal_groups)) & set(np.unique(test_groups))
    if not shared:
        return []
    rows = []
    for a in alphas:
        covered, widths, n = [], [], 0
        for g in shared:
            cm, tm = cal_groups == g, test_groups == g
            if cm.sum() < 10 or tm.sum() == 0:
                continue
            q = conformal_quantile(absolute_residual_scores(y_cal[cm], pred_cal[cm]), a)
            low, high = predict_interval(pred_test[tm], q, *clip)
            covered.append(((y_test[tm] >= low) & (y_test[tm] <= high)).sum())
            widths.append((high - low).sum())
            n += tm.sum()
        if n == 0:
            continue
        rows.append({
            "alpha": float(a),
            "nominal_coverage": float(1 - a),
            "empirical_coverage": float(sum(covered) / n),
            "mean_width": float(sum(widths) / n),
            "n_test": int(n),
            "n_groups": len(covered),
        })
    return rows
