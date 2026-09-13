"""Evaluation metrics, matched to how Doench 2016 reports them.

The paper's headline number (Fig. 4c) is the Spearman correlation between measured
and predicted activity computed *per gene* under leave-one-gene-out, with error
bars being the standard deviation across genes. Pooling all guides into one
Spearman is a different (and generally higher) quantity, because the target is
rank-normalised within each (gene, drug) group. We report both and treat the
per-gene mean as the paper-comparable figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def spearman(y_true, y_pred) -> float:
    if len(y_true) < 3 or np.std(y_pred) == 0:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).statistic)


def per_gene_spearman(df: pd.DataFrame, y_true, y_pred, min_n: int = 10) -> pd.DataFrame:
    """Spearman within each (gene, drug) group -- the unit the target was ranked in."""
    out = []
    work = pd.DataFrame({
        "gene": df["Target gene"].values,
        "drug": df["drug"].values,
        "y": np.asarray(y_true),
        "p": np.asarray(y_pred),
    })
    for (gene, drug), g in work.groupby(["gene", "drug"]):
        if len(g) < min_n:
            continue
        out.append({"gene": gene, "drug": drug, "n": len(g),
                    "spearman": spearman(g["y"], g["p"])})
    return pd.DataFrame(out).sort_values("spearman").reset_index(drop=True)


def summarize(df: pd.DataFrame, y_true, y_pred, label: str = "") -> dict:
    pg = per_gene_spearman(df, y_true, y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "label": label,
        "n": len(y_true),
        "n_gene_groups": len(pg),
        "spearman_pooled": spearman(y_true, y_pred),
        "spearman_per_gene_mean": float(pg["spearman"].mean()) if len(pg) else float("nan"),
        "spearman_per_gene_sd": float(pg["spearman"].std()) if len(pg) > 1 else float("nan"),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan"),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "_per_gene": pg,
    }


def format_summary(s: dict) -> str:
    return (
        f"{s['label']:<38} n={s['n']:<6} "
        f"per-gene Spearman={s['spearman_per_gene_mean']:.4f} "
        f"(sd {s['spearman_per_gene_sd']:.3f}, {s['n_gene_groups']} groups)  "
        f"pooled={s['spearman_pooled']:.4f}  rmse={s['rmse']:.4f}"
    )
