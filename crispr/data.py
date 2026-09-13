"""Loading and splitting the Doench et al. 2016 FC+RES (Azimuth "V3") dataset.

Everything here is validated against the Azimuth source in ``_azimuth_src/``:

  * ``FC_plus_RES_withPredictions.csv`` is the merged V1(FC)+V2(RES) table that
    ``load_data.mergeV1_V2`` writes out under its ``save_to_file`` branch.
  * The target is ``score_drug_gene_rank``: the measured activity rank-transformed
    within each (gene, drug) group and scaled to (0, 1], higher = more active.
    This is ``learn_options['rank-transformed target name']`` for V=3.
  * 30mers are 4nt upstream + 20nt protospacer + 3nt PAM + 3nt downstream, per
    ``featurization.nucleotide_features_dictionary``.

``reconstruct_v3_from_xlsx`` rebuilds the same table from the raw V1/V2 Excel
files so the shipped CSV is verified rather than assumed.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AZIMUTH_DATA = os.path.join(REPO_ROOT, "_azimuth_src", "azimuth", "data")

V3_CSV = os.path.join(AZIMUTH_DATA, "FC_plus_RES_withPredictions.csv")
V1_XLSX = os.path.join(AZIMUTH_DATA, "V1_data.xlsx")
V2_XLSX = os.path.join(AZIMUTH_DATA, "V2_data.xlsx")

TARGET = "score_drug_gene_rank"

# Layout of the 30mer (0-indexed), per Azimuth's own position naming.
CONTEXT_5P = slice(0, 4)      # -4 .. -1
PROTOSPACER = slice(4, 24)    #  1 .. 20
PAM = slice(24, 27)           #  N G G
CONTEXT_3P = slice(27, 30)    # +1 .. +3

# The "seed" region: protospacer positions 17-20, i.e. PAM-proximal.
SEED_PROTOSPACER_POSITIONS = (17, 18, 19, 20)
SEED_30MER_INDICES = (20, 21, 22, 23)

# Drug/gene pairs used to build the RES half, from load_data.read_V2_data's
# ``known_pairs`` (learn_options is None when called via mergeV1_V2, so the
# 'extra pairs'/'all pairs' expansions are NOT applied).
V2_DRUGS_TO_GENES = {
    "AZD_200nM": ["CCDC101", "MED12", "TADA2B", "TADA1"],
    "6TG_2ug/mL": ["HPRT1"],
    "PLX_2uM": ["CUL3", "NF1", "NF2"],
}
V2_DRUGS_TO_GENES["PLX_2uM"].append("MED12")

# FC (V1) human gene -> the cell-type readout columns averaged for that gene,
# from load_data.combine_organisms.
V1_HUMAN_READOUTS = {
    "CD13": ["NB4 CD13", "TF1 CD13"],
    "CD33": ["MOLM13 CD33", "TF1 CD33", "NB4 CD33"],
    "CD15": ["MOLM13 CD15"],
}


def _rank_unit(x: np.ndarray) -> np.ndarray:
    """Azimuth's rank transform: average-tied ranks divided by the max rank."""
    r = stats.rankdata(x, method="average")
    return r / r.max()


def load_v3(verify: bool = True) -> pd.DataFrame:
    """Load the FC+RES (V3) table.

    Returns a frame with one row per (guide, drug) observation and columns:
    ``30mer``, ``Target gene``, ``drug``, ``Percent Peptide``,
    ``Amino Acid Cut position``, ``score_drug_gene_rank``,
    ``score_drug_gene_threshold``, ``azimuth_prediction``, ``dataset``.
    """
    if not os.path.exists(V3_CSV):
        raise FileNotFoundError(
            f"{V3_CSV} not found. Run scripts/fetch_azimuth.sh to download the "
            "Azimuth source tree."
        )
    df = pd.read_csv(V3_CSV, index_col=0).reset_index(drop=True)
    df = df.rename(columns={"predictions": "azimuth_prediction"})

    # 'nodrug' marks the FC (flow cytometry, Doench 2014) half; the RES half
    # carries a real small-molecule label.
    df["dataset"] = np.where(df["drug"] == "nodrug", "FC", "RES")

    if verify:
        assert (df["30mer"].str.len() == 30).all(), "expected 30mers"
        alphabet = set("".join(df["30mer"]))
        assert alphabet <= set("ACGT"), f"unexpected bases: {alphabet - set('ACGT')}"
        # PAM audit, mirroring featurization.Tm_feature / NGGX_interaction_feature.
        assert (df["30mer"].str[25:27] == "GG").all(), "expected GG at 30mer[25:27]"
        assert df[TARGET].between(0, 1).all(), "target should be a unit-scaled rank"
        assert not df.isnull().any().any(), "unexpected nulls"

    return df


def reconstruct_v3_from_xlsx() -> pd.DataFrame:
    """Rebuild the V3 target from V1_data.xlsx / V2_data.xlsx.

    A port of ``load_data.mergeV1_V2`` (Python 2) sufficient to regenerate
    ``score_drug_gene_rank``, used to verify the shipped CSV. Gene-position
    columns are not reconstructed (they need V1_suppl_data.txt joins that only
    affect which rows survive, not the target).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v1_human = pd.read_excel(V1_XLSX, sheet_name=0, index_col=[0, 1])
        v1_mouse = pd.read_excel(V1_XLSX, sheet_name=1, index_col=[0, 1])
        v2 = pd.read_excel(
            V2_XLSX, sheet_name="ResultsFiltered", skiprows=range(0, 7), index_col=[0, 4]
        )

    rows = []

    # --- FC / V1 -----------------------------------------------------------
    # util.get_data ranks each cell-type readout within the gene, then averages
    # those ranks across cell types.
    for gene, readouts in V1_HUMAN_READOUTS.items():
        sub = v1_human.xs(gene, level="Target", drop_level=False)
        ranks = np.column_stack([_rank_unit(sub[c].values) for c in readouts])
        rows.append(
            pd.DataFrame({
                "30mer": sub["30mer"].values,
                "Target gene": gene,
                "drug": "nodrug",
                TARGET: ranks.mean(axis=1),
            })
        )

    for gene in v1_mouse.index.get_level_values(1).unique():
        sub = v1_mouse.xs(gene, level=v1_mouse.index.names[1], drop_level=False)
        rows.append(
            pd.DataFrame({
                "30mer": sub["30mer"].values,
                "Target gene": gene,
                "drug": "nodrug",
                TARGET: _rank_unit(sub["On-target Gene"].values),
            })
        )

    # --- RES / V2 ----------------------------------------------------------
    for drug, genes in V2_DRUGS_TO_GENES.items():
        for gene in genes:
            sub = v2.xs(gene, level="Target gene", drop_level=False)
            rows.append(
                pd.DataFrame({
                    "30mer": sub["30mer"].values,
                    "Target gene": gene,
                    "drug": drug,
                    TARGET: _rank_unit(sub[drug].values),
                })
            )

    out = pd.concat(rows, ignore_index=True)
    out["30mer"] = out["30mer"].str.upper().str[:30]
    return out


def verify_against_xlsx(df: pd.DataFrame) -> dict:
    """Compare the shipped CSV target to one rebuilt from the raw Excel files."""
    rebuilt = reconstruct_v3_from_xlsx()
    key = ["30mer", "Target gene", "drug"]
    merged = df[key + [TARGET]].merge(
        rebuilt[key + [TARGET]].drop_duplicates(key),
        on=key, how="inner", suffixes=("_csv", "_rebuilt"),
    )
    diff = (merged[f"{TARGET}_csv"] - merged[f"{TARGET}_rebuilt"]).abs()
    return {
        "n_csv": len(df),
        "n_rebuilt": len(rebuilt),
        "n_matched": len(merged),
        "max_abs_diff": float(diff.max()) if len(diff) else float("nan"),
        "pearson": float(
            np.corrcoef(merged[f"{TARGET}_csv"], merged[f"{TARGET}_rebuilt"])[0, 1]
        ) if len(merged) > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def leave_one_gene_out_folds(df: pd.DataFrame, min_guides: int = 10):
    """Yield ``(name, train_idx, test_idx)`` for each held-out gene.

    This is Azimuth's default protocol (``learn_options['cv'] = 'gene'``) and is
    how Doench 2016 Fig. 4 reports Spearman correlation. Genes with fewer than
    ``min_guides`` guides are still trained on but not scored as a held-out fold,
    since a Spearman over a handful of points is pure noise.
    """
    genes = df["Target gene"].unique()
    for gene in sorted(genes):
        test_mask = (df["Target gene"] == gene).values
        if test_mask.sum() < min_guides:
            continue
        yield gene, np.where(~test_mask)[0], np.where(test_mask)[0]


def gene_grouped_split(df: pd.DataFrame, test_frac: float = 0.2, cal_frac: float = 0.2):
    """Partition whole genes into train / calibration / test.

    Gene sizes are wildly uneven -- MED12 alone is ~35% of rows -- so a random
    gene assignment gives useless proportions. Instead we walk genes largest-first
    and give each one to whichever split is currently furthest below its row quota.
    This is deterministic, keeps the row proportions close to target, and spreads
    the many small genes across all three splits so each has several genes to
    compute per-gene metrics over.
    """
    sizes = df["Target gene"].value_counts()
    total = len(df)
    quota = {
        "train": (1.0 - test_frac - cal_frac) * total,
        "cal": cal_frac * total,
        "test": test_frac * total,
    }
    acc = {k: 0 for k in quota}
    assign = {}
    for gene in sizes.index:  # value_counts is already descending
        target = max(quota, key=lambda k: quota[k] - acc[k])
        assign[str(gene)] = target
        acc[target] += sizes[gene]

    split = df["Target gene"].astype(str).map(assign).values
    return split, assign


def random_split(df: pd.DataFrame, test_frac=0.2, cal_frac=0.2, seed: int = 0):
    """Row-level random split. Optimistic -- kept only as a leakage contrast.

    Guides tiling the same gene share sequence context and the target is ranked
    within gene, so a random split leaks gene-level information into the test set.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(len(df))
    split = np.where(u < test_frac, "test", np.where(u < test_frac + cal_frac, "cal", "train"))
    return split
