"""Feature extraction mirroring Azimuth / Rule Set 2, plus one-hot tensors for the CNN.

The hand-crafted block reproduces the feature families listed in Doench 2016 and
implemented in ``_azimuth_src/azimuth/features/featurization.py``:

  * order-1 and order-2 position-dependent nucleotide indicators over the 30mer
  * order-1 and order-2 position-independent nucleotide counts
  * GC count of the 20mer protospacer, plus the GC>10 / GC<10 indicators
  * the NGGX interaction feature (one-hot of 30mer[24] + 30mer[27])
  * four melting temperatures: full 30mer, and segments [19:24], [11:19], [6:11]
  * gene position: Percent Peptide, Amino Acid Cut position, Percent Peptide < 50%

One deviation, forced by the library: Azimuth calls ``Bio.SeqUtils.MeltingTemp
.Tm_staluc``, which Biopython removed in 1.77. We use ``Tm_NN`` with the
``DNA_NN2`` table -- the SantaLucia 1998 unified nearest-neighbour parameters that
Tm_staluc implemented -- so the quantity is the same up to salt-correction
defaults. Absolute Tm values shift slightly; the ranking across guides does not.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
from Bio.SeqUtils import MeltingTemp as _Tm

from .data import PROTOSPACER

BASES = ["A", "T", "C", "G"]  # Azimuth's raw_alphabet order, kept for comparability
BASE_INDEX = {b: i for i, b in enumerate(BASES)}

# Azimuth's Tm segments (featurization.Tm_feature default).
TM_SEGMENTS = [(19, 24), (11, 19), (6, 11)]

# Azimuth's human-readable position labels for the 30mer.
POSITION_LABELS = (
    ["-4", "-3", "-2", "-1"]
    + [str(i) for i in range(1, 21)]
    + ["N", "G", "G", "+1", "+2", "+3"]
)


@lru_cache(maxsize=None)
def _alphabet(order: int):
    return ["".join(p) for p in itertools.product(BASES, repeat=order)]


@lru_cache(maxsize=100_000)
def _tm(seq: str) -> float:
    return float(_Tm.Tm_NN(seq, nn_table=_Tm.DNA_NN2))


def nucleotide_features(seqs, order: int):
    """Position-dependent and position-independent k-mer features.

    For 30mers: order 1 gives 30*4=120 dependent + 4 independent, order 2 gives
    29*16=464 dependent + 16 independent.
    """
    alpha = _alphabet(order)
    idx = {k: i for i, k in enumerate(alpha)}
    n_pos = len(seqs[0]) - order + 1

    pd_arr = np.zeros((len(seqs), n_pos * len(alpha)), dtype=np.float32)
    pi_arr = np.zeros((len(seqs), len(alpha)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for p in range(n_pos):
            j = idx[s[p:p + order]]
            pd_arr[i, p * len(alpha) + j] = 1.0
            pi_arr[i, j] += 1.0

    pd_names = [f"pd{order}_{k}_{POSITION_LABELS[p]}" for p in range(n_pos) for k in alpha]
    pi_names = [f"pi{order}_{k}" for k in alpha]
    return pd_arr, pd_names, pi_arr, pi_names


def gc_features(seqs):
    """GC count over the 20mer protospacer only, as in Azimuth's ``countGC``."""
    gc = np.array([len(s[PROTOSPACER].replace("A", "").replace("T", "")) for s in seqs],
                  dtype=np.float32)
    arr = np.column_stack([gc, (gc > 10).astype(np.float32), (gc < 10).astype(np.float32)])
    return arr, ["gc_count", "gc_above_10", "gc_below_10"]


def nggx_features(seqs):
    """One-hot of the N and X flanking the GG in the NGGX PAM context (16 features)."""
    alpha = _alphabet(2)
    idx = {k: i for i, k in enumerate(alpha)}
    arr = np.zeros((len(seqs), 16), dtype=np.float32)
    for i, s in enumerate(seqs):
        arr[i, idx[s[24] + s[27]]] = 1.0
    return arr, [f"NGGX_{k}" for k in alpha]


def tm_features(seqs):
    """Melting temperatures of the full 30mer and Azimuth's three sub-segments."""
    arr = np.zeros((len(seqs), 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        arr[i, 0] = _tm(s)
        for j, (a, b) in enumerate(TM_SEGMENTS, start=1):
            arr[i, j] = _tm(s[a:b])
    return arr, ["tm_global", "tm_5mer_pam_proximal", "tm_8mer_middle", "tm_5mer_start"]


def thermodynamic_features(df: pd.DataFrame):
    """The block fed to the CNN at the fusion step (not through the conv stack).

    Deliberately excludes the positional nucleotide features -- those are the
    conv stack's job -- and keeps the scalar biophysical / gene-position summary.
    """
    seqs = df["30mer"].tolist()
    gc, gc_names = gc_features(seqs)
    tm, tm_names = tm_features(seqs)
    pos = np.column_stack([
        df["Percent Peptide"].values,
        df["Amino Acid Cut position"].values,
        (df["Percent Peptide"].values < 50).astype(np.float32),
    ]).astype(np.float32)
    pos_names = ["percent_peptide", "aa_cut_position", "percent_peptide_lt50"]
    return np.hstack([gc, tm, pos]), gc_names + tm_names + pos_names


def rule_set_2_features(df: pd.DataFrame):
    """Full hand-crafted feature matrix for the GBT baseline."""
    seqs = df["30mer"].tolist()
    blocks, names = [], []

    for order in (1, 2):
        pd_arr, pd_names, pi_arr, pi_names = nucleotide_features(seqs, order)
        blocks += [pd_arr, pi_arr]
        names += pd_names + pi_names

    for arr, nm in (gc_features(seqs), nggx_features(seqs), tm_features(seqs)):
        blocks.append(arr)
        names += nm

    pos = np.column_stack([
        df["Percent Peptide"].values,
        df["Amino Acid Cut position"].values,
        (df["Percent Peptide"].values < 50).astype(np.float32),
    ]).astype(np.float32)
    blocks.append(pos)
    names += ["percent_peptide", "aa_cut_position", "percent_peptide_lt50"]

    return np.hstack(blocks).astype(np.float32), names


def one_hot(df: pd.DataFrame) -> np.ndarray:
    """(N, 30, 4) one-hot encoding of the 30mer context sequence."""
    seqs = df["30mer"].tolist()
    arr = np.zeros((len(seqs), 30, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for p, b in enumerate(s):
            arr[i, p, BASE_INDEX[b]] = 1.0
    return arr
