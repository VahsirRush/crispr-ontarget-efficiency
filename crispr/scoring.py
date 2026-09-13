"""Score an arbitrary guide with the trained CNN ensemble plus a conformal interval.

This is the inference path behind both dashboards. It loads the *gene-held-out*
ensemble -- the one whose test number is reported -- and calibrates intervals on
the held-out calibration genes, so a score produced here is directly comparable
to the benchmark rather than to an in-sample fit.

Two things a caller has to understand:

  * The thermodynamic block the model consumes includes three gene-position
    features (percent peptide, amino-acid cut position, and the <50% indicator).
    They are not derivable from a 30mer. ``score`` therefore takes them as
    explicit arguments and defaults to the training-set median, which makes the
    prediction *sequence-conditional at an average locus* rather than wrong-but-
    silent. ``scripts/03a_sweep_cnn.py`` measured what dropping them costs
    (0.377 -> 0.179 inner-CV Spearman), so they are not decoration.

  * Intervals come from the gene-held-out calibration split, where calibration
    and test genes are disjoint. Exchangeability does not hold there, so the
    coverage guarantee is empirical (93.4% at nominal 90%), not formal.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from . import conformal as C
from . import data as D
from . import features as F
from . import model as M

ART = os.path.join(D.REPO_ROOT, "artifacts")
VALID_BASES = set("ACGT")

# Nominal levels the widget offers. Precomputed so the UI never recalibrates.
ALPHAS = (0.20, 0.10, 0.05)


class GuideSequenceError(ValueError):
    """Raised for input that is not a usable 30mer."""


def validate_30mer(seq: str) -> str:
    """Normalise and check a guide context sequence, or explain why it is unusable."""
    s = (seq or "").strip().upper().replace(" ", "").replace("\n", "")
    if len(s) != 30:
        raise GuideSequenceError(
            f"Expected a 30nt context sequence, got {len(s)}nt. "
            "The layout is 4nt upstream + 20nt protospacer + 3nt PAM + 3nt downstream."
        )
    bad = sorted(set(s) - VALID_BASES)
    if bad:
        raise GuideSequenceError(f"Unexpected characters {bad}; only A, C, G, T are allowed.")
    if s[25:27] != "GG":
        raise GuideSequenceError(
            f"Positions 26-27 must be the 'GG' of the NGG PAM, found '{s[25:27]}'. "
            "The model was only ever trained on SpCas9 NGG sites."
        )
    return s


@lru_cache(maxsize=1)
def _bundle():
    """Load the ensemble, the standardiser, and the conformal quantiles once."""
    ck = torch.load(os.path.join(ART, "cnn_geneheld.pt"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]

    models = []
    for sd in ck["state_dicts"]:
        m = M.CrisprCNN(
            n_channels=ck["n_channels"], n_thermo=ck["n_thermo"],
            dropout=cfg["dropout"], conv1_filters=cfg["c1"], conv2_filters=cfg["c2"],
            attn_dim=cfg["attn_dim"], conv_dropout=cfg["conv_dropout"],
            positional=cfg["positional"],
        )
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    std = M.Standardizer()
    std.mu, std.sd = np.asarray(ck["thermo_mu"]), np.asarray(ck["thermo_sd"])

    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    pred = np.load(os.path.join(ART, "pred_geneheld_cnn.npy"))
    y = df[D.TARGET].values.astype(np.float64)

    cal = df["split_gene"].values == "cal"
    scores = C.absolute_residual_scores(y[cal], pred[cal])
    q = {a: C.conformal_quantile(scores, a) for a in ALPHAS}

    train = df["split_gene"].values == "train"
    defaults = {
        "percent_peptide": float(np.median(df.loc[train, "Percent Peptide"])),
        "aa_cut_position": float(np.median(df.loc[train, "Amino Acid Cut position"])),
    }
    return models, std, cfg, q, int(cal.sum()), defaults


def conformal_quantiles() -> dict[float, float]:
    return _bundle()[3]


def default_gene_position() -> dict[str, float]:
    return _bundle()[5]


def thermo_vector(seq: str, percent_peptide: float, aa_cut_position: float) -> np.ndarray:
    """The 10-feature block in the exact column order the model was trained on."""
    gc, _ = F.gc_features([seq])
    tm, _ = F.tm_features([seq])
    pos = np.array([[percent_peptide, aa_cut_position, float(percent_peptide < 50)]],
                   dtype=np.float32)
    return np.hstack([gc, tm, pos]).astype(np.float32)


def score(seq: str, percent_peptide: float | None = None,
          aa_cut_position: float | None = None, alpha: float = 0.10) -> dict:
    """Predict efficiency for one guide, with an attention profile and an interval.

    Returns the ensemble mean, the per-model spread (which is *model* uncertainty,
    not the calibrated interval), the 30 attention-pooling weights, and the
    conformal interval at the requested level.
    """
    seq = validate_30mer(seq)
    models, std, _, q, n_cal, defaults = _bundle()

    if percent_peptide is None:
        percent_peptide = defaults["percent_peptide"]
    if aa_cut_position is None:
        aa_cut_position = defaults["aa_cut_position"]

    onehot = F.one_hot(pd.DataFrame({"30mer": [seq]}))
    thermo = std.transform(thermo_vector(seq, percent_peptide, aa_cut_position))

    preds, pools = [], []
    for m in models:
        p, pw, _ = M.predict(m, onehot, thermo, return_attn=True)
        preds.append(float(p[0]))
        pools.append(pw[0])

    mean = float(np.mean(preds))
    if alpha not in q:
        raise ValueError(f"alpha must be one of {sorted(q)}; got {alpha}")
    half = q[alpha]

    return {
        "sequence": seq,
        "protospacer": seq[D.PROTOSPACER],
        "prediction": mean,
        "per_seed": preds,
        "seed_spread": float(np.std(preds)),
        "attention": np.mean(pools, axis=0).tolist(),
        "interval": (max(0.0, mean - half), min(1.0, mean + half)),
        "half_width": float(half),
        "alpha": alpha,
        "n_calibration": n_cal,
        "gc_count": int(len(seq[D.PROTOSPACER].replace("A", "").replace("T", ""))),
        "percent_peptide": float(percent_peptide),
        "aa_cut_position": float(aa_cut_position),
        "used_default_position": percent_peptide == defaults["percent_peptide"],
    }
