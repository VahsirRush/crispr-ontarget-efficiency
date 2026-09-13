"""Phase 9: export everything the two dashboards need into one payload.

Writes two files under ``dashboard/data/``:

  ``dashboard.json``  every reported number, the per-guide prediction table, and
                      the curves. Both the Streamlit app and the static build
                      read this, so the two can never drift apart.

  ``model.json``      the gene-held-out CNN ensemble as base64 float32 blobs,
                      plus the standardiser statistics, the conformal quantiles,
                      and the nearest-neighbour thermodynamic constants. This is
                      what lets the static HTML score a pasted guide in the
                      browser with no server.

The melting-temperature export needs care. Azimuth calls Biopython's ``Tm_NN``
with the SantaLucia/Sugimoto ``DNA_NN2`` table against a perfect complement. In
that special case the mismatch and dangling-end branches are dead code and the
whole calculation collapses to a 16-entry dinucleotide sum, which ports cleanly
to JavaScript. This script asserts that the branches really are dead rather than
assuming it, and ``scripts/10_build_static_dashboard.py`` then checks the ported
arithmetic numerically against Biopython.
"""

import base64
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from Bio.SeqUtils import MeltingTemp as _Tm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import conformal as C
from crispr import data as D
from crispr import features as F
from crispr import scoring as S

ART = os.path.join(D.REPO_ROOT, "artifacts")
OUT = os.path.join(D.REPO_ROOT, "dashboard", "data")

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
BASES = "ACGT"


# --------------------------------------------------------------------------
# nearest-neighbour thermodynamics, reduced to what a browser needs
# --------------------------------------------------------------------------

def export_tm_constants():
    """Collapse Biopython's Tm_NN(DNA_NN2, perfect complement) to 16 constants.

    Verifies the reduction is exact rather than trusting it: the terminal- and
    internal-mismatch tables must contain none of the Watson-Crick keys, and the
    DNA_NN2 initiation terms that depend on sequence composition must all be zero.
    """
    nn = _Tm.DNA_NN2

    for name in ("init_A/T", "init_G/C", "init_oneG/C", "init_allA/T", "init_5T/A"):
        assert nn[name] == (0, 0), f"DNA_NN2[{name}] is non-zero; the reduction is invalid"

    dinuc = {}
    for a in BASES:
        for b in BASES:
            d = a + b
            key = d + "/" + COMPLEMENT[a] + COMPLEMENT[b]
            assert key not in _Tm.DNA_IMM1 and key[::-1] not in _Tm.DNA_IMM1, \
                f"{key} collides with the internal-mismatch table"
            assert key not in _Tm.DNA_TMM1 and key[::-1] not in _Tm.DNA_TMM1, \
                f"{key} collides with the terminal-mismatch table"
            if key in nn:
                dh, ds = nn[key]
            elif key[::-1] in nn:
                dh, ds = nn[key[::-1]]
            else:
                raise AssertionError(f"no DNA_NN2 entry for {key}")
            dinuc[d] = [dh, ds]

    return {
        "dinucleotide": dinuc,
        "init_dh": nn["init"][0],
        "init_ds": nn["init"][1],
        # Tm_NN defaults: dnac1 = dnac2 = 25 nM, so k = (25 - 12.5) nM.
        "k": (25 - 25 / 2.0) * 1e-9,
        "R": 1.987,
        # salt_correction method 5 with Na = 50 mM, everything else zero.
        "salt_coeff": 0.368,
        "monovalent_molar": 50 * 1e-3,
        "segments": [[19, 24], [11, 19], [6, 11]],
    }


# --------------------------------------------------------------------------
# model weights
# --------------------------------------------------------------------------

def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()


def export_model():
    ck = torch.load(os.path.join(ART, "cnn_geneheld.pt"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]

    keep = [
        "pos_emb", "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
        "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var",
        "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var",
        "attn.q.weight", "attn.q.bias", "attn.k.weight", "attn.k.bias",
        "attn.v.weight", "attn.v.bias",
        "pool.score.0.weight", "pool.score.0.bias",
        "pool.score.2.weight", "pool.score.2.bias",
        "head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias",
        "head.6.weight", "head.6.bias",
    ]

    ensemble, shapes = [], {}
    for sd in ck["state_dicts"]:
        ensemble.append({k: b64(sd[k].numpy()) for k in keep})
        shapes = {k: list(sd[k].shape) for k in keep}

    q = S.conformal_quantiles()
    return {
        "arch": {
            "c1": cfg["c1"], "c2": cfg["c2"], "attn_dim": cfg["attn_dim"],
            "seq_len": 30, "n_channels": ck["n_channels"], "n_thermo": ck["n_thermo"],
            "positional": cfg["positional"], "bn_eps": 1e-5,
        },
        "shapes": shapes,
        "ensemble": ensemble,
        "thermo_mu": np.asarray(ck["thermo_mu"], dtype=float).tolist(),
        "thermo_sd": np.asarray(ck["thermo_sd"], dtype=float).tolist(),
        "thermo_names": [
            "gc_count", "gc_above_10", "gc_below_10", "tm_global",
            "tm_5mer_pam_proximal", "tm_8mer_middle", "tm_5mer_start",
            "percent_peptide", "aa_cut_position", "percent_peptide_lt50",
        ],
        "conformal_q": {str(a): v for a, v in q.items()},
        "defaults": S.default_gene_position(),
        "tm": export_tm_constants(),
        "bases": list(F.BASES),
    }


# --------------------------------------------------------------------------
# results payload
# --------------------------------------------------------------------------

def export_dashboard():
    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    y = df[D.TARGET].values

    oof = {
        "gbt": np.load(os.path.join(ART, "oof_azimuth_gbt.npy")),
        "lgbm": np.load(os.path.join(ART, "oof_lgbm_gbt.npy")),
        "cnn": np.load(os.path.join(ART, "oof_cnn.npy")),
    }
    gh_cnn = np.load(os.path.join(ART, "pred_geneheld_cnn.npy"))

    load = lambda n: json.load(open(os.path.join(ART, n)))
    bench = load("benchmark.json")
    conf = load("conformal_results.json")
    interp = load("interpretability.json")
    sweep = load("cnn_sweep.json")
    valid = load("validation.json")
    chrom = load("chromatin_feasibility.json")
    splits = load("splits.json")

    attn = pd.read_csv(os.path.join(ART, "attention_by_position.csv"))
    per_gene = (
        pd.read_csv(os.path.join(ART, "baseline_per_gene.csv"))
        .merge(pd.read_csv(os.path.join(ART, "cnn_per_gene.csv")),
               on=["gene", "drug"], suffixes=("_gbt", "_cnn"))
    )

    q = S.conformal_quantiles()
    q90 = q[0.10]

    guides = pd.DataFrame({
        "seq": df["30mer"],
        "gene": df["Target gene"],
        "drug": df["drug"],
        "y": np.round(y, 4),
        "gbt": np.round(oof["gbt"], 4),
        "lgbm": np.round(oof["lgbm"], 4),
        "cnn": np.round(oof["cnn"], 4),
        "gh": np.round(gh_cnn, 4),
        "split": df["split_gene"],
        "pp": np.round(df["Percent Peptide"], 2),
        "cut": np.round(df["Amino Acid Cut position"], 1),
    })
    # Rank within (gene, drug) so the explorer can say "12th best of 924 in MED12".
    guides["rank"] = (
        guides.groupby(["gene", "drug"])["y"].rank(ascending=False, method="min").astype(int)
    )
    guides["group_n"] = guides.groupby(["gene", "drug"])["y"].transform("size").astype(int)

    payload = {
        "meta": {
            "dataset": "Doench et al. 2016 (Nat Biotechnol 34:184) FC+RES, via MicrosoftResearch/Azimuth",
            "n_rows": int(len(df)),
            "n_unique_guides": int(df["30mer"].nunique()),
            "n_genes": int(df["Target gene"].nunique()),
            "target": D.TARGET,
            "target_note": "activity rank-transformed within each (gene, drug) group, scaled to (0, 1]",
            "conformal_q90": q90,
            "n_calibration": int((df["split_gene"] == "cal").sum()),
        },
        "benchmark": bench,
        "conformal": conf,
        "interpretability": interp,
        "sweep": sweep,
        "validation": valid,
        "chromatin": chrom,
        "splits": splits,
        "attention_by_position": attn.to_dict("records"),
        "per_gene": per_gene.to_dict("records"),
        "guides": guides.to_dict("records"),
        "gene_summary": (
            df.groupby("Target gene")
            .agg(n=("30mer", "size"), split=("split_gene", "first"))
            .reset_index().rename(columns={"Target gene": "gene"})
            .sort_values("n", ascending=False).to_dict("records")
        ),
    }
    return payload


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 72)
    print("PHASE 9  EXPORT DASHBOARD PAYLOAD")
    print("=" * 72)

    model = export_model()
    p = os.path.join(OUT, "model.json")
    with open(p, "w") as f:
        json.dump(model, f)
    print(f"  model.json      {os.path.getsize(p)/1e6:.2f} MB  "
          f"({len(model['ensemble'])} seeds, {model['arch']['c1']}/{model['arch']['c2']})")
    print(f"                  Tm reduction verified against DNA_NN2 / IMM1 / TMM1")

    dash = export_dashboard()
    p = os.path.join(OUT, "dashboard.json")
    with open(p, "w") as f:
        json.dump(dash, f)
    print(f"  dashboard.json  {os.path.getsize(p)/1e6:.2f} MB  "
          f"({dash['meta']['n_rows']} guides, {len(dash['validation'])} checks)")

    # Cross-check: the payload must agree with the numbers the README reports.
    tbl = {(r["protocol"], r["model"]): r["spearman_per_gene_mean"] for r in dash["benchmark"]["table"]}
    for key, expect in [
        (("leave-one-gene-out", "Rule Set 2 GBT (Azimuth params)"), 0.5148),
        (("leave-one-gene-out", "CNN + attention"), 0.5073),
    ]:
        got = tbl[key]
        assert abs(got - expect) < 5e-4, f"{key}: payload {got:.4f} != reported {expect}"
    print(f"  cross-check     benchmark values match the reported table")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
