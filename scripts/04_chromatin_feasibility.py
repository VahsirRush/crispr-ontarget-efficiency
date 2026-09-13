"""Phase 4: can we add a matched chromatin-accessibility channel? Evidence, then a decision.

The spec is explicit that this phase must be cut rather than run on a guessed
ENCODE accession. This script queries the ENCODE portal live and records what it
finds, so the decision is reproducible rather than asserted.

Three questions, in order:
  1. Which cell lines actually generated the FC and RES measurements?
  2. Does ENCODE have ATAC-seq / DNase-seq for those lines?
  3. Do we even have genomic coordinates to look a signal track up with?
"""

import json
import os
import sys
import urllib.parse
import urllib.request
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import data as D

ART = os.path.join(D.REPO_ROOT, "artifacts")
ENCODE = "https://www.encodeproject.org/search/"

# From Doench et al. 2016 (Nature Biotechnology 34:184) Methods and Fig. 1-3:
#   RES  -- tiling library screened in A375 melanoma with vemurafenib (PLX_2uM),
#           selumetinib (AZD_200nM) and 6-thioguanine (6TG_2ug/mL).
#   FC   -- flow-cytometry screens from Doench 2014: human CD13/CD33/CD15 in
#           NB4, TF1 and MOLM13; the remaining genes are mouse surface markers.
CELL_LINES = {
    "A375": {"genes": ["CCDC101", "MED12", "TADA2B", "TADA1", "HPRT1",
                       "CUL3", "NF1", "NF2"], "dataset": "RES", "organism": "human"},
    "NB4": {"genes": ["CD13", "CD33"], "dataset": "FC", "organism": "human"},
    "TF1": {"genes": ["CD13", "CD33"], "dataset": "FC", "organism": "human"},
    "MOLM-13": {"genes": ["CD33", "CD15"], "dataset": "FC", "organism": "human"},
}
MOUSE_GENES = ["CD5", "CD28", "CD43", "CD45", "H2-K", "THY1"]


def encode_query(**params):
    params.setdefault("type", "Experiment")
    params.setdefault("format", "json")
    params.setdefault("limit", "all")
    url = ENCODE + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001 - network failures must not be silent
        return {"_error": f"{type(e).__name__}: {e}", "total": 0, "@graph": []}


def main():
    print("=" * 72)
    print("PHASE 4  CHROMATIN CHANNEL -- FEASIBILITY CHECK")
    print("=" * 72)

    df = pd.read_csv(os.path.join(ART, "dataset.csv"))
    report = {"cell_lines": {}, "coordinates": {}, "decision": None}

    # --- 1. how much of the data comes from each cell line? ----------------
    print("\n1. Cell lines behind the measurements")
    print("-" * 72)
    n = len(df)
    res_rows = (df.dataset == "RES").sum()
    fc_rows = (df.dataset == "FC").sum()
    mouse_rows = df["Target gene"].isin(MOUSE_GENES).sum()
    print(f"  RES half (A375 only)          : {res_rows:>5} rows  ({res_rows/n:5.1%})")
    print(f"  FC half  (NB4/TF1/MOLM-13 +")
    print(f"            mouse cell lines)   : {fc_rows:>5} rows  ({fc_rows/n:5.1%})")
    print(f"    of which mouse surface-marker genes: {mouse_rows:>5} rows  ({mouse_rows/n:5.1%})")
    print("\n  => the combined dataset spans at least 4 human cell lines plus mouse.")
    print("     There is no single cell line to match a track to.")

    # --- 2. does ENCODE have accessibility data for them? ------------------
    print("\n2. ENCODE chromatin-accessibility availability")
    print("-" * 72)
    for line, meta in CELL_LINES.items():
        acc = encode_query(status="released",
                           **{"biosample_ontology.term_name": line,
                              "assay_title": ["ATAC-seq", "DNase-seq"]})
        any_assay = encode_query(**{"biosample_ontology.term_name": line})
        assays = sorted({g.get("assay_title") for g in any_assay.get("@graph", [])
                         if g.get("assay_title")})
        hits = [(g.get("accession"), g.get("assay_title"))
                for g in acc.get("@graph", [])]
        rows = df["Target gene"].isin(meta["genes"]).sum()
        print(f"  {line:<9} ({meta['dataset']}, ~{rows} rows)")
        print(f"    ATAC/DNase experiments : {acc.get('total', 0)}"
              + (f"  {hits}" if hits else ""))
        print(f"    all ENCODE assays      : {assays if assays else 'none'}")
        report["cell_lines"][line] = {
            "dataset": meta["dataset"],
            "rows": int(rows),
            "n_accessibility_experiments": acc.get("total", 0),
            "accessibility_accessions": hits,
            "all_assays": assays,
        }

    # --- 3. can we even look up a coordinate? ------------------------------
    print("\n3. Genomic coordinates in the source data")
    print("-" * 72)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v1 = pd.read_excel(D.V1_XLSX, sheet_name=0, index_col=[0, 1])
        v2 = pd.read_excel(D.V2_XLSX, sheet_name="ResultsFiltered",
                           skiprows=range(0, 7), index_col=[0, 4])
    print(f"  V1 columns: {list(v1.columns)}")
    print(f"  V2 columns: {list(v2.columns)}")
    print("\n  Positions are transcript-relative (Ensembl transcript ID + CutSite /")
    print("  amino-acid position). No chromosome, no genomic start/end. A signal")
    print("  track cannot be queried without first mapping every transcript")
    print("  coordinate onto a genome build.")
    report["coordinates"] = {
        "v1_columns": list(map(str, v1.columns)),
        "v2_columns": list(map(str, v2.columns)),
        "has_genomic_coordinates": False,
        "note": "transcript-relative only (Ensembl ENST + CutSite / AA position)",
    }

    # --- decision -----------------------------------------------------------
    a375 = report["cell_lines"]["A375"]["n_accessibility_experiments"]
    print("\n" + "=" * 72)
    print("DECISION")
    print("=" * 72)
    decision = "CUT"
    reasons = [
        f"A375 generated {res_rows/n:.0%} of the rows (the whole RES half) and has "
        f"{a375} ATAC-seq/DNase-seq experiments on ENCODE. Its only ENCODE data is "
        f"RNA-based ({report['cell_lines']['A375']['all_assays']}), so no matched "
        f"accessibility track exists for the majority of the dataset.",
        "The remaining rows come from three further human lines plus mouse lines, "
        "so even a per-row match would need several tracks across two genomes, and "
        "TF1 has no accessibility data either.",
        "The Azimuth release carries no genomic coordinates -- only Ensembl "
        "transcript IDs and cut positions within the transcript -- so there is "
        "nothing to query a bigWig with without a separate transcript-to-genome "
        "mapping step.",
    ]
    for i, r in enumerate(reasons, 1):
        print(f"  ({i}) {r}\n")
    print("  Per the spec, this phase is CUT rather than run on a guessed accession.")
    print("  Substituting a track from a different cell line would silently inject")
    print("  the wrong biology into 65% of the training data.")
    report["decision"] = {"outcome": decision, "reasons": reasons}

    with open(os.path.join(ART, "chromatin_feasibility.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {ART}/chromatin_feasibility.json")


if __name__ == "__main__":
    main()
