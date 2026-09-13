"""Phase 10: build the self-contained static dashboard.

Inlines the data payload, the model weights, and the three scripts into a single
``dashboard/index.html`` that needs no server, no network, and no build step --
it opens from a file:// URL and hosts on GitHub Pages as-is.

Before writing anything it verifies the JavaScript inference path against
PyTorch. The browser scorer is a hand port of ``crispr/model.py`` plus the
feature extraction, and a silent numerical drift there would be invisible in the
UI while making every prediction on the page wrong. So the port is checked on a
sample of real guides and the build aborts if it disagrees.

Run ``scripts/09_export_dashboard.py`` first.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import data as D
from crispr import scoring as S

ROOT = D.REPO_ROOT
DASH = os.path.join(ROOT, "dashboard")
STATIC = os.path.join(DASH, "static")
DATA = os.path.join(DASH, "data")

# float32 round-trips through base64 and the JS engine accumulates in float64,
# so exact equality is not expected; anything above this is a real bug.
TOLERANCE = 1e-5


def verify_js(model_spec, n=150):
    """Score `n` real guides through node and compare against PyTorch."""
    df = pd.read_csv(os.path.join(ROOT, "artifacts", "dataset.csv"))
    rng = np.random.default_rng(20160101)
    idx = rng.choice(len(df), min(n, len(df)), replace=False)
    cases = [
        {
            "seq": df.iloc[i]["30mer"],
            "pp": float(df.iloc[i]["Percent Peptide"]),
            "cut": float(df.iloc[i]["Amino Acid Cut position"]),
        }
        for i in idx
    ]

    proc = subprocess.run(
        ["node", os.path.join(STATIC, "verify_node.js")],
        input=json.dumps({"model": model_spec, "cases": cases}),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("node inference harness failed:\n" + proc.stderr[:2000])
    js = json.loads(proc.stdout)

    d_thermo = d_pred = d_attn = 0.0
    for c, j in zip(cases, js):
        py_thermo = S.thermo_vector(c["seq"], c["pp"], c["cut"])[0]
        d_thermo = max(d_thermo, float(np.max(np.abs(np.asarray(j["thermo"], dtype=np.float32) - py_thermo))))
        out = S.score(c["seq"], c["pp"], c["cut"])
        d_pred = max(d_pred, abs(j["pred"] - out["prediction"]))
        d_attn = max(d_attn, float(np.max(np.abs(np.asarray(j["attention"]) - np.asarray(out["attention"])))))

    return {"n": len(cases), "thermo": d_thermo, "prediction": d_pred, "attention": d_attn}


def read(*parts):
    with open(os.path.join(*parts)) as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the node/PyTorch agreement check (not recommended)")
    ap.add_argument("--cases", type=int, default=150)
    args = ap.parse_args()

    print("=" * 72)
    print("PHASE 10  BUILD STATIC DASHBOARD")
    print("=" * 72)

    for name in ("dashboard.json", "model.json"):
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- run scripts/09_export_dashboard.py first")

    model_json = read(DATA, "model.json")
    dash_json = read(DATA, "dashboard.json")

    if args.skip_verify:
        print("  !! skipping the JS/PyTorch agreement check")
    else:
        print(f"  verifying browser inference against PyTorch on {args.cases} real guides...")
        r = verify_js(json.loads(model_json), args.cases)
        print(f"    thermodynamic block   max |JS - Biopython| = {r['thermo']:.2e}")
        print(f"    prediction            max |JS - PyTorch|   = {r['prediction']:.2e}")
        print(f"    attention weights     max |JS - PyTorch|   = {r['attention']:.2e}")
        worst = max(r["thermo"], r["prediction"], r["attention"])
        if worst > TOLERANCE:
            sys.exit(f"\nABORT: browser inference disagrees with PyTorch by {worst:.2e} "
                     f"(tolerance {TOLERANCE:.0e}). The port in dashboard/static/infer.js is wrong.")
        print(f"    OK -- within {TOLERANCE:.0e}, consistent with float32 round-off")

    html = (
        read(STATIC, "index.template.html")
        .replace("/*__STYLE__*/", read(STATIC, "style.css"))
        .replace("/*__DASHBOARD__*/null", dash_json)
        .replace("/*__MODEL__*/null", model_json)
        .replace("/*__INFER__*/", read(STATIC, "infer.js"))
        .replace("/*__CHARTS__*/", read(STATIC, "charts.js"))
        .replace("/*__APP__*/", read(STATIC, "app.js"))
        .replace("__BUILT__", dt.date.today().isoformat())
    )

    out = os.path.join(DASH, "index.html")
    with open(out, "w") as f:
        f.write(html)

    size = os.path.getsize(out) / 1e6
    print(f"\n  wrote {out}  ({size:.1f} MB, single file, no external requests)")
    if "__" in html.split("<footer>")[1].split("</footer>")[0]:
        sys.exit("ABORT: an unsubstituted placeholder survived into the output")
    print("  open it directly in a browser, or serve the dashboard/ directory")


if __name__ == "__main__":
    main()
