"""Inspect the Azimuth source data before trusting any derived artifact.

Checks:
  1. Structure of the shipped FC_plus_RES_withPredictions.csv (the V3 dataset).
  2. Spearman of the shipped Azimuth `predictions` column vs the target, which is
     the in-repo ground truth for what Rule Set 2 achieves on this data.
  3. Whether V1_data.xlsx / V2_data.xlsx can be re-read in Python 3 so the CSV can
     be independently reconstructed rather than taken on faith.
"""

import numpy as np
import pandas as pd
from scipy import stats

SRC = "_azimuth_src/azimuth/data"

print("=" * 70)
print("1. FC_plus_RES_withPredictions.csv")
print("=" * 70)
df = pd.read_csv(f"{SRC}/FC_plus_RES_withPredictions.csv", index_col=0)
print(f"shape: {df.shape}")
print(f"columns: {list(df.columns)}")
print(f"\ndtypes:\n{df.dtypes}")
print(f"\nnulls:\n{df.isnull().sum()}")

print(f"\n30mer lengths: {sorted(df['30mer'].str.len().unique())}")
print(f"alphabet: {sorted(set(''.join(df['30mer'])))}")
print(f"unique 30mers: {df['30mer'].nunique()}  (rows: {len(df)})")
print(f"\ndrugs:\n{df['drug'].value_counts()}")
print(f"\nn genes: {df['Target gene'].nunique()}")
print(f"guides per gene:\n{df['Target gene'].value_counts()}")

print(f"\ntarget `score_drug_gene_rank`:\n{df['score_drug_gene_rank'].describe()}")
print(f"\n`score_drug_gene_threshold` values: {df['score_drug_gene_threshold'].unique()}")

# PAM audit: Azimuth's 30mer is 4nt upstream + 20nt protospacer + 3nt PAM + 3nt downstream,
# so positions 25-26 (0-indexed) should be "GG".
pam = df["30mer"].str[25:27]
print(f"\nPAM positions [25:27] value counts:\n{pam.value_counts().head()}")
print(f"fraction with GG at [25:27]: {(pam == 'GG').mean():.4f}")

print("\n" + "=" * 70)
print("2. Shipped Azimuth (Rule Set 2) predictions vs measured target")
print("=" * 70)
rho, p = stats.spearmanr(df["predictions"], df["score_drug_gene_rank"])
print(f"OVERALL Spearman (all {len(df)} rows): {rho:.4f}  (p={p:.3e})")

# The paper evaluates per-gene with leave-one-gene-out, reporting mean +/- sd across genes.
per_gene = []
for gene, g in df.groupby("Target gene"):
    if len(g) < 10:
        continue
    r, _ = stats.spearmanr(g["predictions"], g["score_drug_gene_rank"])
    per_gene.append((gene, len(g), r))
pg = pd.DataFrame(per_gene, columns=["gene", "n", "spearman"]).sort_values("spearman")
print(f"\nPER-GENE Spearman (how the paper reports it, Fig 4c):\n{pg.to_string(index=False)}")
print(f"\nmean across genes: {pg['spearman'].mean():.4f}  sd: {pg['spearman'].std():.4f}")

# FC vs RES split: RES guides carry a real drug label, FC guides are 'nodrug'.
fc = df[df["drug"] == "nodrug"]
res = df[df["drug"] != "nodrug"]
print(f"\nFC subset (drug=='nodrug'): n={len(fc)}, genes={fc['Target gene'].nunique()}")
print(f"RES subset (drug!='nodrug'): n={len(res)}, genes={res['Target gene'].nunique()}")
for name, sub in [("FC", fc), ("RES", res)]:
    r, _ = stats.spearmanr(sub["predictions"], sub["score_drug_gene_rank"])
    print(f"  {name} overall Spearman: {r:.4f}")

print("\n" + "=" * 70)
print("3. Can we reconstruct from raw xlsx? (don't trust the derived CSV blindly)")
print("=" * 70)
try:
    v1h = pd.read_excel(f"{SRC}/V1_data.xlsx", sheet_name=0, index_col=[0, 1])
    v1m = pd.read_excel(f"{SRC}/V1_data.xlsx", sheet_name=1, index_col=[0, 1])
    print(f"V1 human sheet: {v1h.shape}, cols={list(v1h.columns)}")
    print(f"V1 mouse sheet: {v1m.shape}, cols={list(v1m.columns)}")
    print(f"V1 human index names: {v1h.index.names}")
except Exception as e:
    print(f"V1 read failed: {type(e).__name__}: {e}")

try:
    v2 = pd.read_excel(
        f"{SRC}/V2_data.xlsx", sheet_name="ResultsFiltered",
        skiprows=range(0, 7), index_col=[0, 4],
    )
    print(f"\nV2 ResultsFiltered: {v2.shape}")
    print(f"V2 index names: {v2.index.names}")
    print(f"V2 cols: {list(v2.columns)}")
except Exception as e:
    print(f"V2 read failed: {type(e).__name__}: {e}")
