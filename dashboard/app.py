"""Streamlit dashboard for the CRISPR on-target efficiency project.

Run with:  streamlit run dashboard/app.py

Reads dashboard/data/dashboard.json for the reported numbers (written by
scripts/09_export_dashboard.py) and loads the trained CNN ensemble directly for
the live scorer, so what the widget predicts is the same model whose test number
appears in the benchmark table.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crispr import scoring as S

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dashboard.json")

POSITION_LABELS = (
    ["-4", "-3", "-2", "-1"] + [str(i) for i in range(1, 21)]
    + ["N", "G", "G", "+1", "+2", "+3"]
)
REGION_OF = (["5' context"] * 4 + ["protospacer"] * 16 + ["seed"] * 4
             + ["PAM"] * 3 + ["3' context"] * 3)

INK = "#111418"
MUTED = "#6b7280"
ACCENT = "#2563eb"
WARM = "#d97706"
GOOD = "#059669"
BAD = "#dc2626"
GRID = "#e5e7eb"


# ---------------------------------------------------------------- plumbing

@st.cache_data(show_spinner=False)
def load():
    with open(DATA) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading the trained ensemble...")
def warm_model():
    """Touch the bundle once so the first score is not the slow one."""
    S.conformal_quantiles()
    return S.default_gene_position()


def base_layout(fig, height=380, xtitle="", ytitle="", legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=24, t=16, b=52),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color=INK),
        showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=xtitle, showgrid=False, linecolor=GRID,
                     ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def caption(text):
    st.markdown(
        f"<p style='color:{MUTED};font-size:12.5px;margin-top:-6px'>{text}</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- sections

def section_benchmark(d):
    st.subheader("Does the CNN beat the gradient-boosted-tree baseline?")
    st.markdown(
        "**No — it ties under the paper's protocol and wins on the frozen test split, "
        "and neither result beats a tuned LightGBM on the same hand-crafted features.** "
        "On roughly 4,400 guides, a learned sequence representation buys nothing over "
        "well-chosen k-mer features."
    )

    tbl = pd.DataFrame(d["benchmark"]["table"])
    short = {
        "Rule Set 2 GBT (Azimuth params)": "Rule Set 2 GBT",
        "LightGBM (Rule Set 2 features)": "LightGBM",
        "CNN + attention": "CNN + attention",
    }
    tbl["m"] = tbl["model"].map(lambda m: short.get(m, m))
    protocols = ["leave-one-gene-out", "gene-held-out test"]
    sub = tbl[tbl.protocol.isin(protocols)]

    fig = go.Figure()
    for name, color in [("Rule Set 2 GBT", MUTED), ("LightGBM", WARM), ("CNN + attention", ACCENT)]:
        rows = [sub[(sub.protocol == p) & (sub.m == name)] for p in protocols]
        vals = [float(r["spearman_per_gene_mean"].iloc[0]) if len(r) else None for r in rows]
        errs = [float(r["spearman_per_gene_sd"].iloc[0]) if len(r) else None for r in rows]
        fig.add_bar(
            name=name, x=["Leave-one-gene-out", "Gene-held-out test"], y=vals,
            marker_color=color,
            error_y=dict(type="data", array=errs, color=MUTED, thickness=1, width=5),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            hovertemplate="%{x}<br>" + name + ": %{y:.4f}<extra></extra>",
        )
    for y, lab in [(0.4, "Doench Fig. 4c lower"), (0.6, "Doench Fig. 4c upper")]:
        fig.add_hline(y=y, line=dict(color=GRID, width=1, dash="dash"),
                      annotation_text=lab, annotation_font_size=10,
                      annotation_font_color=MUTED, annotation_position="right")
    fig.update_layout(barmode="group", yaxis_range=[0, 0.72])
    st.plotly_chart(base_layout(fig, 420, "Evaluation protocol",
                                "Mean per-gene Spearman"), use_container_width=True)
    caption(
        "Error bars are ±1 SD across the 18 (gene, drug) evaluation groups. That spread "
        "(~0.08) is larger than every gap between models, which is the whole story."
    )

    st.markdown("##### Are the differences real?")
    comp = pd.DataFrame(d["benchmark"]["comparisons"])
    show = pd.DataFrame({
        "Comparison": comp["model_a"] + " − " + comp["model_b"],
        "Mean difference": comp["mean_diff"].map("{:+.4f}".format),
        "95% bootstrap CI": [f"[{lo:.3f}, {hi:.3f}]"
                             for lo, hi in zip(comp["ci95_low"], comp["ci95_high"])],
        "Wilcoxon p": comp["wilcoxon_p"].map("{:.3f}".format),
        "Verdict": np.where(comp["significant"], "Consistent sign, small effect",
                            "Tie — CI spans zero"),
    })
    st.dataframe(show, hide_index=True, width="stretch")
    caption(
        "Paired across the 18 evaluation groups; 10,000-sample bootstrap on the mean "
        "difference. For LightGBM vs Rule Set 2 the two tests disagree — the sign is "
        "consistent across genes (p = 0.006) while the CI on the mean spans zero, which "
        "means reliably a little better rather than decisively better."
    )

    st.markdown("##### Per-gene difference, CNN minus Rule Set 2 GBT")
    pg = pd.DataFrame(d["per_gene"])
    pg["label"] = pg.apply(
        lambda r: f"{r['gene']} ({r['drug']})" if r["gene"] == "MED12" else r["gene"], axis=1
    )
    pg["delta"] = pg["spearman_cnn"] - pg["spearman_gbt"]
    pg = pg.sort_values("delta")
    fig = go.Figure(go.Bar(
        x=pg["delta"], y=pg["label"], orientation="h",
        marker_color=[GOOD if v > 0 else BAD for v in pg["delta"]],
        hovertemplate="%{y}<br>CNN − GBT: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    st.plotly_chart(base_layout(fig, 440, "Difference in Spearman (CNN − GBT)",
                                "", legend=False), use_container_width=True)
    caption(
        f"Out-of-fold leave-one-gene-out predictions. The CNN wins on "
        f"{int((pg.delta > 0).sum())} of {len(pg)} groups and loses on "
        f"{int((pg.delta < 0).sum())} — mixed signs are what a statistical tie looks like."
    )


def section_scorer(d):
    st.subheader("Score a guide")
    defaults = warm_model()

    st.markdown(
        "Runs the **gene-held-out ensemble** — the same three-seed model behind the "
        "0.495 test number, not an in-sample refit — and wraps it in a split conformal "
        "interval calibrated on held-out genes."
    )

    guides = pd.DataFrame(d["guides"])
    examples = {
        "Custom": "",
        "Strong guide (measured top 5%)": guides.nlargest(200, "y").iloc[7]["seq"],
        "Weak guide (measured bottom 5%)": guides.nsmallest(200, "y").iloc[7]["seq"],
        "Guide the model gets wrong": guides.assign(e=(guides.gh - guides.y).abs())
                                            .nlargest(30, "e").iloc[3]["seq"],
    }

    col_a, col_b = st.columns([3, 2])
    with col_a:
        pick = st.selectbox("Start from", list(examples), index=1)
        seq = st.text_input(
            "30nt context sequence", value=examples[pick],
            help="4nt upstream + 20nt protospacer + 3nt NGG PAM + 3nt downstream.",
        ).strip().upper()
    with col_b:
        alpha = st.select_slider(
            "Interval confidence", options=[0.80, 0.90, 0.95], value=0.90,
            format_func=lambda v: f"{int(v*100)}%",
        )
        use_defaults = st.checkbox(
            "Use median gene position", value=True,
            help="The model consumes three gene-position features that a bare 30mer "
                 "cannot supply. Unchecking lets you set them explicitly.",
        )

    if use_defaults:
        pp, cut = defaults["percent_peptide"], defaults["aa_cut_position"]
    else:
        c1, c2 = st.columns(2)
        pp = c1.number_input("Percent peptide", 0.0, 100.0, float(defaults["percent_peptide"]), 0.5)
        cut = c2.number_input("Amino-acid cut position", 0.0, 5000.0,
                              float(defaults["aa_cut_position"]), 1.0)

    if not seq:
        st.info("Paste a 30nt sequence or pick an example above.")
        return

    try:
        out = S.score(seq, pp, cut, alpha=round(1 - alpha, 2))
    except S.GuideSequenceError as e:
        st.error(str(e))
        return

    lo, hi = out["interval"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted efficiency", f"{out['prediction']:.3f}",
              help="Rank within a (gene, drug) group scaled to (0, 1]. 0.8 means "
                   "predicted to outrank ~80% of guides against the same gene.")
    m2.metric(f"{int(alpha*100)}% conformal interval", f"{lo:.2f} – {hi:.2f}",
              f"±{out['half_width']:.3f}", delta_color="off")
    m3.metric("Ensemble disagreement", f"±{out['seed_spread']:.4f}",
              help="Spread across the three training seeds. This is model variance, "
                   "not the calibrated interval.")

    match = guides[guides.seq == out["sequence"]]
    if len(match):
        r = match.iloc[0]
        st.success(
            f"This guide is in the dataset: **{r['gene']}** ({r['drug']}), measured "
            f"activity **{r['y']:.3f}**, ranked **{int(r['rank'])} of {int(r['group_n'])}** "
            f"in its group. It was in the **{r['split']}** split."
        )
    else:
        st.info("Not in the training dataset — this is a genuine out-of-sample prediction.")

    if out["half_width"] > 0.35:
        st.warning(
            f"The interval spans {hi-lo:.2f} of a (0, 1] scale, which is most of the range. "
            "These intervals are calibrated, but wide enough to be useful mainly for "
            "flagging predictions not to trust rather than for ranking individual guides."
        )

    st.markdown("##### Where the model looked")
    s = out["sequence"]
    colors = {"5' context": "#cbd5e1", "protospacer": ACCENT, "seed": WARM,
              "PAM": GOOD, "3' context": "#cbd5e1"}
    fig = go.Figure(go.Bar(
        x=list(range(30)), y=out["attention"],
        marker_color=[colors[r] for r in REGION_OF],
        customdata=list(zip(s, POSITION_LABELS, REGION_OF)),
        hovertemplate="%{customdata[0]} at position %{customdata[1]}"
                      "<br>%{customdata[2]}<br>weight %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=1 / 30, line=dict(color=BAD, width=1, dash="dash"),
                  annotation_text="uniform 1/30", annotation_font_size=10,
                  annotation_font_color=BAD)
    fig.update_xaxes(tickmode="array", tickvals=list(range(30)), tickfont_size=9,
                     ticktext=[f"{b}<br>{p}" for b, p in zip(s, POSITION_LABELS)])
    st.plotly_chart(base_layout(fig, 300, "Base and position in the 30mer",
                                "Attention-pooling weight", legend=False),
                    use_container_width=True)
    caption(
        f"GC count in the protospacer: {out['gc_count']}/20. Orange marks the seed "
        "(protospacer 17–20), green the PAM. Averaged over the three ensemble seeds."
    )


def section_explorer(d):
    st.subheader("Guide explorer")
    st.markdown(
        "Every guide in the dataset with its measured activity and all three models' "
        "**out-of-fold** leave-one-gene-out predictions — each one made by a model that "
        "never saw that guide's gene during training."
    )

    g = pd.DataFrame(d["guides"])
    c1, c2, c3 = st.columns([2, 2, 3])
    genes = c1.multiselect("Gene", sorted(g.gene.unique()))
    splits = c2.multiselect("Gene-held-out split", ["train", "cal", "test"])
    query = c3.text_input("Sequence contains", placeholder="e.g. GGGTTT or a full 30mer")

    f = g
    if genes:
        f = f[f.gene.isin(genes)]
    if splits:
        f = f[f.split.isin(splits)]
    if query:
        f = f[f.seq.str.contains(query.strip().upper(), regex=False)]

    st.caption(f"{len(f):,} of {len(g):,} guides")
    if not len(f):
        return

    f = f.assign(err=(f.cnn - f.y).abs()).sort_values("err")
    show = f.head(400)[["seq", "gene", "drug", "y", "gbt", "lgbm", "cnn", "split", "rank", "group_n"]]
    st.dataframe(
        show.rename(columns={
            "seq": "30mer", "gene": "Gene", "drug": "Drug", "y": "Measured",
            "gbt": "Rule Set 2 GBT", "lgbm": "LightGBM", "cnn": "CNN",
            "split": "Split", "rank": "Rank", "group_n": "of",
        }),
        hide_index=True, width="stretch", height=340,
    )
    caption("Sorted by CNN absolute error, closest first. Showing up to 400 rows.")

    st.markdown("##### Measured versus predicted")
    which = st.radio("Model", ["CNN", "Rule Set 2 GBT", "LightGBM"], horizontal=True)
    col = {"CNN": "cnn", "Rule Set 2 GBT": "gbt", "LightGBM": "lgbm"}[which]
    samp = f.sample(min(len(f), 3000), random_state=0)
    fig = go.Figure(go.Scattergl(
        x=samp["y"], y=samp[col], mode="markers",
        marker=dict(size=4, color=ACCENT, opacity=0.35),
        customdata=samp[["gene", "seq"]].values,
        hovertemplate="%{customdata[0]}<br>%{customdata[1]}"
                      "<br>measured %{x:.3f}, predicted %{y:.3f}<extra></extra>",
    ))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=MUTED, width=1, dash="dash"))
    st.plotly_chart(base_layout(fig, 420, "Measured activity (rank, 0–1)",
                                f"{which} out-of-fold prediction", legend=False),
                    use_container_width=True)
    caption(
        f"Spearman on this selection: "
        f"{f['y'].corr(f[col], method='spearman'):.3f}. Dashed line is y = x. "
        "Predictions are compressed toward the middle because the models are fit with "
        "squared error on a rank target."
    )


def section_attention(d):
    st.subheader("What the model learned to look at")
    a = pd.DataFrame(d["attention_by_position"])
    interp = d["interpretability"]

    metric = st.radio(
        "Attribution source", ["CNN attention", "GBT importance (independent control)"],
        horizontal=True,
    )
    is_cnn = metric.startswith("CNN")
    col = "attention_weight" if is_cnn else "gbt_positional_importance"
    colors = {"5' context": "#cbd5e1", "protospacer": ACCENT, "seed": WARM,
              "PAM": GOOD, "3' context": "#cbd5e1"}

    fig = go.Figure(go.Bar(
        x=list(range(30)), y=a[col],
        marker_color=[colors[r] for r in a["region"]],
        customdata=a[["position_label", "region"]].values,
        hovertemplate="position %{customdata[0]} (%{customdata[1]})"
                      "<br>%{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=1 / 30, line=dict(color=BAD, width=1, dash="dash"),
                  annotation_text="uniform 1/30", annotation_font_size=10,
                  annotation_font_color=BAD)
    fig.update_xaxes(tickmode="array", tickvals=list(range(30)),
                     ticktext=a["position_label"], tickfont_size=10)
    st.plotly_chart(
        base_layout(fig, 340, "Position in the 30mer",
                    "Attention weight" if is_cnn else "Normalised Gini importance",
                    legend=False),
        use_container_width=True,
    )
    caption(
        "Mean over 1,066 held-out guides. Orange is the seed (protospacer 17–20), "
        "green the PAM, blue the rest of the protospacer."
        if is_cnn else
        "Order-1 position-dependent nucleotide features, aggregated per position and "
        "normalised to sum to 1."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Weight by region")
        rows = a.groupby("region")[col].mean().reindex(
            ["PAM", "seed", "protospacer", "3' context", "5' context"]
        )
        st.dataframe(
            pd.DataFrame({
                "Region": ["PAM (NGG)", "Seed (protospacer 17–20)",
                           "Rest of protospacer (1–16)", "3′ context", "5′ context"],
                "Mean weight": rows.map("{:.4f}".format).values,
                "vs uniform": (rows / (1 / 30)).map("{:.2f}×".format).values,
            }),
            hide_index=True, width="stretch",
        )
    with c2:
        st.markdown("##### The seed-region prior holds")
        st.markdown(
            f"Attention rises along the protospacer toward the PAM and peaks at the "
            f"PAM's variable N base — consistent with Cas9 requiring PAM recognition "
            f"before it interrogates the protospacer.\n\n"
            f"Seed versus the rest of the protospacer, paired across guides: "
            f"**Wilcoxon p ≈ {interp['wilcoxon_p']:.0e}**."
        )
        st.success(
            "**Independent control agrees.** Attention weights are easy to over-read, so "
            "the same question went to the GBT, which needs no attention mechanism to be "
            f"interpretable. It places **{interp['gbt_seed_importance_share']:.1%}** of its "
            "positional importance in the four seed positions against 13.3% if it were "
            "flat. Two methods with nothing in common agree on where the signal is."
        )


def section_calibration(d):
    st.subheader("Do the uncertainty intervals mean anything?")
    st.markdown(
        "**Yes on exchangeable data, and conservatively on new genes.** Split conformal "
        "with absolute-residual scores and the finite-sample ⌈(n+1)(1−α)⌉ quantile."
    )
    conf = d["conformal"]

    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                    line=dict(color=MUTED, width=1, dash="dash"))
    for key, name, color in [
        ("RANDOM split (exchangeable)", "Random split (exchangeable)", ACCENT),
        ("GENE-HELD split (shifted)", "Gene-held-out split (shifted)", BAD),
    ]:
        c = pd.DataFrame(conf[key]["curve"])
        fig.add_scatter(x=c["nominal_coverage"], y=c["empirical_coverage"], mode="lines+markers",
                        name=name, line=dict(color=color, width=2), marker=dict(size=5),
                        hovertemplate=name + "<br>nominal %{x:.2f} → empirical %{y:.3f}<extra></extra>")
    st.plotly_chart(base_layout(fig, 420, "Nominal coverage (1 − α)",
                                "Empirical coverage on held-out guides"),
                    use_container_width=True)
    caption(
        "Above the dashed line is conservative (intervals wider than needed); below is "
        "under-covering. Source: CNN + attention on "
        f"{conf['RANDOM split (exchangeable)']['n_cal']:,} / "
        f"{conf['GENE-HELD split (shifted)']['n_cal']:,} calibration guides."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Coverage and width at key levels")
        rows = []
        for lvl in (0.95, 0.90, 0.80, 0.50):
            r = next(x for x in conf["RANDOM split (exchangeable)"]["curve"]
                     if abs(x["nominal_coverage"] - lvl) < 1e-6)
            gq = next(x for x in conf["GENE-HELD split (shifted)"]["curve"]
                      if abs(x["nominal_coverage"] - lvl) < 1e-6)
            rows.append({
                "Nominal": f"{lvl:.0%}",
                "Random": f"{r['empirical_coverage']:.1%}",
                "Gene-held": f"{gq['empirical_coverage']:.1%}",
                "Half-width (random)": f"±{r['mean_width']/2:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with c2:
        st.markdown("##### Interval width grows with confidence")
        fig = go.Figure()
        for key, name, color in [
            ("RANDOM split (exchangeable)", "Random split", ACCENT),
            ("GENE-HELD split (shifted)", "Gene-held-out split", BAD),
        ]:
            c = pd.DataFrame(conf[key]["curve"])
            fig.add_scatter(x=c["nominal_coverage"], y=c["mean_width"], mode="lines",
                            name=name, line=dict(color=color, width=2))
        st.plotly_chart(base_layout(fig, 260, "Nominal coverage (1 − α)",
                                    "Mean interval width"), use_container_width=True)

    st.warning(
        "**The guarantee does not formally hold under gene shift.** Conformal requires "
        "calibration and test data to be exchangeable. Under the random split they are, "
        "and coverage tracks nominal almost exactly — that validates the implementation. "
        "Under gene-held-out the genes are disjoint, so the guarantee lapses; empirically "
        "it lands conservative (93.4% at nominal 90%), which is the safe direction but is "
        "luck rather than a theorem. Separately, a ±0.42 interval on a (0, 1] target is "
        "wide enough that these are useful for flagging low-confidence predictions, not "
        "for ranking individual guides."
    )


def section_validation(d):
    v = pd.DataFrame(d["validation"])
    n, passed = len(v), int(v["passed"].sum())
    st.subheader(f"Adversarial validation: {passed} of {n} checks passing")
    st.markdown(
        "Each check is written so it would actually fail if the corresponding bug were "
        "present, rather than restating what the code does. Run with "
        "`python scripts/08_validate.py`; it exits non-zero on any failure."
    )

    if passed == n:
        st.success(f"All {n} checks pass.")
    else:
        st.error(f"{n - passed} checks failing.")

    show = v.copy()
    show["Status"] = np.where(show["passed"], "pass", "FAIL")
    st.dataframe(
        show[["Status", "check", "detail"]].rename(
            columns={"check": "Check", "detail": "Observed"}),
        hide_index=True, width="stretch", height=420,
    )

    st.warning(
        "**Two genuine findings, neither of which changes a conclusion.**\n\n"
        "*The random split leaks duplicate sequences, not just gene context.* 230 test "
        "rows share a 30mer with a training row, because MED12 was screened under two "
        "drugs. Re-running conformal on only the test rows whose sequence the model never "
        "saw gives 89.2% coverage at 90% nominal, against 90.2% overall.\n\n"
        "*Melting temperature on short segments is not physically meaningful.* Rule Set 2 "
        "computes Tm on 5nt and 8nt sub-segments, below the range where nearest-neighbour "
        "thermodynamics is valid, so the 5-mer values come out negative. This is inherited "
        "from the original design, not introduced here."
    )


def section_data(d):
    m = d["meta"]
    st.subheader("Dataset and splits")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Guide × drug rows", f"{m['n_rows']:,}")
    c2.metric("Unique 30mers", f"{m['n_unique_guides']:,}")
    c3.metric("Genes", m["n_genes"])
    c4.metric("Calibration guides", f"{m['n_calibration']:,}")
    caption(
        f"{m['dataset']}. Target is `{m['target']}` — {m['target_note']}."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Rows per gene, and where each gene went")
        gs = pd.DataFrame(d["gene_summary"])
        cmap = {"train": ACCENT, "cal": WARM, "test": GOOD}
        fig = go.Figure()
        for sp in ["train", "cal", "test"]:
            s = gs[gs.split == sp]
            fig.add_bar(x=s["n"], y=s["gene"], orientation="h", name=sp,
                        marker_color=cmap[sp],
                        hovertemplate="%{y}: %{x} rows (" + sp + ")<extra></extra>")
        fig.update_layout(barmode="stack")
        st.plotly_chart(base_layout(fig, 420, "Guide × drug rows", ""),
                        use_container_width=True)
        caption(
            "MED12 alone is 35% of all rows, so genes are assigned largest-first to "
            "whichever split is furthest below its quota rather than at random."
        )
    with c2:
        st.markdown("##### Why three protocols")
        st.markdown(
            "**Leave-one-gene-out (primary).** Azimuth's own default and how Doench 2016 "
            "Fig. 4 reports Spearman. This is the paper-comparable number.\n\n"
            "**Frozen gene-held-out 60/20/20.** Whole genes to train / calibration / test. "
            "Needed for conformal, which requires a disjoint calibration set.\n\n"
            "**Random row-level.** Reported only as a leakage contrast — it inflates the "
            "GBT from 0.454 to 0.544, which is roughly the size of the entire effect "
            "being measured."
        )
        st.success(
            "**Verified against the raw source, not assumed.** The target was independently "
            "re-derived from `V1_data.xlsx` and `V2_data.xlsx` by porting Azimuth's "
            "Python 2 rank-transform logic. Maximum absolute difference across all 5,310 "
            "rows: **5 × 10⁻¹⁰**."
        )

    st.markdown("##### Chromatin accessibility channel: cut, with evidence")
    ch = d["chromatin"]
    st.dataframe(
        pd.DataFrame([
            {"Cell line": "A375 (RES half)", "Rows": "3,473",
             "ENCODE ATAC-seq / DNase-seq": "0 experiments — only RNA-based data"},
            {"Cell line": "NB4 + TF1 + MOLM-13 (FC, human)", "Rows": "882",
             "ENCODE ATAC-seq / DNase-seq": "TF1 none; NB4 1; MOLM-13 3"},
            {"Cell line": "Mouse lines (FC)", "Rows": "955",
             "ENCODE ATAC-seq / DNase-seq": "different genome entirely"},
        ]),
        hide_index=True, width="stretch",
    )
    st.error(
        "**Three independent blockers, any one sufficient.** A375 generated 65% of the "
        "rows and has zero chromatin accessibility experiments on ENCODE. The remaining "
        "rows span three more human lines plus mouse, so a per-row match would need "
        "several tracks across two genomes. And the Azimuth release carries no genomic "
        "coordinates — only Ensembl transcript IDs and transcript-relative cut positions "
        "— so there is nothing to query a bigWig with."
    )
    with st.expander("Raw ENCODE query results"):
        st.json(ch)


def section_selection(d):
    st.subheader("How the architecture was chosen")
    sw = pd.DataFrame(d["sweep"]["results"]).drop_duplicates("config")
    sw = sw.sort_values("mean_spearman")
    sel = d["sweep"]["selected"]["config"]
    gbt_ref = d["sweep"]["gbt_reference_same_protocol"]

    fig = go.Figure(go.Bar(
        x=sw["mean_spearman"], y=sw["config"], orientation="h",
        marker_color=[ACCENT if c == sel else "#cbd5e1" for c in sw["config"]],
        hovertemplate="%{y}<br>inner-CV Spearman %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=gbt_ref, line=dict(color=WARM, width=1.5, dash="dash"),
                  annotation_text="GBT, same protocol", annotation_font_size=10,
                  annotation_font_color=WARM)
    st.plotly_chart(base_layout(fig, 520, "Mean Spearman over 6 held-out training genes",
                                "", legend=False), use_container_width=True)
    caption(
        f"{len(sw)} configurations, scored by leave-one-gene-out restricted to the "
        "training genes — the test and calibration genes were never loaded during the "
        f"sweep, so the reported test numbers stay honest. Selected: **{sel}** (blue)."
    )

    c1, c2 = st.columns(2)
    c1.success(
        "**What worked.**\n\n"
        "*Feature-map dropout (+0.055).* The literal spec architecture reached 0.99 "
        "training Spearman against 0.21 held-out. `Dropout1d` was the single most "
        "effective fix.\n\n"
        "*Learned positional embedding (+0.010).* Convolutions are translation-"
        "equivariant, so without one the network cannot tell position 4 from position 20 "
        "— yet Rule Set 2 draws 58% of its Gini importance from position-specific "
        "nucleotide identity."
    )
    c2.error(
        "**What did not work.**\n\n"
        "*Dropping gene-position features (0.377 → 0.179).* The hypothesis was that "
        "percent-peptide and cut position are gene-level shortcuts that cannot transfer "
        "across genes. Measurement said the opposite; they were kept.\n\n"
        "*Narrowing the network.* Every narrower variant scored below the 64/128 stack. "
        "Capacity was not the problem; regularisation was."
    )
    st.info(
        f"**Why {d['sweep']['selected']['mean_spearman']:.3f} here did not mean abandoning "
        "the model.** The inner protocol trains on 5 genes instead of 16, so everything "
        f"scores lower. Re-measuring the GBT on the same protocol gave {gbt_ref:.3f} rather "
        "than its 0.515 leave-one-gene-out figure, which established the correct bar. "
        "Given the full protocol, the CNN reached 0.507."
    )


# ---------------------------------------------------------------- shell

SECTIONS = {
    "Benchmark": section_benchmark,
    "Score a guide": section_scorer,
    "Guide explorer": section_explorer,
    "Attention": section_attention,
    "Calibration": section_calibration,
    "Architecture selection": section_selection,
    "Validation": section_validation,
    "Data & splits": section_data,
}


def main():
    st.set_page_config(page_title="CRISPR On-Target Efficiency",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """<style>
        .block-container {padding-top: 2.2rem; max-width: 1320px;}
        h1, h2, h3 {letter-spacing: -0.015em;}
        [data-testid="stMetricValue"] {font-size: 1.7rem;}
        </style>""",
        unsafe_allow_html=True,
    )

    d = load()
    with st.sidebar:
        st.markdown("### CRISPR On-Target Efficiency")
        st.caption(
            "CNN + attention benchmarked against a reproduced Rule Set 2 "
            "gradient-boosted-tree baseline, with attention interpretability and "
            "conformal prediction intervals."
        )
        choice = st.radio("Section", list(SECTIONS), label_visibility="collapsed")
        st.divider()
        st.metric("Baseline, LOGO Spearman", "0.515")
        st.metric("CNN, LOGO Spearman", "0.507")
        st.metric("CNN, gene-held-out test", "0.495")
        st.caption(
            f"Doench et al. 2016 · {d['meta']['n_rows']:,} guide × drug rows · "
            f"{d['meta']['n_genes']} genes · {len(d['validation'])} validation checks"
        )

    st.title("CRISPR On-Target Efficiency")
    SECTIONS[choice](d)


if __name__ == "__main__":
    main()
