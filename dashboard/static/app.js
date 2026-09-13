/* Static dashboard logic. Reads the DASHBOARD and MODEL globals that
 * scripts/10_build_static_dashboard.py inlines above this script. */

(function () {
  "use strict";

  var D = window.DASHBOARD;
  var C = window.Charts.C;
  var model = new window.CrisprInfer.Model(window.MODEL);

  var POS = ["-4", "-3", "-2", "-1"];
  for (var i = 1; i <= 20; i++) POS.push(String(i));
  POS = POS.concat(["N", "G", "G", "+1", "+2", "+3"]);

  var REGION = [];
  ["5' context", "5' context", "5' context", "5' context"].forEach(function (r) { REGION.push(r); });
  for (i = 0; i < 16; i++) REGION.push("protospacer");
  for (i = 0; i < 4; i++) REGION.push("seed");
  for (i = 0; i < 3; i++) REGION.push("PAM");
  for (i = 0; i < 3; i++) REGION.push("3' context");

  var REGION_COLOR = {
    "5' context": C.pale, protospacer: C.accent, seed: C.warm,
    PAM: C.good, "3' context": C.pale,
  };

  var SHORT = {
    "Rule Set 2 GBT (Azimuth params)": "Rule Set 2 GBT",
    "LightGBM (Rule Set 2 features)": "LightGBM",
    "CNN + attention": "CNN + attention",
  };

  function $(sel, ctx) { return (ctx || document).querySelector(sel); }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }
  function pct(v, d) { return (v * 100).toFixed(d == null ? 1 : d) + "%"; }

  function table(headers, rows, opts) {
    opts = opts || {};
    var h = "<table><thead><tr>" + headers.map(function (x, i) {
      return "<th" + (opts.align && opts.align[i] === "n" ? " class='n'" : "") + ">" + x + "</th>";
    }).join("") + "</tr></thead><tbody>" + rows.map(function (r) {
      return "<tr>" + r.map(function (c, i) {
        var cls = [];
        if (opts.align && opts.align[i] === "n") cls.push("n");
        if (opts.align && opts.align[i] === "m") cls.push("mono");
        return "<td" + (cls.length ? " class='" + cls.join(" ") + "'" : "") + ">" + c + "</td>";
      }).join("") + "</tr>";
    }).join("") + "</tbody></table>";
    return opts.scroll ? "<div class='scroll'>" + h + "</div>" : h;
  }

  // ------------------------------------------------------------ benchmark

  function benchmark(root) {
    var protocols = ["leave-one-gene-out", "gene-held-out test"];
    var names = ["Rule Set 2 GBT", "LightGBM", "CNN + attention"];
    var colors = [C.muted, C.warm, C.accent];
    var byKey = {};
    D.benchmark.table.forEach(function (r) {
      byKey[r.protocol + "|" + (SHORT[r.model] || r.model)] = r;
    });

    root.innerHTML =
      "<h2>Does the CNN beat the gradient-boosted-tree baseline?</h2>" +
      "<p class='lead'><strong>No — it ties under the paper's protocol and wins on the frozen " +
      "test split, and neither result beats a tuned LightGBM on the same hand-crafted " +
      "features.</strong> On roughly 4,400 guides, a learned sequence representation buys " +
      "nothing over well-chosen k-mer features.</p>" +
      "<div class='chart' id='c-bench'></div>" +
      "<p class='cap'>Mean Spearman across the 18 (gene, drug) evaluation groups; error bars " +
      "are ±1 SD across those groups. That spread (~0.08) is larger than every gap between " +
      "models, which is the whole story. Dashed lines mark the 0.4–0.6 band Doench 2016 " +
      "reports graphically for boosted regression trees in Fig. 4c.</p>" +
      "<h3>Are the differences real?</h3>" +
      "<div id='t-sig'></div>" +
      "<p class='cap'>Paired across the 18 evaluation groups, with a 10,000-sample bootstrap " +
      "on the mean difference. For LightGBM vs Rule Set 2 the two tests disagree: the sign is " +
      "consistent across genes (p = 0.006) while the CI on the mean spans zero, which means " +
      "reliably a little better rather than decisively better.</p>" +
      "<h3>Per-gene difference, CNN minus Rule Set 2 GBT</h3>" +
      "<div class='chart' id='c-pergene'></div>" +
      "<p class='cap' id='cap-pergene'></p>";

    window.Charts.bars($("#c-bench", root), {
      categories: ["Leave-one-gene-out", "Gene-held-out test"],
      series: names.map(function (n, i) {
        return {
          name: n, color: colors[i],
          data: protocols.map(function (p) { var r = byKey[p + "|" + n]; return r ? r.spearman_per_gene_mean : null; }),
          error: protocols.map(function (p) { var r = byKey[p + "|" + n]; return r ? r.spearman_per_gene_sd : null; }),
        };
      }),
      yMax: 0.72, showValues: true, valueDecimals: 3,
      xTitle: "Evaluation protocol", yTitle: "Mean per-gene Spearman",
      refLines: [
        { value: 0.6, label: "Doench Fig. 4c upper", color: C.faint },
        { value: 0.4, label: "Doench Fig. 4c lower", color: C.faint },
      ],
      height: 380,
    });

    $("#t-sig", root).innerHTML = table(
      ["Comparison", "Mean diff", "95% CI", "Wilcoxon p", "Verdict"],
      D.benchmark.comparisons.map(function (c) {
        return [
          "<span class='dot " + (c.significant ? "info" : "warn") + "'></span>" +
            esc(c.model_a) + " − " + esc(c.model_b),
          (c.mean_diff >= 0 ? "+" : "") + c.mean_diff.toFixed(4),
          "[" + c.ci95_low.toFixed(3) + ", " + c.ci95_high.toFixed(3) + "]",
          c.wilcoxon_p.toFixed(3),
          c.significant ? "Consistent sign, small effect" : "Tie — CI spans zero",
        ];
      }),
      { align: ["", "n", "n", "n", ""] }
    );

    var pg = D.per_gene.map(function (r) {
      return {
        label: r.gene === "MED12" ? r.gene + " (" + r.drug + ")" : r.gene,
        d: r.spearman_cnn - r.spearman_gbt,
        gbt: r.spearman_gbt, cnn: r.spearman_cnn, n: r.n,
      };
    }).sort(function (a, b) { return a.d - b.d; });

    window.Charts.hbars($("#c-pergene", root), {
      categories: pg.map(function (r) { return r.label; }),
      values: pg.map(function (r) { return r.d; }),
      colors: pg.map(function (r) { return r.d > 0 ? C.good : C.bad; }),
      tipExtra: pg.map(function (r) {
        return "GBT " + r.gbt.toFixed(3) + " · CNN " + r.cnn.toFixed(3) + " · n = " + r.n;
      }),
      xTitle: "Difference in Spearman (CNN − GBT)", labelWidth: 170, tickDecimals: 2,
    });

    var wins = pg.filter(function (r) { return r.d > 0; }).length;
    $("#cap-pergene", root).textContent =
      "Out-of-fold leave-one-gene-out predictions. The CNN wins on " + wins + " of " +
      pg.length + " groups and loses on " + (pg.length - wins) +
      " — mixed signs are what a statistical tie looks like.";
  }

  // ---------------------------------------------------------------- scorer

  var scorerState = { alpha: "0.1", useDefaults: true };

  function scorer(root) {
    var g = D.guides;
    var byY = g.slice().sort(function (a, b) { return b.y - a.y; });
    var worst = g.slice().sort(function (a, b) {
      return Math.abs(b.gh - b.y) - Math.abs(a.gh - a.y);
    });
    var examples = [
      ["Strong guide (measured top 5%)", byY[7].seq],
      ["Weak guide (measured bottom 5%)", byY[byY.length - 8].seq],
      ["Guide the model gets wrong", worst[3].seq],
      ["Custom", ""],
    ];

    root.innerHTML =
      "<h2>Score a guide</h2>" +
      "<p class='lead'>Runs the <strong>gene-held-out ensemble</strong> — the same three-seed " +
      "model behind the 0.495 test number, not an in-sample refit — entirely in your browser, " +
      "then wraps it in a split conformal interval calibrated on held-out genes.</p>" +
      "<div class='grid g32'><div class='panel'>" +
        "<div class='field'><label for='ex'>Start from</label><select id='ex'>" +
          examples.map(function (e, i) { return "<option value='" + i + "'>" + esc(e[0]) + "</option>"; }).join("") +
        "</select></div>" +
        "<div class='field'><label for='seq'>30nt context sequence</label>" +
          "<input type='text' id='seq' spellcheck='false' autocomplete='off'>" +
          "<p class='hint' style='margin:6px 0 0'>4nt upstream + 20nt protospacer + 3nt NGG PAM + 3nt downstream.</p>" +
        "</div>" +
        "<div class='field'><label>Interval confidence</label>" +
          "<div class='seg' id='alpha'>" +
            "<button data-a='0.2'>80%</button><button data-a='0.1'>90%</button><button data-a='0.05'>95%</button>" +
          "</div></div>" +
        "<div class='field'><label class='check'><input type='checkbox' id='defpos' checked> " +
          "Use median gene position</label>" +
          "<p class='hint' style='margin:6px 0 0'>The model consumes three gene-position " +
          "features a bare 30mer cannot supply.</p></div>" +
        "<div class='field row2' id='posin' style='display:none'>" +
          "<div><label for='pp'>Percent peptide</label><input type='number' id='pp' step='0.5' min='0' max='100'></div>" +
          "<div><label for='cut'>AA cut position</label><input type='number' id='cut' step='1' min='0'></div>" +
        "</div>" +
      "</div><div id='out'></div></div>" +
      "<h3>Where the model looked</h3>" +
      "<div class='chart' id='c-attn1'></div>" +
      "<p class='cap' id='cap-attn1'></p>";

    var seqIn = $("#seq", root);
    seqIn.value = examples[0][1];
    $("#pp", root).value = window.MODEL.defaults.percent_peptide;
    $("#cut", root).value = window.MODEL.defaults.aa_cut_position;

    function syncAlpha() {
      Array.prototype.forEach.call($("#alpha", root).children, function (b) {
        b.setAttribute("aria-pressed", b.dataset.a === scorerState.alpha);
      });
    }
    syncAlpha();

    function run() {
      var out = $("#out", root);
      var attn = $("#c-attn1", root);
      var cap = $("#cap-attn1", root);
      var seq;
      try {
        seq = window.CrisprInfer.validate30mer(seqIn.value);
      } catch (e) {
        out.innerHTML = "<div class='err'>" + esc(e.message) + "</div>";
        attn.innerHTML = "";
        cap.textContent = "";
        return;
      }

      var pp = scorerState.useDefaults ? window.MODEL.defaults.percent_peptide : parseFloat($("#pp", root).value);
      var cut = scorerState.useDefaults ? window.MODEL.defaults.aa_cut_position : parseFloat($("#cut", root).value);
      if (isNaN(pp) || isNaN(cut)) {
        out.innerHTML = "<div class='err'>Percent peptide and cut position must both be numbers.</div>";
        return;
      }

      var r = model.predict(seq, pp, cut);
      var iv = model.interval(r.prediction, scorerState.alpha);
      var lvl = Math.round((1 - parseFloat(scorerState.alpha)) * 100);

      var known = null;
      for (var i = 0; i < D.guides.length; i++) {
        if (D.guides[i].seq === seq) { known = D.guides[i]; break; }
      }

      out.innerHTML =
        "<div class='metrics'>" +
          "<div class='metric'><div class='v'>" + r.prediction.toFixed(3) + "</div>" +
            "<div class='l'>Predicted efficiency</div>" +
            "<div class='s'>rank within a (gene, drug) group, 0–1</div></div>" +
          "<div class='metric'><div class='v'>" + iv.low.toFixed(2) + " – " + iv.high.toFixed(2) + "</div>" +
            "<div class='l'>" + lvl + "% conformal interval</div>" +
            "<div class='s'>±" + iv.halfWidth.toFixed(3) + "</div></div>" +
          "<div class='metric'><div class='v'>±" + r.seedSpread.toFixed(4) + "</div>" +
            "<div class='l'>Ensemble disagreement</div>" +
            "<div class='s'>spread across 3 seeds</div></div>" +
        "</div>" +
        (known
          ? "<p class='ok-line'><span class='dot good'></span>In the dataset: <strong>" +
            esc(known.gene) + "</strong> (" + esc(known.drug) + "), measured activity <strong>" +
            known.y.toFixed(3) + "</strong>, ranked <strong>" + known.rank + " of " + known.group_n +
            "</strong> in its group. It was in the <strong>" + esc(known.split) + "</strong> split.</p>"
          : "<p class='ok-line'><span class='dot info'></span>Not in the training dataset — this " +
            "is a genuine out-of-sample prediction.</p>") +
        (iv.halfWidth > 0.35
          ? "<div class='note warn'><span class='t'>This interval is wide</span><p>It spans " +
            (iv.high - iv.low).toFixed(2) + " of a (0, 1] scale. The intervals are calibrated, " +
            "but wide enough to be useful mainly for flagging predictions not to trust rather " +
            "than for ranking individual guides.</p></div>"
          : "") +
        "<p class='hint'>GC in protospacer " + r.gcCount + "/20 · " +
          "Tm(30mer) " + r.thermoRaw[3].toFixed(1) + " °C · " +
          "per-seed " + r.perSeed.map(function (p) { return p.toFixed(3); }).join(", ") + "</p>";

      window.Charts.bars(attn, {
        categories: seq.split(""),
        tipCategories: seq.split("").map(function (b, i) {
          return b + " at position " + POS[i] + " (" + REGION[i] + ")";
        }),
        series: [{ name: "Attention weight", color: C.accent, data: r.attention }],
        xTitle: "Base and position in the 30mer",
        yTitle: "Attention-pooling weight",
        refLines: [{ value: 1 / 30, label: "uniform 1/30", color: C.bad }],
        height: 300, tickSize: 10.5, yDecimals: 2,
      });
      // Recolour bars by region -- the chart helper colours by series, not category.
      var rects = attn.querySelectorAll("rect");
      Array.prototype.forEach.call(rects, function (rc, i) {
        if (i < 30) rc.setAttribute("fill", REGION_COLOR[REGION[i]]);
      });
      var labels = attn.querySelectorAll("text");
      Array.prototype.forEach.call(labels, function (t) {
        if (t.textContent.length === 1 && "ACGT".indexOf(t.textContent) >= 0) {
          t.setAttribute("font-family", "ui-monospace, Menlo, monospace");
        }
      });

      cap.textContent =
        "Computed live in your browser for this exact sequence, averaged over the three " +
        "ensemble seeds. Orange marks the seed region (protospacer 17–20), green the PAM, " +
        "blue the rest of the protospacer, grey the flanking context.";
    }

    seqIn.addEventListener("input", run);
    $("#ex", root).addEventListener("change", function () {
      var v = examples[parseInt(this.value, 10)][1];
      if (v) { seqIn.value = v; run(); }
    });
    $("#alpha", root).addEventListener("click", function (e) {
      if (e.target.dataset && e.target.dataset.a) {
        scorerState.alpha = e.target.dataset.a;
        syncAlpha();
        run();
      }
    });
    $("#defpos", root).addEventListener("change", function () {
      scorerState.useDefaults = this.checked;
      $("#posin", root).style.display = this.checked ? "none" : "grid";
      run();
    });
    $("#pp", root).addEventListener("input", run);
    $("#cut", root).addEventListener("input", run);
    run();
  }

  // -------------------------------------------------------------- explorer

  function explorer(root) {
    var genes = {};
    D.guides.forEach(function (r) { genes[r.gene] = 1; });
    var geneList = Object.keys(genes).sort();

    root.innerHTML =
      "<h2>Guide explorer</h2>" +
      "<p class='lead'>Every guide in the dataset with its measured activity and all three " +
      "models' <strong>out-of-fold</strong> leave-one-gene-out predictions — each made by a " +
      "model that never saw that guide's gene during training.</p>" +
      "<div class='panel' style='margin-bottom:20px'><div class='row2'>" +
        "<div><label for='fg'>Gene</label><select id='fg'><option value=''>All genes</option>" +
          geneList.map(function (g) { return "<option>" + esc(g) + "</option>"; }).join("") +
        "</select></div>" +
        "<div><label for='fs'>Gene-held-out split</label><select id='fs'>" +
          "<option value=''>All splits</option><option>train</option><option>cal</option><option>test</option>" +
        "</select></div></div>" +
        "<div class='field' style='margin-top:12px'><label for='fq'>Sequence contains</label>" +
        "<input type='text' id='fq' placeholder='e.g. GGGTTT' spellcheck='false'></div>" +
      "</div>" +
      "<p class='cap' id='count' style='margin-bottom:10px'></p>" +
      "<div id='tbl'></div>" +
      "<h3>Measured versus predicted</h3>" +
      "<div class='seg' id='mpick' style='margin-bottom:10px'>" +
        "<button data-m='cnn'>CNN</button><button data-m='gbt'>Rule Set 2 GBT</button>" +
        "<button data-m='lgbm'>LightGBM</button></div>" +
      "<div class='chart' id='c-scatter'></div>" +
      "<p class='cap' id='cap-scatter'></p>";

    var pick = "cnn";
    function sync() {
      Array.prototype.forEach.call($("#mpick", root).children, function (b) {
        b.setAttribute("aria-pressed", b.dataset.m === pick);
      });
    }

    function spearman(rows, col) {
      var n = rows.length;
      if (n < 3) return NaN;
      function ranks(vals) {
        var idx = vals.map(function (v, i) { return [v, i]; })
          .sort(function (a, b) { return a[0] - b[0]; });
        var r = new Array(n), i = 0;
        while (i < n) {
          var j = i;
          while (j + 1 < n && idx[j + 1][0] === idx[i][0]) j++;
          var avg = (i + j) / 2 + 1;
          for (var k = i; k <= j; k++) r[idx[k][1]] = avg;
          i = j + 1;
        }
        return r;
      }
      var a = ranks(rows.map(function (r) { return r.y; }));
      var b = ranks(rows.map(function (r) { return r[col]; }));
      var ma = a.reduce(function (s, v) { return s + v; }, 0) / n;
      var mb = b.reduce(function (s, v) { return s + v; }, 0) / n;
      var num = 0, da = 0, db = 0;
      for (var i2 = 0; i2 < n; i2++) {
        num += (a[i2] - ma) * (b[i2] - mb);
        da += (a[i2] - ma) * (a[i2] - ma);
        db += (b[i2] - mb) * (b[i2] - mb);
      }
      return num / Math.sqrt(da * db);
    }

    function apply() {
      var g = $("#fg", root).value, s = $("#fs", root).value;
      var q = $("#fq", root).value.trim().toUpperCase();
      var f = D.guides.filter(function (r) {
        return (!g || r.gene === g) && (!s || r.split === s) && (!q || r.seq.indexOf(q) >= 0);
      });

      $("#count", root).textContent =
        f.length.toLocaleString() + " of " + D.guides.length.toLocaleString() + " guides";

      var sorted = f.slice().sort(function (a, b) {
        return Math.abs(a.cnn - a.y) - Math.abs(b.cnn - b.y);
      }).slice(0, 300);

      $("#tbl", root).innerHTML = f.length
        ? table(
            ["30mer", "Gene", "Drug", "Measured", "Rule Set 2 GBT", "LightGBM", "CNN", "Split", "Rank"],
            sorted.map(function (r) {
              return [r.seq, esc(r.gene), esc(r.drug), r.y.toFixed(3), r.gbt.toFixed(3),
                r.lgbm.toFixed(3), r.cnn.toFixed(3), r.split, r.rank + " / " + r.group_n];
            }),
            { align: ["m", "", "", "n", "n", "n", "n", "", "n"], scroll: true }
          )
        : "<p class='cap'>No guides match those filters.</p>";

      if (!f.length) {
        $("#c-scatter", root).innerHTML = "";
        $("#cap-scatter", root).textContent = "";
        return;
      }

      var step = Math.max(1, Math.ceil(f.length / 3000));
      var pts = [];
      for (var i = 0; i < f.length; i += step) pts.push([f[i].y, f[i][pick]]);

      window.Charts.scatter($("#c-scatter", root), {
        points: pts, xMin: 0, xMax: 1, yMin: 0, yMax: 1, width: 560, height: 420,
        xTitle: "Measured activity (rank, 0–1)",
        yTitle: "Out-of-fold prediction",
      });
      var label = { cnn: "CNN", gbt: "Rule Set 2 GBT", lgbm: "LightGBM" }[pick];
      $("#cap-scatter", root).textContent =
        label + " on this selection: Spearman " + spearman(f, pick).toFixed(3) +
        " over " + f.length.toLocaleString() + " guides (" + pts.length.toLocaleString() +
        " plotted). Dashed line is y = x. Predictions compress toward the middle because " +
        "the models are fit with squared error on a rank target.";
    }

    $("#fg", root).addEventListener("change", apply);
    $("#fs", root).addEventListener("change", apply);
    $("#fq", root).addEventListener("input", apply);
    $("#mpick", root).addEventListener("click", function (e) {
      if (e.target.dataset && e.target.dataset.m) { pick = e.target.dataset.m; sync(); apply(); }
    });
    sync();
    apply();
  }

  // ------------------------------------------------------------- attention

  function attention(root) {
    var a = D.attention_by_position;
    var it = D.interpretability;

    root.innerHTML =
      "<h2>What the model learned to look at</h2>" +
      "<div class='seg' id='apick' style='margin-bottom:14px'>" +
        "<button data-k='attention_weight'>CNN attention</button>" +
        "<button data-k='gbt_positional_importance'>GBT importance (independent control)</button>" +
      "</div>" +
      "<div class='chart' id='c-attn'></div>" +
      "<p class='cap' id='cap-attn'></p>" +
      "<div class='grid g2' style='margin-top:30px'>" +
        "<div><h3>Weight by region, CNN attention</h3><div id='t-region'></div>" +
          "<p class='cap'>Seed versus the rest of the protospacer, paired across guides: " +
          "Wilcoxon p ≈ " + it.wilcoxon_p.toExponential(0) + ".</p></div>" +
        "<div><h3>The seed-region prior holds</h3>" +
          "<p>Attention rises along the protospacer toward the PAM and peaks at the PAM's " +
          "variable N base — consistent with Cas9 requiring PAM recognition before it " +
          "interrogates the protospacer, and with seed-proximal mismatches being the most " +
          "disruptive.</p>" +
          "<div class='note good'><span class='t'>Independent control agrees</span>" +
          "<p>Attention weights are easy to over-read, so the same question went to the GBT, " +
          "which needs no attention mechanism to be interpretable. It places <strong>" +
          pct(it.gbt_seed_importance_share) + "</strong> of its positional importance in the " +
          "four seed positions, against 13.3% if it were flat. Two methods with nothing in " +
          "common agree on where the signal is.</p></div></div>" +
      "</div>";

    var key = "attention_weight";
    function draw() {
      window.Charts.bars($("#c-attn", root), {
        categories: a.map(function (r) { return r.position_label; }),
        tipCategories: a.map(function (r) { return "position " + r.position_label + " (" + r.region + ")"; }),
        series: [{ name: key === "attention_weight" ? "Attention weight" : "Gini importance",
                   color: C.accent, data: a.map(function (r) { return r[key]; }) }],
        xTitle: "Position in the 30mer",
        yTitle: key === "attention_weight" ? "Attention-pooling weight" : "Normalised Gini importance",
        refLines: [{ value: 1 / 30, label: "uniform 1/30", color: C.bad }],
        height: 330, tickSize: 10.5, yDecimals: 2,
      });
      var rects = $("#c-attn", root).querySelectorAll("rect");
      Array.prototype.forEach.call(rects, function (rc, i) {
        if (i < a.length) rc.setAttribute("fill", REGION_COLOR[a[i].region]);
      });
      Array.prototype.forEach.call($("#apick", root).children, function (b) {
        b.setAttribute("aria-pressed", b.dataset.k === key);
      });
      $("#cap-attn", root).textContent = key === "attention_weight"
        ? "Mean attention-pooling weight over " + it.n_guides.toLocaleString() +
          " held-out guides. Orange is the seed (protospacer 17–20), green the PAM, blue the " +
          "rest of the protospacer, grey the flanking context. Weights sum to 1 across positions."
        : "Order-1 position-dependent nucleotide features from the Rule Set 2 GBT, aggregated " +
          "per position and normalised to sum to 1. This model has no attention mechanism, so " +
          "it is an independent read on the same question.";
    }

    var order = ["PAM", "seed", "protospacer", "3' context", "5' context"];
    var labels = {
      PAM: "PAM (NGG)", seed: "Seed, protospacer 17–20",
      protospacer: "Rest of protospacer, 1–16",
      "3' context": "3′ context (+1…+3)", "5' context": "5′ context (−4…−1)",
    };
    var means = {};
    a.forEach(function (r) {
      (means[r.region] = means[r.region] || []).push(r.attention_weight);
    });
    $("#t-region", root).innerHTML = table(
      ["Region", "Mean weight", "vs uniform"],
      order.map(function (k, i) {
        var m = means[k].reduce(function (s, v) { return s + v; }, 0) / means[k].length;
        return ["<span class='dot " + (i === 0 ? "good" : i === 1 ? "warn" : "info") + "'></span>" + labels[k],
                m.toFixed(4), (m / (1 / 30)).toFixed(2) + "×"];
      }),
      { align: ["", "n", "n"] }
    );

    $("#apick", root).addEventListener("click", function (e) {
      if (e.target.dataset && e.target.dataset.k) { key = e.target.dataset.k; draw(); }
    });
    draw();
  }

  // ----------------------------------------------------------- calibration

  function calibration(root) {
    var rnd = D.conformal["RANDOM split (exchangeable)"];
    var gh = D.conformal["GENE-HELD split (shifted)"];

    root.innerHTML =
      "<h2>Do the uncertainty intervals mean anything?</h2>" +
      "<p class='lead'><strong>Yes on exchangeable data, and conservatively on new genes.</strong> " +
      "Split conformal with absolute-residual nonconformity scores and the finite-sample " +
      "⌈(n+1)(1−α)⌉ quantile correction.</p>" +
      "<div class='chart' id='c-cal'></div>" +
      "<p class='cap'>Above the dashed diagonal is conservative (intervals wider than needed); " +
      "below is under-covering. Source: CNN + attention with " + rnd.n_cal.toLocaleString() +
      " / " + gh.n_cal.toLocaleString() + " calibration guides.</p>" +
      "<div class='grid g2' style='margin-top:30px'>" +
        "<div><h3>Coverage and width at key levels</h3><div id='t-cal'></div></div>" +
        "<div><h3>Interval width grows with confidence</h3><div class='chart' id='c-width'></div>" +
          "<p class='cap'>Mean width on a target bounded in (0, 1], so a width of 1.0 is " +
          "entirely uninformative.</p></div>" +
      "</div>" +
      "<div class='note warn'><span class='t'>The guarantee does not formally hold under gene shift</span>" +
      "<p>Conformal requires calibration and test data to be exchangeable. Under the random " +
      "split they are, and coverage tracks nominal almost exactly — that validates the " +
      "implementation. Under gene-held-out the calibration and test genes are disjoint, so the " +
      "guarantee lapses; empirically it lands conservative (" + pct(gh.coverage_at_90) +
      " at nominal 90%), which is the safe direction but is luck rather than a theorem. " +
      "Separately, a ±" + gh.half_width_at_90.toFixed(2) + " interval on a (0, 1] target is " +
      "wide enough that these are useful for flagging low-confidence predictions, not for " +
      "ranking individual guides.</p></div>";

    var xs = rnd.curve.map(function (r) { return r.nominal_coverage; });
    window.Charts.lines($("#c-cal", root), {
      x: xs,
      series: [
        { name: "Perfect calibration", color: C.faint, data: xs.slice(), dash: "4 3", width: 1, markers: false },
        { name: "Random split (exchangeable)", color: C.accent,
          data: rnd.curve.map(function (r) { return r.empirical_coverage; }) },
        { name: "Gene-held-out split (shifted)", color: C.bad,
          data: gh.curve.map(function (r) { return r.empirical_coverage; }) },
      ],
      xTitle: "Nominal coverage (1 − α)", yTitle: "Empirical coverage on held-out guides",
      yMin: 0, yMax: 1, height: 400,
    });

    window.Charts.lines($("#c-width", root), {
      x: xs,
      series: [
        { name: "Random split", color: C.accent, data: rnd.curve.map(function (r) { return r.mean_width; }), markers: false },
        { name: "Gene-held-out split", color: C.bad, data: gh.curve.map(function (r) { return r.mean_width; }), markers: false },
      ],
      xTitle: "Nominal coverage (1 − α)", yTitle: "Mean interval width",
      yMin: 0, height: 300, width: 520,
    });

    function at(curve, lvl) {
      for (var i = 0; i < curve.length; i++) {
        if (Math.abs(curve[i].nominal_coverage - lvl) < 1e-6) return curve[i];
      }
      return null;
    }
    $("#t-cal", root).innerHTML = table(
      ["Nominal", "Random", "Gene-held", "Half-width"],
      [0.95, 0.9, 0.8, 0.5].map(function (l) {
        var r = at(rnd.curve, l), g = at(gh.curve, l);
        return [pct(l, 0), pct(r.empirical_coverage), pct(g.empirical_coverage),
                "±" + (r.mean_width / 2).toFixed(3)];
      }),
      { align: ["n", "n", "n", "n"] }
    );
  }

  // ------------------------------------------------------------- selection

  function selection(root) {
    var seen = {}, sw = [];
    D.sweep.results.forEach(function (r) {
      if (!seen[r.config]) { seen[r.config] = 1; sw.push(r); }
    });
    sw.sort(function (a, b) { return a.mean_spearman - b.mean_spearman; });
    var sel = D.sweep.selected;
    var ref = D.sweep.gbt_reference_same_protocol;

    root.innerHTML =
      "<h2>How the architecture was chosen</h2>" +
      "<p class='lead'>" + sw.length + " configurations, scored by leave-one-gene-out " +
      "restricted to the <strong>training genes only</strong> — the test and calibration genes " +
      "were never loaded during the sweep, so the reported test numbers stay honest.</p>" +
      "<div class='chart' id='c-sweep'></div>" +
      "<p class='cap'>Selected configuration in blue: <code>" + esc(sel.config) + "</code> at " +
      sel.mean_spearman.toFixed(4) + ". The dashed line is the Rule Set 2 GBT measured on this " +
      "same inner protocol, which is the bar that matters here.</p>" +
      "<div class='grid g2' style='margin-top:30px'>" +
        "<div class='note good'><span class='t'>What worked</span>" +
          "<p><strong>Feature-map dropout (+0.055).</strong> The literal spec architecture " +
          "reached 0.99 training Spearman against 0.21 held-out. <code>Dropout1d</code> was the " +
          "single most effective fix.</p>" +
          "<p><strong>Learned positional embedding (+0.010).</strong> Convolutions are " +
          "translation-equivariant, so without one the network cannot tell position 4 from " +
          "position 20 — yet Rule Set 2 draws 58% of its Gini importance from position-specific " +
          "nucleotide identity.</p></div>" +
        "<div class='note bad'><span class='t'>What did not work</span>" +
          "<p><strong>Dropping gene-position features (0.377 → 0.179).</strong> The hypothesis " +
          "was that percent-peptide and cut position are gene-level shortcuts that cannot " +
          "transfer across genes. Measurement said the opposite; they were kept.</p>" +
          "<p><strong>Narrowing the network.</strong> Every narrower variant scored below the " +
          "64/128 stack. Capacity was not the problem; regularisation was.</p></div>" +
      "</div>" +
      "<div class='note info'><span class='t'>Why " + sel.mean_spearman.toFixed(3) +
        " did not mean abandoning the model</span><p>The inner protocol trains on 5 genes " +
        "instead of 16, so everything scores lower here. Re-measuring the GBT on the same " +
        "protocol gave " + ref.toFixed(3) + " rather than its 0.515 leave-one-gene-out figure, " +
        "which established the correct bar. Given the full protocol, the CNN reached 0.507.</p></div>";

    window.Charts.hbars($("#c-sweep", root), {
      categories: sw.map(function (r) { return r.config; }),
      values: sw.map(function (r) { return r.mean_spearman; }),
      colors: sw.map(function (r) { return r.config === sel.config ? C.accent : C.pale; }),
      tipExtra: sw.map(function (r) {
        return "conv drop " + r.conv_dropout + " · wd " + r.weight_decay +
               " · " + r.c1 + "/" + r.c2 + (r.positional ? " · positional" : "");
      }),
      refLines: [{ value: ref, label: "GBT, same protocol", color: C.warm }],
      xTitle: "Mean Spearman over held-out training genes",
      labelWidth: 250, rowHeight: 23, tickDecimals: 2,
    });
  }

  // ------------------------------------------------------------ validation

  function validation(root) {
    var v = D.validation;
    var passed = v.filter(function (c) { return c.passed; }).length;

    root.innerHTML =
      "<h2>Adversarial validation: " + passed + " of " + v.length + " checks passing</h2>" +
      "<p class='lead'>Each check is written so it would actually fail if the corresponding bug " +
      "were present, rather than restating what the code does. Run with " +
      "<code>python scripts/08_validate.py</code>; it exits non-zero on any failure.</p>" +
      "<div id='t-val'></div>" +
      "<div class='note warn'><span class='t'>Two genuine findings, neither changing a conclusion</span>" +
      "<p><strong>The random split leaks duplicate sequences, not just gene context.</strong> " +
      "230 test rows share a 30mer with a training row, because MED12 was screened under two " +
      "drugs. Re-running conformal on only the test rows whose sequence the model never saw " +
      "gives 89.2% coverage at 90% nominal, against 90.2% overall.</p>" +
      "<p><strong>Melting temperature on short segments is not physically meaningful.</strong> " +
      "Rule Set 2 computes Tm on 5nt and 8nt sub-segments, below the range where " +
      "nearest-neighbour thermodynamics is valid, so the 5-mer values come out negative " +
      "(−69 to +1 °C). This is inherited from the original design, not introduced here.</p></div>";

    $("#t-val", root).innerHTML = table(
      ["Check", "Observed"],
      v.map(function (c) {
        return ["<span class='dot " + (c.passed ? "good" : "bad") + "'></span>" + esc(c.check),
                "<span style='color:var(--muted)'>" + esc(c.detail || "—") + "</span>"];
      }),
      { scroll: true }
    );
  }

  // ------------------------------------------------------------------ data

  function dataSection(root) {
    var m = D.meta;
    var gs = D.gene_summary;
    var cmap = { train: C.accent, cal: C.warm, test: C.good };

    root.innerHTML =
      "<h2>Dataset and splits</h2>" +
      "<div class='stats' style='margin-bottom:8px'>" +
        "<div class='stat'><div class='v'>" + m.n_rows.toLocaleString() + "</div><div class='l'>Guide × drug rows</div></div>" +
        "<div class='stat'><div class='v'>" + m.n_unique_guides.toLocaleString() + "</div><div class='l'>Unique 30mers</div></div>" +
        "<div class='stat'><div class='v'>" + m.n_genes + "</div><div class='l'>Genes</div></div>" +
        "<div class='stat'><div class='v'>" + m.n_calibration.toLocaleString() + "</div><div class='l'>Calibration guides</div></div>" +
      "</div>" +
      "<p class='cap'>" + esc(m.dataset) + ". Target is <code>" + esc(m.target) + "</code> — " +
        esc(m.target_note) + ".</p>" +
      "<div class='grid g32' style='margin-top:28px'>" +
        "<div><h3>Rows per gene, coloured by split</h3><div class='chart' id='c-genes'></div>" +
          "<p class='cap'>Blue train, orange calibration, green test. MED12 alone is 35% of all " +
          "rows, so genes are assigned largest-first to whichever split is furthest below its " +
          "quota rather than at random.</p></div>" +
        "<div><h3>Why three protocols</h3>" +
          "<p><strong>Leave-one-gene-out (primary).</strong> Azimuth's own default and how " +
          "Doench 2016 Fig. 4 reports Spearman. The paper-comparable number.</p>" +
          "<p><strong>Frozen gene-held-out 60/20/20.</strong> Whole genes to train, calibration, " +
          "and test. Needed for conformal, which requires a disjoint calibration set.</p>" +
          "<p><strong>Random row-level.</strong> Reported only as a leakage contrast — it " +
          "inflates the GBT from 0.454 to 0.544, roughly the size of the entire effect being " +
          "measured.</p>" +
          "<div class='note good'><span class='t'>Verified against the raw source</span>" +
          "<p>The target was independently re-derived from <code>V1_data.xlsx</code> and " +
          "<code>V2_data.xlsx</code> by porting Azimuth's Python 2 rank-transform logic. " +
          "Maximum absolute difference across all " + m.n_rows.toLocaleString() +
          " rows: <strong>5 × 10⁻¹⁰</strong>.</p></div></div>" +
      "</div>" +
      "<h3>Chromatin accessibility channel: cut, with evidence</h3>" +
      "<div id='t-chrom'></div>" +
      "<div class='note bad'><span class='t'>Three independent blockers, any one sufficient</span>" +
      "<p>A375 generated 65% of the rows and has <strong>zero</strong> chromatin accessibility " +
      "experiments on ENCODE; its only ENCODE data is RNA-based.</p>" +
      "<p>The remaining rows span three more human lines plus mouse, so a per-row match would " +
      "need several tracks across two genomes — and TF1 has no ENCODE data at all.</p>" +
      "<p>The Azimuth release carries no genomic coordinates, only Ensembl transcript IDs and " +
      "transcript-relative cut positions, so there is nothing to query a bigWig with.</p></div>";

    window.Charts.hbars($("#c-genes", root), {
      categories: gs.map(function (r) { return r.gene; }),
      values: gs.map(function (r) { return r.n; }),
      colors: gs.map(function (r) { return cmap[r.split]; }),
      tipExtra: gs.map(function (r) { return r.split + " split"; }),
      xTitle: "Guide × drug rows", labelWidth: 96, rowHeight: 22,
      tipDecimals: 0, tickDecimals: 0,
    });

    $("#t-chrom", root).innerHTML = table(
      ["Cell line", "Rows", "ENCODE ATAC-seq / DNase-seq"],
      [
        ["<span class='dot bad'></span>A375 (RES half)", "3,473", "0 experiments — only RNA-based data"],
        ["<span class='dot warn'></span>NB4 + TF1 + MOLM-13 (FC, human)", "882", "TF1 none; NB4 1; MOLM-13 3"],
        ["<span class='dot bad'></span>Mouse lines (FC)", "955", "different genome entirely"],
      ],
      { align: ["", "n", ""] }
    );
  }

  // ----------------------------------------------------------------- shell

  var SECTIONS = [
    ["Benchmark", benchmark],
    ["Score a guide", scorer],
    ["Guide explorer", explorer],
    ["Attention", attention],
    ["Calibration", calibration],
    ["Architecture selection", selection],
    ["Validation", validation],
    ["Data & splits", dataSection],
  ];

  function init() {
    var nav = $("#nav"), host = $("#sections");
    var built = {};

    SECTIONS.forEach(function (s, i) {
      var b = document.createElement("button");
      b.textContent = s[0];
      b.setAttribute("role", "tab");
      b.dataset.i = i;
      nav.appendChild(b);
      var sec = document.createElement("section");
      sec.id = "sec-" + i;
      host.appendChild(sec);
    });

    function show(i) {
      SECTIONS.forEach(function (s, j) {
        $("#sec-" + j).classList.toggle("on", i === j);
        nav.children[j].setAttribute("aria-selected", i === j);
      });
      if (!built[i]) { SECTIONS[i][1]($("#sec-" + i)); built[i] = 1; }
      if (location.hash !== "#" + i) history.replaceState(null, "", "#" + i);
    }

    nav.addEventListener("click", function (e) {
      if (e.target.dataset && e.target.dataset.i != null) show(parseInt(e.target.dataset.i, 10));
    });

    var start = parseInt((location.hash || "#0").slice(1), 10);
    show(isNaN(start) || start < 0 || start >= SECTIONS.length ? 0 : start);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
