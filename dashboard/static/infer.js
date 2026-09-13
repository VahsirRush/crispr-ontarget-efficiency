/* Browser-side forward pass for the gene-held-out CNN + attention ensemble.
 *
 * This is a hand port of crispr/model.py CrisprCNN.forward plus the feature
 * extraction in crispr/features.py, so a static HTML page can score a pasted
 * guide with no server. It is deliberately written as plain loops rather than
 * anything clever: it runs once per click on a 98k-parameter network, so
 * clarity is worth more than speed here.
 *
 * scripts/10_build_static_dashboard.py checks this file against PyTorch on
 * every guide in the dataset and refuses to build if they disagree, so the
 * port is verified rather than asserted.
 *
 * Dropout and Dropout1d are identity at eval time and BatchNorm uses its
 * running statistics, which is why neither appears below.
 */

(function (root) {
  "use strict";

  var BASES = ["A", "T", "C", "G"]; // crispr/features.py BASES order

  function decodeF32(b64) {
    var bin =
      typeof atob === "function"
        ? atob(b64)
        : Buffer.from(b64, "base64").toString("binary");
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  // ---- primitives --------------------------------------------------------

  /** Conv1d with padding="same" over an [L][Cin] input; odd kernels only. */
  function conv1d(x, w, b, cIn, cOut, k, L) {
    var pad = (k - 1) / 2;
    var out = [];
    for (var t = 0; t < L; t++) {
      var row = new Float64Array(cOut);
      for (var o = 0; o < cOut; o++) {
        var acc = b[o];
        for (var c = 0; c < cIn; c++) {
          var base = o * cIn * k + c * k;
          for (var j = 0; j < k; j++) {
            var s = t + j - pad;
            if (s >= 0 && s < L) acc += w[base + j] * x[s][c];
          }
        }
        row[o] = acc;
      }
      out.push(row);
    }
    return out;
  }

  /** BatchNorm1d in eval mode, applied per channel across the length axis. */
  function batchNorm(x, gamma, beta, mean, varr, eps, C) {
    for (var t = 0; t < x.length; t++) {
      for (var c = 0; c < C; c++) {
        x[t][c] = ((x[t][c] - mean[c]) / Math.sqrt(varr[c] + eps)) * gamma[c] + beta[c];
      }
    }
    return x;
  }

  function relu(x) {
    for (var t = 0; t < x.length; t++) {
      for (var c = 0; c < x[t].length; c++) if (x[t][c] < 0) x[t][c] = 0;
    }
    return x;
  }

  /** y = x @ W^T + b, with W stored row-major as [out][in]. */
  function linear(x, w, b, nIn, nOut) {
    var y = new Float64Array(nOut);
    for (var o = 0; o < nOut; o++) {
      var acc = b[o];
      var base = o * nIn;
      for (var i = 0; i < nIn; i++) acc += w[base + i] * x[i];
      y[o] = acc;
    }
    return y;
  }

  function linearSeq(x, w, b, nIn, nOut) {
    return x.map(function (row) {
      return linear(row, w, b, nIn, nOut);
    });
  }

  function softmax(v) {
    var m = -Infinity;
    for (var i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    var s = 0;
    var out = new Float64Array(v.length);
    for (i = 0; i < v.length; i++) {
      out[i] = Math.exp(v[i] - m);
      s += out[i];
    }
    for (i = 0; i < v.length; i++) out[i] /= s;
    return out;
  }

  // ---- features ----------------------------------------------------------

  /** Biopython Tm_NN(DNA_NN2) against a perfect complement, reduced to a sum.
   *  Constants and the validity of the reduction come from
   *  scripts/09_export_dashboard.py export_tm_constants(). */
  function meltingTemp(seq, tm) {
    var dh = tm.init_dh;
    var ds = tm.init_ds;
    for (var i = 0; i < seq.length - 1; i++) {
      var e = tm.dinucleotide[seq.substring(i, i + 2)];
      dh += e[0];
      ds += e[1];
    }
    ds += tm.salt_coeff * (seq.length - 1) * Math.log(tm.monovalent_molar);
    return (1000 * dh) / (ds + tm.R * Math.log(tm.k)) - 273.15;
  }

  function oneHot(seq) {
    var idx = {};
    for (var i = 0; i < BASES.length; i++) idx[BASES[i]] = i;
    var x = [];
    for (var t = 0; t < seq.length; t++) {
      var row = new Float64Array(4);
      row[idx[seq[t]]] = 1;
      x.push(row);
    }
    return x;
  }

  /** The 10-feature thermodynamic block, in the trained column order. */
  function thermoVector(seq, percentPeptide, aaCutPosition, tm) {
    var proto = seq.substring(4, 24);
    var gc = 0;
    for (var i = 0; i < proto.length; i++) {
      if (proto[i] === "G" || proto[i] === "C") gc++;
    }
    var v = [gc, gc > 10 ? 1 : 0, gc < 10 ? 1 : 0, meltingTemp(seq, tm)];
    for (var s = 0; s < tm.segments.length; s++) {
      v.push(meltingTemp(seq.substring(tm.segments[s][0], tm.segments[s][1]), tm));
    }
    v.push(percentPeptide, aaCutPosition, percentPeptide < 50 ? 1 : 0);
    return v;
  }

  // ---- model -------------------------------------------------------------

  function Model(spec) {
    this.spec = spec;
    this.arch = spec.arch;
    this.weights = spec.ensemble.map(function (e) {
      var w = {};
      for (var k in e) w[k] = decodeF32(e[k]);
      return w;
    });
  }

  /** One member of the ensemble. Mirrors CrisprCNN.forward exactly. */
  Model.prototype.forwardOne = function (w, x, thermo) {
    var a = this.arch;
    var L = a.seq_len;

    var h = conv1d(x, w["conv1.weight"], w["conv1.bias"], a.n_channels, a.c1, 3, L);
    h = relu(batchNorm(h, w["bn1.weight"], w["bn1.bias"], w["bn1.running_mean"],
      w["bn1.running_var"], a.bn_eps, a.c1));
    h = conv1d(h, w["conv2.weight"], w["conv2.bias"], a.c1, a.c2, 5, L);
    h = relu(batchNorm(h, w["bn2.weight"], w["bn2.bias"], w["bn2.running_mean"],
      w["bn2.running_var"], a.bn_eps, a.c2));

    if (a.positional) {
      var pe = w["pos_emb"];
      for (var t = 0; t < L; t++) {
        for (var c = 0; c < a.c2; c++) h[t][c] += pe[t * a.c2 + c];
      }
    }

    // Single-head scaled dot-product self-attention over the 30 positions.
    var q = linearSeq(h, w["attn.q.weight"], w["attn.q.bias"], a.c2, a.attn_dim);
    var kk = linearSeq(h, w["attn.k.weight"], w["attn.k.bias"], a.c2, a.attn_dim);
    var v = linearSeq(h, w["attn.v.weight"], w["attn.v.bias"], a.c2, a.c2);
    var scale = Math.pow(a.attn_dim, -0.5);

    var ctx = [];
    for (t = 0; t < L; t++) {
      var scores = new Float64Array(L);
      for (var j = 0; j < L; j++) {
        var d = 0;
        for (var z = 0; z < a.attn_dim; z++) d += q[t][z] * kk[j][z];
        scores[j] = d * scale;
      }
      var aw = softmax(scores);
      var row = new Float64Array(a.c2);
      for (j = 0; j < L; j++) {
        for (c = 0; c < a.c2; c++) row[c] += aw[j] * v[j][c];
      }
      ctx.push(row);
    }
    for (t = 0; t < L; t++) {
      for (c = 0; c < a.c2; c++) h[t][c] += ctx[t][c]; // residual
    }

    // Attention pooling: a learned scalar per position, softmaxed into weights.
    var raw = new Float64Array(L);
    for (t = 0; t < L; t++) {
      var hid = linear(h[t], w["pool.score.0.weight"], w["pool.score.0.bias"], a.c2, 64);
      for (z = 0; z < 64; z++) hid[z] = Math.tanh(hid[z]);
      raw[t] = linear(hid, w["pool.score.2.weight"], w["pool.score.2.bias"], 64, 1)[0];
    }
    var pw = softmax(raw);
    var pooled = new Float64Array(a.c2);
    for (t = 0; t < L; t++) {
      for (c = 0; c < a.c2; c++) pooled[c] += pw[t] * h[t][c];
    }

    var fused = new Float64Array(a.c2 + a.n_thermo);
    fused.set(pooled, 0);
    fused.set(thermo, a.c2);

    var o = linear(fused, w["head.0.weight"], w["head.0.bias"], a.c2 + a.n_thermo, 64);
    for (z = 0; z < 64; z++) if (o[z] < 0) o[z] = 0;
    o = linear(o, w["head.3.weight"], w["head.3.bias"], 64, 16);
    for (z = 0; z < 16; z++) if (o[z] < 0) o[z] = 0;
    var out = linear(o, w["head.6.weight"], w["head.6.bias"], 16, 1)[0];

    return { pred: 1 / (1 + Math.exp(-out)), pool: Array.from(pw) };
  };

  /** Ensemble mean over the trained seeds, matching scripts/03_cnn.py. */
  Model.prototype.predict = function (seq, percentPeptide, aaCutPosition) {
    var spec = this.spec;
    var raw = thermoVector(seq, percentPeptide, aaCutPosition, spec.tm);
    var thermo = raw.map(function (val, i) {
      return (val - spec.thermo_mu[i]) / spec.thermo_sd[i];
    });
    var x = oneHot(seq);

    var preds = [];
    var pool = new Float64Array(this.arch.seq_len);
    for (var m = 0; m < this.weights.length; m++) {
      var r = this.forwardOne(this.weights[m], x, thermo);
      preds.push(r.pred);
      for (var t = 0; t < pool.length; t++) pool[t] += r.pool[t] / this.weights.length;
    }
    var mean = preds.reduce(function (s, p) { return s + p; }, 0) / preds.length;
    var sd = Math.sqrt(
      preds.reduce(function (s, p) { return s + (p - mean) * (p - mean); }, 0) / preds.length
    );

    return {
      prediction: mean,
      perSeed: preds,
      seedSpread: sd,
      attention: Array.from(pool),
      thermoRaw: raw,
      gcCount: raw[0],
    };
  };

  /** Symmetric split-conformal interval, clipped to the target's (0, 1] support. */
  Model.prototype.interval = function (pred, alpha) {
    var q = this.spec.conformal_q[String(alpha)];
    return { low: Math.max(0, pred - q), high: Math.min(1, pred + q), halfWidth: q };
  };

  var VALID = { A: 1, C: 1, G: 1, T: 1 };

  /** Mirrors crispr/scoring.py validate_30mer, including the NGG requirement. */
  function validate30mer(seq) {
    var s = (seq || "").toUpperCase().replace(/[\s\n\r]/g, "");
    if (s.length !== 30) {
      throw new Error(
        "Expected a 30nt context sequence, got " + s.length + "nt. " +
        "The layout is 4nt upstream + 20nt protospacer + 3nt PAM + 3nt downstream."
      );
    }
    for (var i = 0; i < 30; i++) {
      if (!VALID[s[i]]) throw new Error("Unexpected character '" + s[i] + "'; only A, C, G, T are allowed.");
    }
    if (s.substring(25, 27) !== "GG") {
      throw new Error(
        "Positions 26-27 must be the 'GG' of the NGG PAM, found '" + s.substring(25, 27) +
        "'. The model was only ever trained on SpCas9 NGG sites."
      );
    }
    return s;
  }

  var api = { Model: Model, validate30mer: validate30mer, meltingTemp: meltingTemp, BASES: BASES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.CrisprInfer = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
