/* Minimal inline-SVG charts for the static dashboard.
 *
 * Hand-rolled rather than pulled from a CDN so the built page is a single file
 * that works offline. Only the four chart types the dashboard actually uses are
 * implemented; every one takes explicit axis titles because an unlabelled axis
 * is worse than no chart.
 */

(function (root) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var C = {
    ink: "#111418", body: "#374151", muted: "#6b7280", faint: "#9ca3af",
    line: "#e5e7eb", accent: "#2563eb", warm: "#d97706", good: "#059669",
    bad: "#dc2626", pale: "#cbd5e1",
  };

  function el(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }

  function svgRoot(w, h) {
    var s = el("svg", { viewBox: "0 0 " + w + " " + h, width: w, height: h });
    s.style.maxWidth = "100%";
    return s;
  }

  // ---- shared tooltip ----

  function tipEl() {
    var t = document.getElementById("tip");
    if (!t) {
      t = document.createElement("div");
      t.id = "tip";
      document.body.appendChild(t);
    }
    return t;
  }

  function bindTip(node, html) {
    node.addEventListener("mousemove", function (e) {
      var t = tipEl();
      t.innerHTML = html;
      t.style.opacity = 1;
      var x = e.clientX + 14, y = e.clientY + 14;
      var r = t.getBoundingClientRect();
      if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 14;
      if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 14;
      t.style.left = x + "px";
      t.style.top = y + "px";
    });
    node.addEventListener("mouseleave", function () { tipEl().style.opacity = 0; });
  }

  // ---- axis helpers ----

  function ticks(lo, hi, n) {
    var span = hi - lo;
    if (span <= 0) return [lo];
    var raw = span / n;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      out.push(Math.abs(v) < 1e-12 ? 0 : v);
    }
    return out;
  }

  function fmt(v, d) { return v.toFixed(d == null ? 2 : d); }

  function axisTitle(s, text, x, y, rotate) {
    if (!text) return;
    var a = { x: x, y: y, "font-size": 11.5, fill: C.muted, "text-anchor": "middle" };
    if (rotate) a.transform = "rotate(-90 " + x + " " + y + ")";
    s.appendChild(el("text", a, text));
  }

  function legend(s, items, x, y) {
    var dx = 0;
    items.forEach(function (it) {
      s.appendChild(el("rect", { x: x + dx, y: y - 8, width: 9, height: 9, rx: 2, fill: it.color }));
      var t = el("text", { x: x + dx + 14, y: y, "font-size": 12, fill: C.body }, it.name);
      s.appendChild(t);
      dx += 24 + it.name.length * 6.6;
    });
  }

  // ---- grouped / single vertical bars ----

  function bars(node, o) {
    node.innerHTML = "";
    var w = o.width || 720, h = o.height || 340;
    var m = { t: o.series.length > 1 ? 34 : 14, r: 18, b: 56, l: 62 };
    var pw = w - m.l - m.r, ph = h - m.t - m.b;
    var s = svgRoot(w, h);

    var all = [];
    o.series.forEach(function (se) { all = all.concat(se.data); });
    (o.refLines || []).forEach(function (r) { all.push(r.value); });
    var hi = o.yMax != null ? o.yMax : Math.max.apply(null, all) * 1.18;
    var lo = o.yMin != null ? o.yMin : Math.min(0, Math.min.apply(null, all) * 1.12);
    var Y = function (v) { return m.t + ph - ((v - lo) / (hi - lo)) * ph; };

    ticks(lo, hi, 5).forEach(function (v) {
      s.appendChild(el("line", { x1: m.l, y1: Y(v), x2: m.l + pw, y2: Y(v), stroke: C.line }));
      s.appendChild(el("text", {
        x: m.l - 9, y: Y(v) + 4, "font-size": 11, fill: C.muted, "text-anchor": "end",
      }, fmt(v, o.yDecimals)));
    });

    var n = o.categories.length, k = o.series.length;
    var slot = pw / n, bw = (slot * 0.66) / k;

    o.series.forEach(function (se, si) {
      se.data.forEach(function (v, i) {
        if (v == null) return;
        var x = m.l + slot * i + slot * 0.17 + bw * si;
        var y = Y(Math.max(v, lo)), hh = Math.abs(Y(v) - Y(0));
        var r = el("rect", { x: x, y: Math.min(y, Y(0)), width: bw, height: Math.max(hh, 1), fill: se.color, rx: 2 });
        var cat = (o.tipCategories || o.categories)[i];
        bindTip(r, "<span class='k'>" + cat + "</span><br>" + se.name + ": <b>" + fmt(v, o.tipDecimals || 4) + "</b>");
        s.appendChild(r);
        if (o.showValues) {
          s.appendChild(el("text", {
            x: x + bw / 2, y: Y(v) - 6, "font-size": 11, fill: C.body, "text-anchor": "middle",
          }, fmt(v, o.valueDecimals || 3)));
        }
        if (se.error && se.error[i] != null) {
          var e = se.error[i], cx = x + bw / 2;
          s.appendChild(el("line", { x1: cx, y1: Y(v - e), x2: cx, y2: Y(v + e), stroke: C.muted }));
          s.appendChild(el("line", { x1: cx - 4, y1: Y(v + e), x2: cx + 4, y2: Y(v + e), stroke: C.muted }));
          s.appendChild(el("line", { x1: cx - 4, y1: Y(v - e), x2: cx + 4, y2: Y(v - e), stroke: C.muted }));
        }
      });
    });

    (o.refLines || []).forEach(function (r) {
      s.appendChild(el("line", {
        x1: m.l, y1: Y(r.value), x2: m.l + pw, y2: Y(r.value),
        stroke: r.color || C.faint, "stroke-dasharray": "4 3",
      }));
      s.appendChild(el("text", {
        x: m.l + pw, y: Y(r.value) - 5, "font-size": 10.5,
        fill: r.color || C.faint, "text-anchor": "end",
      }, r.label));
    });

    o.categories.forEach(function (c, i) {
      s.appendChild(el("text", {
        x: m.l + slot * i + slot / 2, y: m.t + ph + 17,
        "font-size": o.tickSize || 11.5, fill: C.body, "text-anchor": "middle",
      }, c));
    });

    s.appendChild(el("line", { x1: m.l, y1: m.t + ph, x2: m.l + pw, y2: m.t + ph, stroke: C.line }));
    axisTitle(s, o.xTitle, m.l + pw / 2, h - 10);
    axisTitle(s, o.yTitle, 15, m.t + ph / 2, true);
    if (k > 1) legend(s, o.series, m.l, 14);
    node.appendChild(s);
  }

  // ---- horizontal bars, for long category labels ----

  function hbars(node, o) {
    node.innerHTML = "";
    var rows = o.categories.length;
    var w = o.width || 720, rh = o.rowHeight || 21;
    var m = { t: 12, r: 22, b: 50, l: o.labelWidth || 210 };
    var ph = rows * rh, h = m.t + ph + m.b;
    var pw = w - m.l - m.r;
    var s = svgRoot(w, h);

    var all = o.values.slice();
    (o.refLines || []).forEach(function (r) { all.push(r.value); });
    var hi = Math.max.apply(null, all), lo = Math.min(0, Math.min.apply(null, all));
    var pad = (hi - lo) * 0.12;
    hi += pad; lo -= (lo < 0 ? pad : 0);
    var X = function (v) { return m.l + ((v - lo) / (hi - lo)) * pw; };

    ticks(lo, hi, 5).forEach(function (v) {
      s.appendChild(el("line", { x1: X(v), y1: m.t, x2: X(v), y2: m.t + ph, stroke: C.line }));
      s.appendChild(el("text", {
        x: X(v), y: m.t + ph + 16, "font-size": 11, fill: C.muted, "text-anchor": "middle",
      }, fmt(v, o.tickDecimals == null ? 2 : o.tickDecimals)));
    });

    o.categories.forEach(function (cat, i) {
      var v = o.values[i], y = m.t + i * rh;
      var x0 = X(Math.min(0, v)), x1 = X(Math.max(0, v));
      var r = el("rect", {
        x: x0, y: y + 3, width: Math.max(x1 - x0, 1), height: rh - 6,
        fill: (o.colors && o.colors[i]) || C.accent, rx: 2,
      });
      bindTip(r, "<span class='k'>" + cat + "</span><br><b>" + fmt(v, o.tipDecimals || 4) + "</b>" +
        (o.tipExtra ? "<br>" + o.tipExtra[i] : ""));
      s.appendChild(r);
      s.appendChild(el("text", {
        x: m.l - 9, y: y + rh / 2 + 4, "font-size": 11.5, fill: C.body, "text-anchor": "end",
      }, cat));
    });

    (o.refLines || []).forEach(function (r) {
      s.appendChild(el("line", {
        x1: X(r.value), y1: m.t, x2: X(r.value), y2: m.t + ph,
        stroke: r.color || C.warm, "stroke-dasharray": "4 3",
      }));
      s.appendChild(el("text", {
        x: X(r.value) + 5, y: m.t + 10, "font-size": 10.5, fill: r.color || C.warm,
      }, r.label));
    });

    if (lo < 0) s.appendChild(el("line", { x1: X(0), y1: m.t, x2: X(0), y2: m.t + ph, stroke: C.ink }));
    axisTitle(s, o.xTitle, m.l + pw / 2, h - 10);
    node.appendChild(s);
  }

  // ---- multi-series lines ----

  function lines(node, o) {
    node.innerHTML = "";
    var w = o.width || 720, h = o.height || 340;
    var m = { t: 32, r: 18, b: 56, l: 62 };
    var pw = w - m.l - m.r, ph = h - m.t - m.b;
    var s = svgRoot(w, h);

    var xs = o.x, all = [];
    o.series.forEach(function (se) { all = all.concat(se.data); });
    var yhi = o.yMax != null ? o.yMax : Math.max.apply(null, all) * 1.06;
    var ylo = o.yMin != null ? o.yMin : Math.min.apply(null, all) * 0.94;
    var xhi = Math.max.apply(null, xs), xlo = Math.min.apply(null, xs);
    var X = function (v) { return m.l + ((v - xlo) / (xhi - xlo)) * pw; };
    var Y = function (v) { return m.t + ph - ((v - ylo) / (yhi - ylo)) * ph; };

    ticks(ylo, yhi, 5).forEach(function (v) {
      s.appendChild(el("line", { x1: m.l, y1: Y(v), x2: m.l + pw, y2: Y(v), stroke: C.line }));
      s.appendChild(el("text", {
        x: m.l - 9, y: Y(v) + 4, "font-size": 11, fill: C.muted, "text-anchor": "end",
      }, fmt(v, o.yDecimals)));
    });
    ticks(xlo, xhi, 5).forEach(function (v) {
      s.appendChild(el("text", {
        x: X(v), y: m.t + ph + 17, "font-size": 11, fill: C.muted, "text-anchor": "middle",
      }, fmt(v, o.xDecimals)));
    });

    o.series.forEach(function (se) {
      var d = se.data.map(function (v, i) { return (i ? "L" : "M") + X(xs[i]) + " " + Y(v); }).join(" ");
      s.appendChild(el("path", {
        d: d, fill: "none", stroke: se.color, "stroke-width": se.width || 2,
        "stroke-dasharray": se.dash || "none",
        "stroke-linejoin": "round", "stroke-linecap": "round",
      }));
      if (se.markers !== false) {
        se.data.forEach(function (v, i) {
          var c = el("circle", { cx: X(xs[i]), cy: Y(v), r: 3, fill: se.color });
          bindTip(c, se.name + "<br><span class='k'>" + o.xTitle + "</span> " + fmt(xs[i], 2) +
            "<br><b>" + fmt(v, 4) + "</b>");
          s.appendChild(c);
        });
      }
    });

    s.appendChild(el("line", { x1: m.l, y1: m.t + ph, x2: m.l + pw, y2: m.t + ph, stroke: C.line }));
    axisTitle(s, o.xTitle, m.l + pw / 2, h - 10);
    axisTitle(s, o.yTitle, 15, m.t + ph / 2, true);
    legend(s, o.series, m.l, 14);
    node.appendChild(s);
  }

  // ---- scatter ----

  function scatter(node, o) {
    node.innerHTML = "";
    var w = o.width || 560, h = o.height || 400;
    var m = { t: 14, r: 18, b: 56, l: 62 };
    var pw = w - m.l - m.r, ph = h - m.t - m.b;
    var s = svgRoot(w, h);
    var X = function (v) { return m.l + ((v - o.xMin) / (o.xMax - o.xMin)) * pw; };
    var Y = function (v) { return m.t + ph - ((v - o.yMin) / (o.yMax - o.yMin)) * ph; };

    ticks(o.yMin, o.yMax, 5).forEach(function (v) {
      s.appendChild(el("line", { x1: m.l, y1: Y(v), x2: m.l + pw, y2: Y(v), stroke: C.line }));
      s.appendChild(el("text", { x: m.l - 9, y: Y(v) + 4, "font-size": 11, fill: C.muted, "text-anchor": "end" }, fmt(v, 1)));
    });
    ticks(o.xMin, o.xMax, 5).forEach(function (v) {
      s.appendChild(el("text", { x: X(v), y: m.t + ph + 17, "font-size": 11, fill: C.muted, "text-anchor": "middle" }, fmt(v, 1)));
    });

    s.appendChild(el("line", {
      x1: X(o.xMin), y1: Y(o.yMin), x2: X(o.xMax), y2: Y(o.yMax),
      stroke: C.faint, "stroke-dasharray": "4 3",
    }));

    var g = el("g", {});
    o.points.forEach(function (p) {
      g.appendChild(el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 2, fill: C.accent, opacity: 0.3 }));
    });
    s.appendChild(g);

    axisTitle(s, o.xTitle, m.l + pw / 2, h - 10);
    axisTitle(s, o.yTitle, 15, m.t + ph / 2, true);
    node.appendChild(s);
  }

  root.Charts = { bars: bars, hbars: hbars, lines: lines, scatter: scatter, C: C };
})(typeof globalThis !== "undefined" ? globalThis : this);
