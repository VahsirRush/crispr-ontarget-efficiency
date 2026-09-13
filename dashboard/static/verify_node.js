/* Harness used by scripts/10_build_static_dashboard.py.
 *
 * Reads {model, cases} on stdin, runs the browser inference path over the cases,
 * and writes the predictions to stdout so Python can diff them against PyTorch.
 * Kept separate from infer.js so nothing test-only ships in the built page.
 */

const fs = require("fs");
const { Model } = require("./infer.js");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const model = new Model(input.model);

const out = input.cases.map((c) => {
  const r = model.predict(c.seq, c.pp, c.cut);
  return { pred: r.prediction, thermo: r.thermoRaw, attention: r.attention };
});

process.stdout.write(JSON.stringify(out));
