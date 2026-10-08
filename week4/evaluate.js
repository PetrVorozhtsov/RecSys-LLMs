"use strict";

// Reproduce the report's measurements from the committed UCI-derived baskets.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { mine } = require("./apriori.js");

const root = __dirname;
const data = JSON.parse(fs.readFileSync(path.join(root, "data", "baskets.json"), "utf8"));
const settings = { minSupport: 0.01, minConfidence: 0.30, maxSize: 10 };
const cutoff = "2011-09-01";
const examples = {
  promising: { antecedent: ["22386"], consequent: ["85099B"] },
  rejected: { antecedent: ["22423"], consequent: ["85099B"] },
};

function summary(baskets, minSupport, minConfidence) {
  const result = mine(baskets, { minSupport, minConfidence, maxSize: settings.maxSize });
  return result.counts;
}

function measurePair(baskets, A, B) {
  let nA = 0;
  let nB = 0;
  let nAB = 0;
  for (const basket of baskets) {
    const hasA = basket.includes(A);
    const hasB = basket.includes(B);
    if (hasA) nA++;
    if (hasB) nB++;
    if (hasA && hasB) nAB++;
  }
  const N = baskets.length;
  return {
    N, nA, nB, nAB,
    support: nAB / N,
    confidence: nAB / nA,
    lift: (nAB * N) / (nA * nB),
    baselineConsequence: nB / N,
  };
}

const result = mine(data.baskets, settings);
const early = [];
const late = [];
for (let index = 0; index < data.baskets.length; index++) {
  (data.dates[index] < cutoff ? early : late).push(data.baskets[index]);
}
assert.equal(result.N, data.preprocessing.basket_count);
assert.equal(early.length + late.length, result.N);

const reported = {};
for (const [kind, pair] of Object.entries(examples)) {
  const { antecedent, consequent } = pair;
  const candidate = result.candidates.find(rule =>
    JSON.stringify(rule.antecedent) === JSON.stringify(antecedent) &&
    JSON.stringify(rule.consequent) === JSON.stringify(consequent));
  assert.ok(candidate, `${kind} rule must occur in the mined candidate list`);
  const full = measurePair(data.baskets, antecedent[0], consequent[0]);
  assert.deepEqual([candidate.nA, candidate.nB, candidate.nAB], [full.nA, full.nB, full.nAB]);
  const retained = result.rules.includes(candidate);
  assert.equal(retained, kind === "promising");
  reported[kind] = {
    ...pair,
    antecedentLabel: data.items[antecedent[0]],
    consequentLabel: data.items[consequent[0]],
    retained,
    full,
    early: measurePair(early, antecedent[0], consequent[0]),
    late: measurePair(late, antecedent[0], consequent[0]),
  };
}

const output = {
  data: data.preprocessing,
  source: data.source,
  settings,
  defaultRun: result.counts,
  thresholdSweep: [
    { name: "Explore rare", minSupport: 0.005, minConfidence: 0.20, counts: summary(data.baskets, 0.005, 0.20) },
    { name: "Balanced", minSupport: 0.01, minConfidence: 0.30, counts: result.counts },
    { name: "Higher support", minSupport: 0.02, minConfidence: 0.30, counts: summary(data.baskets, 0.02, 0.30) },
    { name: "Higher confidence", minSupport: 0.01, minConfidence: 0.50, counts: summary(data.baskets, 0.01, 0.50) },
    { name: "High evidence", minSupport: 0.02, minConfidence: 0.50, counts: summary(data.baskets, 0.02, 0.50) },
  ],
  examples: reported,
  reversePromising: measurePair(data.baskets, "85099B", "22386"),
  temporalCutoff: cutoff,
};
const destination = path.join(root, "evaluation-results.json");
fs.writeFileSync(destination, JSON.stringify(output, null, 2) + "\n");
console.log(`Wrote ${destination}`);
console.log(JSON.stringify({ defaultRun: output.defaultRun, thresholdSweep: output.thresholdSweep }, null, 2));
