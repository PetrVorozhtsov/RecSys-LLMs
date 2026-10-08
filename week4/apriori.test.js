"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const { buildBaskets, mine } = require("./apriori.js");

function ruleFor(results, antecedent, consequent) {
  return results.candidates.find((rule) =>
    JSON.stringify(rule.antecedent) === JSON.stringify(antecedent) &&
    JSON.stringify(rule.consequent) === JSON.stringify(consequent));
}

test("buildBaskets groups invoice lines and excludes returns and cancellations", () => {
  const rows = [
    { InvoiceNo: "2", StockCode: "B", Description: "Blue cup", Quantity: 2 },
    { InvoiceNo: "1", StockCode: "A", Description: "Small mug", Quantity: 1 },
    { InvoiceNo: "1", StockCode: "A", Description: "Small mug", Quantity: 3 },
    { InvoiceNo: "1", StockCode: "B", Description: "Blue cup", Quantity: 1 },
    { InvoiceNo: "C3", StockCode: "Z", Description: "Cancelled", Quantity: 1 },
    { InvoiceNo: "4", StockCode: "Z", Description: "Returned", Quantity: -1 },
    { InvoiceNo: "5", StockCode: "", Description: "Missing code", Quantity: 1 },
    { InvoiceNo: "6", StockCode: "Q", Description: "No quantity", Quantity: "not a number" },
  ];
  const baskets = buildBaskets(rows);
  assert.deepEqual(baskets, [["A", "B"], ["B"]]);
  assert.equal(baskets.descriptions.get("A"), "Small mug");
  assert.equal(baskets.descriptions.get("B"), "Blue cup");
});

test("Apriori joins and prunes through size 3 with exact directional counts", () => {
  const baskets = [
    ["A", "B", "C"], ["A", "B", "C"], ["A", "B"], ["A"],
    ["B"], ["C"], ["D"], ["A", "B", "C"],
  ];
  const result = mine(baskets, { minSupport: 0.25, minConfidence: 0.3, maxSize: 3 });
  assert.equal(result.N, 8);
  assert.deepEqual(result.itemsets.find((set) => set.items.join() === "A,B,C"), {
    items: ["A", "B", "C"], count: 3, support: 3 / 8,
  });
  assert.equal(result.itemsets.length, 7); // A, B, C, AB, AC, BC, ABC
  assert.equal(result.counts.candidateRules, 12); // six pair directions + six triple directions
  const forward = ruleFor(result, ["A"], ["B", "C"]);
  assert.deepEqual({ nA: forward.nA, nB: forward.nB, nAB: forward.nAB },
    { nA: 5, nB: 3, nAB: 3 });
  assert.equal(forward.support, 3 / 8);
  assert.equal(forward.confidence, 3 / 5);
  assert.equal(forward.lift, 8 / 5);
  assert.equal(result.rules.length, result.candidates.length);
});

test("the lift filter rejects a misleading high-confidence rule", () => {
  const baskets = [
    ["A", "B"], ["A", "B"], ["A", "B"], ["A", "B"], ["A"],
    ["B"], ["B"], ["B"], ["B"], ["C"],
  ];
  const result = mine(baskets, { minSupport: 0.2, minConfidence: 0.4 });
  const forward = ruleFor(result, ["A"], ["B"]);
  const reverse = ruleFor(result, ["B"], ["A"]);
  assert.deepEqual({ nA: forward.nA, nB: forward.nB, nAB: forward.nAB },
    { nA: 5, nB: 8, nAB: 4 });
  assert.equal(forward.support, 0.4);
  assert.equal(forward.confidence, 0.8);
  assert.equal(reverse.confidence, 0.5);
  assert.equal(forward.lift, 1); // B is so prevalent that A adds no lift.
  assert.equal(result.rules.some((rule) => rule.antecedent[0] === "A" && rule.consequent[0] === "B"), false);
  assert.equal(result.counts.rejectedRules > 0, true);
});

test("duplicate items never inflate support and thresholds control candidate levels", () => {
  const baskets = [
    { items: ["A", "A", "B", "C"] },
    ["A", "B", "C"], ["A", "B"], ["B"],
  ];
  const low = mine(baskets, { minSupport: 0.5, minConfidence: 0, maxSize: 3 });
  assert.equal(low.itemsets.find((set) => set.items.join() === "A").count, 3);
  assert.equal(low.itemsets.find((set) => set.items.join() === "A,B,C").count, 2);
  const capped = mine(baskets, { minSupport: 0.5, minConfidence: 0, maxSize: 2 });
  assert.equal(capped.itemsets.some((set) => set.items.length === 3), false);
  const high = mine(baskets, { minSupport: 0.75, minConfidence: 0, maxSize: 3 });
  assert.equal(high.itemsets.some((set) => set.items.join() === "A,B,C"), false);
  assert.equal(high.counts.frequentItemsets < low.counts.frequentItemsets, true);
});

test("Apriori joins higher levels only when every required subset is frequent", () => {
  const baskets = [
    ["A", "B", "C", "D"], ["A", "B", "C", "D"], ["A", "B", "C", "D"],
    ["A", "B"], ["C", "D"],
  ];
  const low = mine(baskets, { minSupport: 0.6, minConfidence: 0.5, maxSize: 4 });
  assert.equal(low.itemsets.find((set) => set.items.join() === "A,B,C,D").count, 3);
  const high = mine(baskets, { minSupport: 0.8, minConfidence: 0.5, maxSize: 4 });
  assert.equal(high.itemsets.some((set) => set.items.length > 2), false);
  assert.equal(high.counts.candidateItemsets, 10); // Four singles + six pair candidates; no triples.
});

test("empty data, invalid options, deterministic order, and browser global", () => {
  const empty = mine([], { minSupport: 0.01 });
  assert.equal(empty.N, 0);
  assert.deepEqual(empty.rules, []);
  assert.equal(empty.counts.transactions, 0);
  assert.throws(() => mine([["A"]], { minSupport: 0 }), RangeError);
  assert.throws(() => mine([["A"]], { minConfidence: 1.1 }), RangeError);
  assert.throws(() => mine([["A"]], { maxSize: 0 }), RangeError);

  const baskets = [["B", "A"], ["C", "A"], ["A", "B", "C"], ["A", "B"]];
  assert.deepEqual(mine(baskets), mine([...baskets].reverse()));

  const sandbox = { globalThis: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "apriori.js"), "utf8"), sandbox);
  assert.equal(typeof sandbox.globalThis.AssociationRules.buildBaskets, "function");
  assert.equal(typeof sandbox.globalThis.AssociationRules.mine, "function");
});
