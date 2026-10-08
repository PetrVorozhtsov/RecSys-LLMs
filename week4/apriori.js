/* Association-rule mining for invoice baskets. No build step or external library is needed. */
(function (root) {
  "use strict";

  function compareText(left, right) {
    return left < right ? -1 : left > right ? 1 : 0;
  }

  function key(items) {
    return JSON.stringify(items);
  }

  /**
   * Make one transaction per invoice. A product repeated on several invoice lines
   * counts once, because support measures baskets rather than purchased units.
   */
  function buildBaskets(rows) {
    if (!Array.isArray(rows)) {
      throw new TypeError("rows must be an array of invoice records");
    }

    const invoices = new Map();
    const descriptions = new Map();
    for (const row of rows) {
      if (!row) continue;
      const invoice = String(row.InvoiceNo == null ? "" : row.InvoiceNo).trim();
      const code = String(row.StockCode == null ? "" : row.StockCode).trim();
      const quantity = Number(row.Quantity == null ? 1 : row.Quantity);
      // C-prefixed invoices are cancellations; returns and missing identifiers
      // must not turn into positive shopping baskets.
      if (!invoice || /^C/i.test(invoice) || !code || !Number.isFinite(quantity) || quantity <= 0) {
        continue;
      }
      if (!invoices.has(invoice)) invoices.set(invoice, new Set());
      invoices.get(invoice).add(code);

      const description = String(row.Description == null ? "" : row.Description).trim();
      if (description) {
        const previous = descriptions.get(code);
        // Pick a stable label even when source rows have conflicting descriptions.
        if (!previous || description.length > previous.length ||
            (description.length === previous.length && compareText(description, previous) < 0)) {
          descriptions.set(code, description);
        }
      }
    }

    const baskets = [...invoices.entries()]
      .sort((left, right) => compareText(left[0], right[0]))
      .map(([, codes]) => [...codes].sort(compareText));
    // Arrays remain directly usable by mine() and JSON.stringify(); callers
    // loading raw records can also use this non-enumerable map for UI labels.
    Object.defineProperty(baskets, "descriptions", { value: descriptions });
    return baskets;
  }

  function validateOptions(options) {
    const minSupport = options.minSupport == null ? 0.01 : Number(options.minSupport);
    const minConfidence = options.minConfidence == null ? 0.3 : Number(options.minConfidence);
    const maxSize = options.maxSize == null ? 3 : Number(options.maxSize);
    if (!Number.isFinite(minSupport) || minSupport <= 0 || minSupport > 1) {
      throw new RangeError("minSupport must be greater than 0 and at most 1");
    }
    if (!Number.isFinite(minConfidence) || minConfidence < 0 || minConfidence > 1) {
      throw new RangeError("minConfidence must be between 0 and 1");
    }
    if (!Number.isSafeInteger(maxSize) || maxSize < 1) {
      throw new RangeError("maxSize must be a positive integer");
    }
    return { minSupport, minConfidence, maxSize };
  }

  function normaliseBasket(basket) {
    const source = Array.isArray(basket) ? basket : basket && basket.items;
    if (!source || typeof source[Symbol.iterator] !== "function") {
      throw new TypeError("each basket must be an item array or an object with items");
    }
    const unique = new Set();
    for (const item of source) {
      if (item == null) continue;
      const code = String(item).trim();
      if (code) unique.add(code);
    }
    return [...unique].sort(compareText);
  }

  // Vertical transaction-ID lists count every candidate exactly while avoiding
  // a full scan of all baskets for each joined Apriori candidate.
  function intersect(left, right, minCount) {
    const result = [];
    let i = 0;
    let j = 0;
    while (i < left.length && j < right.length) {
      if (result.length + Math.min(left.length - i, right.length - j) < minCount) {
        return null;
      }
      if (left[i] === right[j]) {
        result.push(left[i]);
        i++;
        j++;
      } else if (left[i] < right[j]) {
        i++;
      } else {
        j++;
      }
    }
    return result.length >= minCount ? result : null;
  }

  function samePrefix(left, right, prefixLength) {
    for (let i = 0; i < prefixLength; i++) {
      if (left[i] !== right[i]) return false;
    }
    return true;
  }

  function allSubsetsFrequent(items, previousLevel) {
    for (let drop = 0; drop < items.length; drop++) {
      const subset = items.filter((_, index) => index !== drop);
      if (!previousLevel.has(key(subset))) return false;
    }
    return true;
  }

  function ruleSort(left, right) {
    return right.lift - left.lift || right.support - left.support ||
      right.confidence - left.confidence ||
      compareText(key(left.antecedent), key(right.antecedent)) ||
      compareText(key(left.consequent), key(right.consequent));
  }

  /**
   * Full Apriori join-and-prune through maxSize. All counts use the exact number
   * of invoice baskets; directional rules are produced from every frequent set.
   */
  function mine(baskets, options = {}) {
    if (!Array.isArray(baskets)) throw new TypeError("baskets must be an array");
    const { minSupport, minConfidence, maxSize } = validateOptions(options);
    const N = baskets.length;
    const minCount = Math.max(1, Math.ceil(minSupport * N - 1e-10));
    const itemTids = new Map();
    for (let tid = 0; tid < N; tid++) {
      for (const item of normaliseBasket(baskets[tid])) {
        if (!itemTids.has(item)) itemTids.set(item, []);
        itemTids.get(item).push(tid);
      }
    }

    let candidateItemsets = itemTids.size;
    let level = [...itemTids.entries()]
      .filter(([, tids]) => tids.length >= minCount)
      .sort((left, right) => compareText(left[0], right[0]))
      .map(([item, tids]) => ({ items: [item], tids }));
    const frequent = [...level];

    for (let size = 2; size <= maxSize && level.length > 1; size++) {
      const previous = new Set(level.map((entry) => key(entry.items)));
      const next = [];
      for (let i = 0; i < level.length; i++) {
        for (let j = i + 1; j < level.length; j++) {
          const left = level[i];
          const right = level[j];
          if (!samePrefix(left.items, right.items, size - 2)) {
            // Sorted previous level keeps equal-prefix groups contiguous.
            break;
          }
          const joined = left.items.concat(right.items[right.items.length - 1]);
          if (!allSubsetsFrequent(joined, previous)) continue;
          candidateItemsets++;
          const tids = intersect(left.tids, right.tids, minCount);
          if (tids) next.push({ items: joined, tids });
        }
      }
      frequent.push(...next);
      level = next;
    }

    const frequencies = new Map(frequent.map(({ items, tids }) => [key(items), tids.length]));
    const itemsets = frequent.map(({ items, tids }) => ({
      items: [...items], count: tids.length, support: N ? tids.length / N : 0,
    }));
    const candidates = [];
    for (const { items, tids } of frequent) {
      if (items.length < 2) continue;
      const nAB = tids.length;
      // Every nonempty proper subset is a possible directional antecedent.
      function addSubsets(index, antecedent) {
        if (index === items.length) {
          if (!antecedent.length || antecedent.length === items.length) return;
          const selected = new Set(antecedent);
          const consequent = items.filter((item) => !selected.has(item));
          const nA = frequencies.get(key(antecedent));
          const nB = frequencies.get(key(consequent));
          if (!nA || !nB) throw new Error("Apriori subset count is missing");
          const confidence = nAB / nA;
          candidates.push({
            antecedent: [...antecedent], consequent, nA, nB, nAB,
            support: nAB / N,
            confidence,
            lift: (nAB * N) / (nA * nB),
          });
          return;
        }
        addSubsets(index + 1, antecedent);
        antecedent.push(items[index]);
        addSubsets(index + 1, antecedent);
        antecedent.pop();
      }
      addSubsets(0, []);
    }
    candidates.sort(ruleSort);
    const rules = candidates.filter((rule) =>
      rule.confidence + 1e-12 >= minConfidence && rule.nAB * N > rule.nA * rule.nB);
    const counts = {
      transactions: N,
      uniqueItems: itemTids.size,
      minCount,
      candidateItemsets,
      frequentItemsets: itemsets.length,
      candidateRules: candidates.length,
      retainedRules: rules.length,
      rejectedRules: candidates.length - rules.length,
    };
    return { N, itemsets, rules, candidates, counts };
  }

  const api = { buildBaskets, mine };
  root.AssociationRules = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
