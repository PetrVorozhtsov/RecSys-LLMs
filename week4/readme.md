# Week 4 — Association Rules on Online Retail

I implemented the association-rules assignment from the Week 4 lecture on the
[UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).
The former MovieLens Two-Tower example in this folder did not match that task.

## Run the demo

From this directory, start a static server and open `http://localhost:8000/`:

```bash
python -m http.server 8000
```

The page loads the committed `data/baskets.json` file, so it does not require
Python packages, a build step, or an external service at runtime. Change the
minimum support and confidence controls and press **Run rules**. The retained
list contains only rules with lift above 1; the rejected-candidates view shows
the evidence for rules that fail a threshold or the lift criterion. Select a
rule to see its transaction counts and all three metrics. Search accepts a
product name, StockCode, or an exact directed pair such as `22386 → 85099B`.
**Run tests** checks
the calculations on a small, known basket example in the browser.

## Data and preprocessing

- Source: Chen, D. (2015), *Online Retail*, UCI Machine Learning Repository,
  [DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33), CC BY 4.0.
- Each distinct `InvoiceNo` is a transaction. `StockCode` is the item ID and
  the most frequent `Description` for that code is its display label.
- Duplicate invoice–item lines count once. I excluded C-prefixed cancellation
  invoices, nonpositive quantity or price, and rows without a product code,
  description, or invoice date. This leaves **530,104 rows**, **19,960 baskets**,
  and **3,922 products** from the original 541,909 rows. The JSON records the
  exact exclusion counts, date range, original file SHA-256 values, and labels.
- To rebuild the JSON from the official archive, run
  `python prepare_data.py`. This preparation step requires `openpyxl`.
  `python prepare_data.py --source "Online Retail.xlsx"` also works with a
  downloaded workbook.

The committed data file contains product codes, names, basket membership, and
invoice dates for temporal analysis. It does not include customer identifiers.

## Mining and interpretation

I used Apriori candidate generation and the subset-pruning property to mine
frequent itemsets and their directional rules. The interface permits itemsets
up to size 10; at the lowest offered support of 0.5%, the largest frequent
itemset in these data has size 6. The default minimum support is **1%** (at least 200 of
19,960 invoices), and the default minimum confidence is **30%**. A 1% floor
avoids relying on a handful of accidental co-purchases, while 30% asks that
the consequence occur in a substantial share of baskets containing the
antecedent. Both controls are adjustable because a single threshold is not
universally suitable. For a rule `A → B`:

```text
support(A → B)    = count(A ∪ B) / N
confidence(A → B) = count(A ∪ B) / count(A)
lift(A → B)       = confidence(A → B) / (count(B) / N)
```

`N` includes all cleaned baskets, including one-product invoices. I retain
rules with support and confidence at or above the chosen thresholds and
**lift > 1** for further analysis. The rejected view is deliberately drawn
from candidate rules *before* the confidence/lift filter, so an apparently
attractive but weak rule can be examined without being called a retained rule.

Association does not establish that a bundle or cross-sell intervention will
increase sales. The report compares a promising and a rejected rule using
counts and metrics, checks a later period, and proposes a controlled test.

## Files and verification

- `prepare_data.py` creates the deterministic basket JSON from the original
  UCI workbook.
- `apriori.js` implements basket construction, Apriori itemsets, directional
  rules, exact counts, and metric calculations.
- `apriori.test.js` tests a known example and edge cases independently of the
  UI. Run with `node apriori.test.js`.
- `index.html`, `style.css`, and `app.js` provide the interactive explorer.
- `evaluate.js` reruns the miner at several threshold settings, independently
  counts two business examples across early/later invoices, checks them
  against the miner, and writes `evaluation-results.json`. Run with
  `node evaluate.js`.
- `report_assets/` includes the three real browser screenshots, the optional
  Playwright capture script, and the PDF builder. The PDF documents the
  measured run and my interpretation.

Project: [Petr Vorozhtsov's fork](https://github.com/PetrVorozhtsov/RecSys-LLMs/tree/main/week4).
