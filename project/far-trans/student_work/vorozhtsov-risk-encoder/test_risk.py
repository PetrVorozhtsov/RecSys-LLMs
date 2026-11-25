"""Small independent checks of temporal features and shared risk scoring."""
import csv
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from risk_features import RiskStore, calculate_metrics
from risk_method import build_candidates, normalize_profile, score_candidates, statistical_profile


def observations(prices, start=date(2021, 1, 1)):
    return [((start + timedelta(days=i)).isoformat(), p) for i, p in enumerate(prices)]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PriceFeatureTests(unittest.TestCase):
    def test_monotone_known_log_returns(self):
        prices = [100 * math.exp(0.01 * i) for i in range(41)]
        result = calculate_metrics(observations(prices), "2021-02-11")
        self.assertEqual(result["return_count"], 40)
        expected = [0, 0, 0, math.exp(0.05) - 1, 0, 0, 0]
        for actual, target in zip(result["metrics"], expected):
            self.assertAlmostEqual(actual, target, places=12)

    def test_manual_drawdown_and_downside_denominator(self):
        prices = [100, 110, 99] + [110 if i % 2 else 100 for i in range(3, 42)]
        result = calculate_metrics(observations(prices), "2021-02-12")
        returns = [math.log(b / a) for a, b in zip(prices, prices[1:])]
        downside = math.sqrt(sum(r * r for r in returns if r < 0) / len(returns))
        self.assertAlmostEqual(result["metrics"][1], 0.10)
        self.assertAlmostEqual(result["metrics"][2], downside)
        # A positive or zero return still contributes to the denominator.
        self.assertLess(downside, math.sqrt(sum(r * r for r in returns if r < 0)
                                            / sum(r < 0 for r in returns)))

    def test_window_and_cutoff_exclude_future_price(self):
        points = observations([100 + (i % 7) for i in range(61)])
        baseline = calculate_metrics(points, "2021-03-03")
        poisoned = points + [("2021-03-03", 1000000), ("2021-04-01", 0.01),
                             ("2020-01-01", 1000000)]
        self.assertEqual(baseline, calculate_metrics(poisoned, "2021-03-03"))

    def test_long_gap_does_not_become_log_return(self):
        points = observations([100 + i for i in range(45)])
        # Last pair is nine calendar days apart and has a huge price jump.
        points.append(("2021-02-23", 1000000))
        result = calculate_metrics(points, "2021-02-24")
        self.assertEqual(result["return_count"], 44)
        valid = [math.log((101 + i) / (100 + i)) for i in range(44)]
        mean = sum(valid) / len(valid)
        sd = math.sqrt(sum((r - mean) ** 2 for r in valid) / (len(valid) - 1))
        self.assertAlmostEqual(result["metrics"][0], sd)
        self.assertAlmostEqual(result["metrics"][3], 144 / 139 - 1)

    def test_flat_short_and_stale_windows_are_unknown(self):
        flat = calculate_metrics(observations([100] * 50), "2021-02-20")
        short = calculate_metrics(observations(list(range(100, 140))), "2021-02-10")
        stale = calculate_metrics(observations(list(range(100, 150))), "2021-03-01")
        self.assertEqual(flat["unknown_reason"], "constant_price_window")
        self.assertEqual(short["unknown_reason"], "fewer_than_40_valid_changes")
        self.assertEqual(stale["unknown_reason"], "missing_or_stale_price")
        self.assertTrue(all(value is None for record in (flat, short, stale)
                            for value in record["metrics"]))

    def test_store_asof_metadata_and_purchases(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_csv(folder / "asset_information.csv", ["ISIN", "assetCategory", "timestamp"], [
                {"ISIN": "A", "assetCategory": "Stock", "timestamp": "2020-12-01"},
                {"ISIN": "A", "assetCategory": "FutureCategory", "timestamp": "2021-03-03"},
                {"ISIN": "B", "assetCategory": "Bond", "timestamp": "2021-03-03"},
                {"ISIN": "A", "assetCategory": "Conflict", "timestamp": "2020-12-01"},
            ])
            prices = [{"ISIN": isin, "timestamp": day, "closePrice": value}
                      for isin in ("A", "B")
                      for day, value in observations([100 + i % 5 for i in range(61)])]
            prices.append({"ISIN": "A", "timestamp": "2021-03-03", "closePrice": 1000000})
            write_csv(folder / "close_prices.csv", ["ISIN", "timestamp", "closePrice"], prices)
            good = {"customerID": "u", "ISIN": "A", "transactionID": "1",
                    "transactionType": "Buy", "timestamp": "2021-03-02"}
            transactions = [good, dict(good), dict(good, transactionID="-1"),
                            dict(good, transactionID="2", timestamp="2021-03-03"),
                            dict(good, transactionID="3", transactionType="Sell"),
                            dict(good, transactionID="1", customerID="another")]
            write_csv(folder / "transactions.csv", list(good), transactions)
            store = RiskStore(folder)
            snapshot = store.features("2021-03-03")
            self.assertEqual(set(snapshot), {"A"})
            self.assertEqual(snapshot["A"]["category"], "Stock")
            self.assertEqual(snapshot["A"]["last_price_date"], "2021-03-02")
            self.assertEqual(store.same_date_metadata_ties, 1)
            self.assertEqual(store.excluded_duplicate_transactions, 1)
            self.assertEqual(store.excluded_reconstructed_transactions, 1)
            purchases = store.purchase_context("u", "2021-03-03")
            self.assertEqual(len(purchases), 1)
            self.assertEqual(purchases[0]["age_days"], 1)
            self.assertEqual(len(store.purchase_context("another", "2021-03-03")), 1)

    def test_cross_section_respects_category_and_loss_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            metadata = [{"ISIN": isin, "assetCategory": category, "timestamp": "2020-12-01"}
                        for isin, category in (("A", "Stock"), ("B", "Stock"),
                                               ("C", "Stock"), ("D", "Bond"))]
            metadata.append({"ISIN": "E", "assetCategory": "Stock", "timestamp": "2021-03-03"})
            write_csv(folder / "asset_information.csv", ["ISIN", "assetCategory", "timestamp"], metadata)
            prices = []
            for isin, swing in (("A", 0.01), ("B", 0.02), ("C", 0.04),
                                ("D", 0.80), ("E", 0.90)):
                for day, value in observations([100 * (1 + swing * (i % 2)) for i in range(61)]):
                    prices.append({"ISIN": isin, "timestamp": day, "closePrice": value})
            write_csv(folder / "close_prices.csv", ["ISIN", "timestamp", "closePrice"], prices)
            write_csv(folder / "transactions.csv", ["customerID", "ISIN", "transactionID",
                                                     "transactionType", "timestamp"], [])
            snapshot = RiskStore(folder).features("2021-03-03")
            self.assertNotIn("E", snapshot)
            self.assertEqual([snapshot[i]["bands"][0] for i in ("A", "B", "C")], [0, 1, 2])
            self.assertEqual([snapshot[i]["bands"][3] for i in ("A", "B", "C")], [0, 1, 2])
            self.assertEqual(snapshot["D"]["bands"][0], 0)


class RiskMethodTests(unittest.TestCase):
    def test_half_life_and_unknown_dimensions(self):
        purchases = [{"age_days": 0, "bands": [0, None, 0, 0, 0, 0, 0]},
                     {"age_days": 90, "bands": [2, None, 2, 2, 2, 2, 2]}]
        profile = statistical_profile(purchases)
        self.assertEqual(profile[0], [2 / 3, 0, 1 / 3])
        self.assertEqual(profile[1], [1 / 3] * 3)
        self.assertEqual(statistical_profile([]), [[1 / 3] * 3 for _ in range(7)])

    def test_self_exclusion_neighbor_ties_and_popular_fill(self):
        universe = [f"A{i:02}" for i in range(15)]
        history = {"target": {"A00": 100, "A09": 1},
                   "neighbor_b": {"A00": 1, "A02": 1},
                   "neighbor_a": {"A00": 1, "A01": 1},
                   "unrelated": {"A05": 1}}
        popular = {isin: i for i, isin in enumerate(universe)}
        candidate = build_candidates("target", history, popular, universe, max_neighbors=1)
        self.assertEqual(candidate[:2], ["A00", "A01"])
        self.assertEqual(candidate[2:5], ["A14", "A13", "A12"])
        self.assertEqual(len(candidate), 10)
        self.assertEqual(candidate, build_candidates("target", dict(reversed(list(history.items()))),
                                                    popular, list(reversed(universe)), max_neighbors=1))

    def test_popularity_does_not_expand_sufficient_candidate_pool(self):
        universe = [f"A{i:02}" for i in range(30)]
        history = {"target": {"A00": 1}, "neighbor": {isin: 1 for isin in universe[:12]}}
        result = build_candidates("target", history, {i: 100 for i in universe[12:]}, universe)
        self.assertEqual(result, universe[:12])

    def test_shared_scoring_moves_relevant_band_and_keeps_uniform_order(self):
        candidates = [f"A{i:02}" for i in range(50)]
        features = {isin: {"bands": [0] * 7} for isin in candidates}
        features["A10"] = {"bands": [2] * 7}
        uniform = [[1 / 3] * 3 for _ in range(7)]
        self.assertEqual(score_candidates(candidates, uniform, features), candidates[:10])
        ranked = score_candidates(candidates, [[0, 0, 1] for _ in range(7)], features)
        self.assertEqual(ranked[0], "A10")
        self.assertEqual(len(set(ranked)), 10)

    def test_profile_rejects_bad_values(self):
        for bad in [True, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                normalize_profile([[bad, 1, 1] for _ in range(7)])
        with self.assertRaises(ValueError):
            normalize_profile([[0, 0, 0] for _ in range(7)])
        self.assertEqual(normalize_profile([[2, 0, 2] for _ in range(7)]),
                         [[0.5, 0, 0.5] for _ in range(7)])


if __name__ == "__main__":
    unittest.main()
