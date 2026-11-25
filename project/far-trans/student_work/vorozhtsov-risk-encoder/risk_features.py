"""Point-in-time price risk features for the FAR-Trans Encoder experiment.

Only positive closing prices and metadata dated strictly before ``asof`` are
used. The 90-calendar-day window is [asof-90, asof). Quantile ties are assigned
to the lower band: <= q33 is 0, <= q67 is 1, otherwise 2.
"""
from __future__ import annotations

import bisect
import csv
import math
from collections import OrderedDict, defaultdict
from datetime import date, timedelta
from pathlib import Path

METRIC_NAMES = (
    "volatility", "max_drawdown", "downside_rms", "worst_5_return",
    "current_drawdown", "share_drawdown_gt5", "vol_change",
)


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _sample_sd(values):
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1))


def _quantile(values, fraction):
    """Linear interpolation, equivalent to numpy.quantile(method='linear')."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    left = math.floor(index)
    right = math.ceil(index)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def calculate_metrics(observations, asof):
    """Calculate seven features from an already point-in-time price window.

    ``observations`` is a sequence of (ISO date, positive close) pairs. The
    coverage policy suppresses all features when fewer than 40 valid price
    changes exist, the last close is more than seven days old, or prices are
    constant. A five-interval return is unknown when every such interval
    contains a gap longer than seven days.
    """
    cutoff = date.fromisoformat(str(asof))
    lower = cutoff - timedelta(days=90)
    selected = [(date.fromisoformat(day), float(value)) for day, value in observations
                if lower.isoformat() <= day < cutoff.isoformat()
                and math.isfinite(float(value)) and float(value) > 0]
    selected.sort()
    prices = [p for _, p in selected]
    gaps = [(selected[i][0] - selected[i - 1][0]).days
            for i in range(1, len(selected))]
    returns = [math.log(prices[i] / prices[i - 1])
               for i in range(1, len(prices)) if 0 < gaps[i - 1] <= 7]
    last = selected[-1][0] if selected else None
    result = {
        "metrics": [None] * 7,
        "price_count": len(prices),
        "return_count": len(returns),
        "last_price_date": last.isoformat() if last else None,
        "window_start": lower.isoformat(),
        "window_end": cutoff.isoformat(),
        "unknown_reason": None,
    }
    if last is None or (cutoff - last).days > 7:
        result["unknown_reason"] = "missing_or_stale_price"
        return result
    if len(returns) < 40:
        result["unknown_reason"] = "fewer_than_40_valid_changes"
        return result
    if max(prices) == min(prices):
        result["unknown_reason"] = "constant_price_window"
        return result
    peak = prices[0]
    drawdowns = []
    for price in prices:
        peak = max(peak, price)
        drawdowns.append(1 - price / peak)
    worst_five = [prices[i] / prices[i - 5] - 1 for i in range(5, len(prices))
                  if all(0 < gap <= 7 for gap in gaps[i - 5:i])]
    volatility = _sample_sd(returns)
    result["metrics"] = [
        volatility,
        max(drawdowns),
        math.sqrt(math.fsum(min(r, 0) ** 2 for r in returns) / len(returns)),
        min(worst_five) if worst_five else None,
        drawdowns[-1],
        sum(d > 0.05 for d in drawdowns) / len(drawdowns),
        _sample_sd(returns[-20:]) - volatility,
    ]
    return result


class RiskStore:
    """Reusable as-of snapshots, historical purchase bands and audit details.

    Full feature snapshots use a small LRU. Historical band tables are retained
    because many investors share purchase dates; this avoids recomputing a
    cross-section for each transaction without retaining all price audit data.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        metadata = defaultdict(dict)
        self.same_date_metadata_ties = 0
        for row in _rows(self.data_dir / "asset_information.csv"):
            day = row["timestamp"][:10]
            if day in metadata[row["ISIN"]]:
                self.same_date_metadata_ties += 1
            # Mirror starter.latest_before: first row wins equal timestamps.
            metadata[row["ISIN"]].setdefault(day, row)
        self._metadata = {}
        for isin, by_day in metadata.items():
            days = sorted(by_day)
            self._metadata[isin] = (days, [by_day[d] for d in days])
        closes = defaultdict(dict)
        for row in _rows(self.data_dir / "close_prices.csv"):
            value = float(row["closePrice"])
            if not math.isfinite(value) or value <= 0:
                continue
            isin, day = row["ISIN"], row["timestamp"][:10]
            if day in closes[isin] and closes[isin][day] != value:
                raise ValueError(f"Conflicting positive closes for {isin} on {day}")
            closes[isin][day] = value
        self._closes = {}
        for isin, by_day in closes.items():
            days = sorted(by_day)
            self._closes[isin] = (days, [by_day[d] for d in days])
        self._purchases = defaultdict(list)
        self._transaction_rows = defaultdict(list)
        seen = set()
        self.excluded_duplicate_transactions = 0
        self.excluded_reconstructed_transactions = 0
        for row in _rows(self.data_dir / "transactions.csv"):
            key = tuple(sorted(row.items()))
            if key in seen:
                self.excluded_duplicate_transactions += 1
                continue
            seen.add(key)
            if int(row["transactionID"]) < 0:
                self.excluded_reconstructed_transactions += 1
                continue
            if row["transactionType"] != "Buy":
                continue
            item = {"date": row["timestamp"][:10], "isin": row["ISIN"],
                    "transaction_id": row["transactionID"]}
            self._purchases[row["customerID"]].append(item)
            self._transaction_rows[row["transactionID"]].append(item)
        for items in self._purchases.values():
            items.sort(key=lambda r: (r["date"], int(r["transaction_id"]), r["isin"]))
        self._feature_cache = OrderedDict()
        self._band_cache = {}

    @staticmethod
    def _day(asof):
        return asof.isoformat() if isinstance(asof, date) else str(asof)[:10]

    def _metadata_before(self, isin, asof):
        days, rows = self._metadata[isin]
        index = bisect.bisect_left(days, asof) - 1
        return rows[index] if index >= 0 else None

    def _window(self, isin, asof):
        days, prices = self._closes[isin]
        start = (date.fromisoformat(asof) - timedelta(days=90)).isoformat()
        left, right = bisect.bisect_left(days, start), bisect.bisect_left(days, asof)
        return list(zip(days[left:right], prices[left:right])), right

    def features(self, asof):
        asof = self._day(asof)
        if asof in self._feature_cache:
            self._feature_cache.move_to_end(asof)
            return self._feature_cache[asof]
        snapshot = {}
        for isin in sorted(set(self._metadata) & set(self._closes)):
            metadata = self._metadata_before(isin, asof)
            if metadata is None:
                continue
            window, historical_price_count = self._window(isin, asof)
            if not historical_price_count:
                continue
            record = calculate_metrics(window, asof)
            if record["last_price_date"] is None:
                record["last_price_date"] = self._closes[isin][0][historical_price_count - 1]
            record.update(category=metadata.get("assetCategory", ""),
                          metadata_date=metadata["timestamp"][:10], bands=[None] * 7)
            snapshot[isin] = record
        groups = defaultdict(list)
        for isin, record in snapshot.items():
            groups[record["category"]].append(isin)
        for members in groups.values():
            for metric in range(7):
                values = []
                for isin in members:
                    value = snapshot[isin]["metrics"][metric]
                    if value is not None:
                        values.append(-value if metric == 3 else value)
                if not values:
                    continue
                low, high = _quantile(values, 0.33), _quantile(values, 0.67)
                for isin in members:
                    value = snapshot[isin]["metrics"][metric]
                    if value is not None:
                        risk = -value if metric == 3 else value
                        snapshot[isin]["bands"][metric] = 0 if risk <= low else (1 if risk <= high else 2)
        self._band_cache[asof] = {isin: tuple(r["bands"]) for isin, r in snapshot.items()}
        self._feature_cache[asof] = snapshot
        if len(self._feature_cache) > 8:
            self._feature_cache.popitem(last=False)
        return snapshot

    def purchase_context(self, uid, cutoff, limit=40):
        cutoff = self._day(cutoff)
        cutoff_date = date.fromisoformat(cutoff)
        if limit <= 0:
            raise ValueError("Purchase history limit must be positive")
        if cutoff not in self._band_cache:
            self.features(cutoff)
        eligible = self._band_cache[cutoff]
        selected = [r for r in self._purchases.get(uid, ())
                    if r["date"] < cutoff and r["isin"] in eligible][-limit:]
        result = []
        for purchase in selected:
            day = purchase["date"]
            if day not in self._band_cache:
                self.features(day)
            bands = self._band_cache[day].get(purchase["isin"], (None,) * 7)
            result.append(dict(purchase,
                               age_days=(cutoff_date - date.fromisoformat(day)).days,
                               bands=list(bands)))
        return result

    def audit_transaction(self, transaction_id="8996071"):
        rows = self._transaction_rows.get(str(transaction_id), [])
        if len(rows) != 1:
            raise ValueError(f"Expected one observed purchase for transaction {transaction_id}, got {len(rows)}")
        purchase = rows[0]
        snapshot = self.features(purchase["date"])
        feature = snapshot.get(purchase["isin"])
        observations, _ = self._window(purchase["isin"], purchase["date"])
        peak, peak_date, maximum = 0.0, None, (-1.0, None, None, None, None)
        for day, price in observations:
            if price > peak:
                peak, peak_date = price, day
            drawdown = 1 - price / peak
            if drawdown > maximum[0]:
                maximum = (drawdown, peak_date, peak, day, price)
        return dict(purchase, feature=feature,
                    observations=[{"date": d, "close": p} for d, p in observations],
                    max_drawdown_check={"value": maximum[0], "peak_date": maximum[1],
                                        "peak_price": maximum[2], "trough_date": maximum[3],
                                        "trough_price": maximum[4]})
