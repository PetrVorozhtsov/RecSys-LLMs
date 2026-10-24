#!/usr/bin/env python3
"""Fast, leakage-safe SWEET.TV exam pipeline.

The script is deliberately standard-library first so it can run directly after
the Kaggle archive is unpacked. It joins viewing intervals to labelled schedule
intervals, evaluates popularity/content/association recommenders on a temporal
hold-out, and writes the Kaggle submission plus audit artefacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path


VIEW_FILE = "dataset11-30.csv"
SCHEDULE_FILES = {
    "train": "export_arh_11-20-final.csv",
    "valid": "export_arh_21-30-final.csv",
    "test": "export_arh_31-42-final.csv",
}
SUBMISSION_FILE = "submission.csv"
MIN_OVERLAP = 60.0
MIN_FRACTION = 0.10
MAX_PAIR_HISTORY = 60
PERIOD_STARTS = {
    "train": datetime(2020, 3, 9).timestamp(),
    "valid": datetime(2020, 5, 18).timestamp(),
    "test": datetime(2020, 7, 27).timestamp(),
}
FEATURE_CACHE: dict[str, Counter[str]] = {}


def number(value: str) -> float:
    value = (value or "").strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        text = value.replace("Z", "+00:00")
        for fmt in (None, "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
            try:
                if fmt is None:
                    return datetime.fromisoformat(text).timestamp()
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
        raise ValueError(f"Unsupported timestamp: {value!r}")


def norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")


def find_column(row: dict[str, str], *names: str) -> str:
    keys = {norm_key(k): k for k in row}
    for name in names:
        if norm_key(name) in keys:
            return keys[norm_key(name)]
    raise KeyError(f"Missing one of {names}; columns={list(row)}")


def schedule_metadata(row: dict[str, str]) -> tuple[str, dict[str, str]] | None:
    sid = row[find_column(row, "tv_show_id")].strip()
    if not sid or sid == "0":
        return None
    title = row.get(find_column(row, "tv_show_title"), "")
    category = row.get(find_column(row, "tv_show_category"), "")
    genres = [row.get(find_column(row, f"tv_show_genre_{i}"), "") for i in (1, 2, 3)]
    return sid, {
        "title": title.strip(),
        "category": category.strip(),
        "genres": "|".join(g.strip() for g in genres if g.strip()),
        "year": row.get(find_column(row, "year_of_production"), "").strip(),
        "director": row.get(find_column(row, "director"), "").strip(),
        "actors": row.get(find_column(row, "actors"), "").strip(),
    }


def read_schedule(path: Path):
    """Return channel -> sorted intervals and id -> metadata."""
    by_channel: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {name.strip(): i for i, name in enumerate(header)}
        for row in reader:
            if len(row) <= idx["tv_show_id"]:
                continue
            sid = row[idx["tv_show_id"]].strip()
            if not sid or sid == "0":
                continue
            meta = {
                "title": row[idx["tv_show_title"]].strip(),
                "category": row[idx["tv_show_category"]].strip(),
                "genres": "|".join(row[idx[f"tv_show_genre_{i}"]].strip() for i in (1, 2, 3) if row[idx[f"tv_show_genre_{i}"]].strip()),
                "year": row[idx["year_of_production"]].strip(),
                "director": row[idx["director"]].strip(),
                "actors": row[idx["actors"]].strip(),
            }
            metadata.setdefault(sid, meta)
            channel = row[idx["channel_id"]].strip()
            start = number(row[idx["start_time"]])
            duration = number(row[idx["duration"]])
            if duration <= 0:
                continue
            by_channel[channel].append((start, start + duration, sid))
    for channel in by_channel:
        by_channel[channel].sort(key=lambda x: x[0])
    return by_channel, metadata


def join_views(view_path: Path, schedules, allowed_users: set[str] | None = None):
    """Join views to the best-overlap labelled programme for each period."""
    joined: dict[str, dict[str, dict[str, float]]] = {k: defaultdict(dict) for k in schedules}
    audit = Counter()
    channel_indexes = {period: {c: [x[0] for x in rows] for c, rows in by_channel.items()} for period, by_channel in schedules.items()}
    with view_path.open("r", encoding="utf-8-sig", newline="", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {name.strip(): i for i, name in enumerate(header)}
        for row in reader:
            audit["views"] += 1
            user = row[idx["user_id"]].strip()
            if allowed_users is not None and user not in allowed_users:
                audit["filtered_users"] += 1
                continue
            channel = row[idx["vsetv_id"]].strip()
            start = number(row[idx["start_time"]])
            stop = number(row[idx["stop_time"]])
            if stop <= start:
                audit["bad_intervals"] += 1
                continue
            if start >= PERIOD_STARTS["valid"]:
                period = "valid"
            else:
                period = "train"
            rows = schedules[period].get(channel)
            if not rows:
                audit["unmatched_or_unlabelled"] += 1
                continue
            starts = channel_indexes[period][channel]
            cutoff = bisect_right(starts, stop)
            best_overlap = 0.0
            best_sid = None
            for j in range(cutoff - 1, -1, -1):
                ss, ee, sid = rows[j]
                if ee <= start:
                    break
                overlap = min(stop, ee) - max(start, ss)
                if overlap <= best_overlap:
                    continue
                fraction = overlap / max(ee - ss, 1.0)
                if overlap >= MIN_OVERLAP and fraction >= MIN_FRACTION:
                    best_overlap, best_sid = overlap, sid
            if best_sid is None:
                audit["unmatched_or_unlabelled"] += 1
                continue
            overlap, sid = best_overlap, best_sid
            old = joined[period][user].get(sid, 0.0)
            joined[period][user][sid] = old + overlap
            audit["matched"] += 1
            if audit["views"] % 1_000_000 == 0:
                print(f"joined {audit['views']:,} views", flush=True)
    return joined, audit


def histories(*periods):
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for period in periods:
        for user, items in period.items():
            for sid, weight in items.items():
                out[user][sid] = out[user].get(sid, 0.0) + weight
    return out


def popularity(hist: dict[str, dict[str, float]]) -> list[str]:
    users = Counter()
    for items in hist.values():
        for sid in items:
            users[sid] += 1
    return [sid for sid, _ in users.most_common()]


def tokens(meta: dict[str, str]) -> Counter[str]:
    out = Counter()
    if meta["category"]:
        out[f"cat:{meta['category'].lower()}"] += 2.0
    for genre in meta["genres"].split("|"):
        if genre:
            out[f"genre:{genre.lower()}"] += 1.0
    year = re.search(r"\d{4}", meta["year"])
    if year:
        out[f"year:{year.group()[:3]}0s"] += 0.5
    for field, prefix in (("director", "director"), ("actors", "actor")):
        for word in re.findall(r"[a-zа-я0-9]+", meta[field].lower()):
            if len(word) > 2:
                out[f"{prefix}:{word}"] += 0.25
    return out


def item_tokens(sid: str, meta: dict[str, dict[str, str]]) -> Counter[str]:
    if sid not in FEATURE_CACHE:
        FEATURE_CACHE[sid] = tokens(meta.get(sid, {}))
    return FEATURE_CACHE[sid]


def cosine_profile(hist: dict[str, float], meta: dict[str, dict[str, str]], blocked: set[str], candidates: set[str]):
    profile = Counter()
    for sid, weight in hist.items():
        vec = item_tokens(sid, meta) if sid in meta else Counter()
        scale = math.log1p(max(weight, 1.0))
        for key, value in vec.items():
            profile[key] += value * scale
    pnorm = math.sqrt(sum(v * v for v in profile.values())) or 1.0
    scored = []
    for sid in candidates - blocked:
        vec = item_tokens(sid, meta) if sid in meta else Counter()
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        dot = sum(profile[k] * v for k, v in vec.items())
        scored.append((dot / (pnorm * norm), sid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [sid for _, sid in scored]


def association_model(histories_train: dict[str, dict[str, float]]):
    item_users = Counter()
    pair = defaultdict(Counter)
    for items in histories_train.values():
        ids = list(items)[:MAX_PAIR_HISTORY]
        for sid in ids:
            item_users[sid] += 1
        for a, b in combinations(sorted(set(ids)), 2):
            pair[a][b] += 1
            pair[b][a] += 1
    return item_users, pair


def association_rank(hist: dict[str, float], item_users, pair, blocked: set[str], candidates: set[str], total_item_users: int):
    scored = Counter()
    for a, weight in hist.items():
        denom = max(item_users.get(a, 1), 1)
        for b, count in pair.get(a, {}).items():
            if b in candidates and b not in blocked:
                confidence = count / denom
                lift = confidence / max(item_users.get(b, 1) / max(total_item_users, 1), 1e-9)
                scored[b] += math.log1p(weight) * confidence * math.log1p(lift)
    ranked = [sid for sid, _ in scored.most_common()]
    return ranked


def top5(ranked, fallback, blocked):
    result = []
    for sid in list(ranked) + list(fallback):
        if sid in blocked or sid in result:
            continue
        result.append(sid)
        if len(result) == 5:
            return result
    return result[:5]


def map5(predictions, truth):
    values = []
    for user, relevant in truth.items():
        rel = set(relevant)
        if not rel:
            continue
        hits = 0
        ap = 0.0
        for rank, sid in enumerate(predictions.get(user, []), 1):
            if sid in rel:
                hits += 1
                ap += hits / rank
        values.append(ap / min(5, len(rel)))
    return sum(values) / len(values) if values else 0.0


def read_submission(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    schedules, metadata = {}, {}
    schedule_ids = {}
    for period, filename in SCHEDULE_FILES.items():
        by_channel, meta = read_schedule(args.data_dir / filename)
        schedules[period] = by_channel
        metadata.update(meta)
        schedule_ids[period] = set(meta)
    header, template_rows = read_submission(args.data_dir / SUBMISSION_FILE)
    allowed_users = {row[0].strip() for row in template_rows}
    joined, audit = join_views(args.data_dir / VIEW_FILE, schedules, allowed_users)
    train, valid = joined["train"], joined["valid"]
    final_hist = histories(train, valid)
    train_ids = set(schedule_ids["train"])
    valid_ids = train_ids | schedule_ids["valid"]
    final_ids = valid_ids | schedule_ids["test"]
    pop_train = popularity(train)
    # Pair mining is only needed for users evaluated in validation or submitted
    # to Kaggle; restricting it keeps the exam run bounded on large logs.
    model_users = set(valid) | {row[0].strip() for row in template_rows}
    pair_train = {u: train[u] for u in model_users if u in train}
    item_users, pair = association_model(pair_train)
    total_item_users = sum(item_users.values())

    def make_predictions(source, candidate_ids, fallback):
        result = {}
        for user, hist in source.items():
            blocked = set(hist)
            cands = candidate_ids
            cbf = cosine_profile(hist, metadata, blocked, cands)
            assoc = association_rank(hist, item_users, pair, blocked, cands, total_item_users)
            result[user] = {
                "popularity": top5([], fallback, blocked),
                "content": top5(cbf, fallback, blocked),
                "association": top5(assoc, fallback, blocked),
            }
        return result

    valid_truth = {u: list(items) for u, items in valid.items()}
    valid_source = {u: train.get(u, {}) for u in valid if u in train}
    valid_predictions = make_predictions(valid_source, valid_ids, pop_train)
    validation = {}
    for method in ("popularity", "content", "association"):
        validation[method] = map5({u: p[method] for u, p in valid_predictions.items()}, valid_truth)

    # Final predictions are generated from train+validation histories.
    final_fallback = popularity(final_hist)
    test_users = [row[0].strip() for row in template_rows]
    final_source = {u: final_hist.get(u, {}) for u in test_users}
    final_predictions = make_predictions(final_source, final_ids, final_fallback)
    chosen = max(("content", validation["content"]), ("association", validation["association"]), key=lambda x: x[1])[0]
    rows = [header]
    examples = {}
    for row in template_rows:
        user = row[0].strip()
        hist = final_hist.get(user, {})
        preds = final_predictions.get(user, {}).get(chosen)
        if preds is None:
            preds = top5([], popularity(final_hist), set(hist))
        while len(preds) < 5:
            for sid in popularity(final_hist):
                if sid not in preds and sid not in hist:
                    preds.append(sid)
                if len(preds) == 5:
                    break
        rows.append([user, " ".join(preds[:5])])
        if len(examples) < 2:
            examples[user] = {"history": list(hist), "predictions": final_predictions.get(user, {})}
    with (args.out_dir / "submission.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    (args.out_dir / "validation_results.json").write_text(json.dumps({"selected": chosen, "map5": validation}, indent=2), encoding="utf-8")
    (args.out_dir / "audit.json").write_text(json.dumps({"join": audit, "schedule_ids": {k: len(v) for k, v in schedule_ids.items()}, "metadata": len(metadata), "users_train": len(train), "users_final": len(final_hist), "candidate_final": len(final_ids)}, indent=2), encoding="utf-8")
    (args.out_dir / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected": chosen, "map5": validation, "audit": audit, "submission_rows": len(rows) - 1}, indent=2))


if __name__ == "__main__":
    main()
