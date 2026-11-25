"""Deterministic retrieval, statistical control and shared risk-match scoring."""
from __future__ import annotations

import math
from collections import defaultdict


def build_candidates(uid, history, pop, eligible, max_neighbors=50, max_candidates=50):
    """Binary user-kNN with self exclusion and deterministic similarity ties.

    Neighbor and asset ties use their identifiers. Only positive-similarity
    neighbors contribute. Popularity fills to ten only when retrieval finds
    fewer than ten candidates; it never changes an already sufficient pool.
    """
    if max_neighbors <= 0 or max_candidates < 10:
        raise ValueError("Require positive neighbor count and at least ten candidates")
    allowed = set(eligible)
    if len(allowed) < 10:
        raise ValueError("At least ten eligible assets are required")
    own = {isin for isin, count in history.get(uid, {}).items() if count > 0 and isin in allowed}
    neighbors = []
    if own:
        for other, counts in history.items():
            if other == uid:
                continue
            assets = {isin for isin, count in counts.items() if count > 0 and isin in allowed}
            common = len(own & assets)
            if common:
                similarity = common / math.sqrt(len(own) * len(assets))
                neighbors.append((similarity, other, assets))
    neighbors.sort(key=lambda row: (-row[0], row[1]))
    scores = defaultdict(float)
    for similarity, _, assets in neighbors[:max_neighbors]:
        for isin in sorted(assets):
            scores[isin] += similarity
    candidates = sorted(scores, key=lambda isin: (-scores[isin], isin))[:max_candidates]
    chosen = set(candidates)
    if len(candidates) < 10:
        for isin in sorted(allowed, key=lambda item: (-pop.get(item, 0), item)):
            if isin not in chosen:
                candidates.append(isin)
                chosen.add(isin)
                if len(candidates) == 10:
                    break
    return candidates


def statistical_profile(purchases):
    """Seven independent triples from recency-weighted observed risk bands."""
    weights = [[0.0, 0.0, 0.0] for _ in range(7)]
    for purchase in purchases:
        age = purchase["age_days"]
        bands = purchase["bands"]
        if age < 0 or len(bands) != 7:
            raise ValueError("Invalid purchase age or band count")
        weight = 2 ** (-age / 90)
        for metric, band in enumerate(bands):
            if band is None:
                continue
            if isinstance(band, bool) or not isinstance(band, int) or band not in (0, 1, 2):
                raise ValueError("Risk bands must be integers 0, 1, 2 or null")
            weights[metric][band] += weight
    return [([v / total for v in triple] if (total := sum(triple)) else [1 / 3] * 3)
            for triple in weights]


def normalize_profile(profile):
    """Validate seven finite, nonnegative triples and normalize each to one."""
    if not isinstance(profile, list) or len(profile) != 7:
        raise ValueError("Risk profile must contain seven triples")
    result = []
    for triple in profile:
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError("Risk profile entries must be triples")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in triple):
            raise ValueError("Risk profile weights must be finite and nonnegative")
        total = sum(triple)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Risk profile triples must have finite positive sums")
        result.append([v / total for v in triple])
    return result


def score_candidates(candidates, profile, features):
    """Shared 0.75 rank prior + 0.25 average risk-band profile weight.

    The kNN rank prior is 1-i/(n-1), where i=0 is the first of n candidates.
    An unknown candidate metric contributes 1/3. Score ties preserve the
    original kNN order. The result is exactly ten unique candidate IDs.
    """
    if len(candidates) < 10 or len(set(candidates)) != len(candidates):
        raise ValueError("Need at least ten unique candidates")
    normalized = normalize_profile(profile)
    scored = []
    for index, isin in enumerate(candidates):
        bands = features.get(isin, {}).get("bands", [None] * 7)
        if len(bands) != 7:
            raise ValueError("Candidate feature must have seven bands")
        match = 0.0
        for metric, band in enumerate(bands):
            if band is None:
                match += 1 / 3
            elif not isinstance(band, bool) and isinstance(band, int) and band in (0, 1, 2):
                match += normalized[metric][band]
            else:
                raise ValueError("Candidate band must be 0, 1, 2 or null")
        prior = 1 - index / (len(candidates) - 1)
        score = 0.75 * prior + 0.25 * match / 7
        scored.append((score, index, isin))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [isin for _, _, isin in scored[:10]]
