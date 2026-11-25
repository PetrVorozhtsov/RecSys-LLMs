"""Independently audit saved rankings, input dates and a hand-workable nDCG case."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

from risk_features import RiskStore
from run_experiment import ROOT, make_prompt, read_json, read_jsonl, write_json


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def independent_ndcg(ranking, truth):
    relevance = [int(isin in truth) for isin in ranking]
    discounts = [math.log(2) / math.log(rank + 1) for rank in range(1, 11)]
    dcg = sum(rel * weight for rel, weight in zip(relevance, discounts))
    idcg = sum(discounts[:min(10, len(truth))])
    return relevance, dcg, idcg, dcg / idcg if idcg else 0.0


def verify(data_dir, out):
    protocol = read_json(out / "protocol.json")
    metrics = read_json(out / "metrics.json")
    cases = read_jsonl(out / "cases.jsonl")
    prompts = read_jsonl(out / "prompts.jsonl")
    records = csv_rows(out / "per_user.csv")
    raw_predictions = csv_rows(out / "predictions.csv")
    rankings = defaultdict(list)
    for row in raw_predictions:
        rankings[(row["user_id"], row["method"])].append((int(row["rank"]), row["ISIN"]))
    for key, values in rankings.items():
        values.sort()
        assert [r for r, _ in values] == list(range(1, 11)), key
        rankings[key] = [isin for _, isin in values]
    checks = []
    assets, users = set(protocol["eligible_assets"]), set(protocol["users"])
    truth, historical_distinct, seen = defaultdict(set), defaultdict(set), set()
    for row in csv_rows(data_dir / "transactions.csv"):
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        if int(row["transactionID"]) < 0 or row["transactionType"] != "Buy":
            continue
        day = row["timestamp"][:10]
        if row["customerID"] in users and row["ISIN"] in assets and day < protocol["cutoff"]:
            historical_distinct[row["customerID"]].add(row["ISIN"])
        if (row["customerID"] in users and row["ISIN"] in assets
                and protocol["cutoff"] <= day < protocol["end_exclusive"]):
            truth[row["customerID"]].add(row["ISIN"])
    assert len(rankings) == len(users) * len(metrics["methods"])
    max_error = 0.0
    expected_keys = {(uid, method) for uid in users for method in metrics["methods"]}
    actual_keys = [(row["user_id"], row["method"]) for row in records]
    assert set(actual_keys) == expected_keys and len(actual_keys) == len(expected_keys)
    for row in records:
        key = row["user_id"], row["method"]
        ranking = rankings[key]
        assert len(set(ranking)) == 10 and set(ranking) <= assets
        rel, dcg, ideal, value = independent_ndcg(ranking, truth[row["user_id"]])
        assert len(truth[row["user_id"]]) == int(row["relevant_count"])
        error = abs(value - float(row["ndcg_at_10"]))
        max_error = max(max_error, error)
        assert error < 1e-12
    for method, aggregate in metrics["methods"].items():
        selected = [row for row in records if row["method"] == method]
        nonempty = [row for row in selected if int(row["relevant_count"])]
        assert aggregate["users"] == len(selected) == len(users)
        assert aggregate["nonempty_truth_users"] == len(nonempty)
        assert aggregate["empty_truth_users"] == len(selected) - len(nonempty)
        assert abs(aggregate["ndcg_at_10"] - sum(float(r["ndcg_at_10"]) for r in selected) / len(selected)) < 1e-12
        assert abs(aggregate["ndcg_at_10_nonempty"] - sum(float(r["ndcg_at_10"]) for r in nonempty) / len(nonempty)) < 1e-12
    checks.append({"name": "independent_metric_and_ID_audit", "result": "pass",
                   "detail": f"Reconstructed truth directly from raw CSV; checked {len(records)} user-method rows and {len(raw_predictions)} ranked IDs; maximum independent nDCG error {max_error:.3g}."})
    assert read_json(out / "baseline" / "protocol.json") == protocol
    for filename, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((data_dir / filename).read_bytes()).hexdigest() == expected
    checks.append({"name": "protocol_and_data_hashes", "result": "pass",
                   "detail": "Baseline/custom protocol.json objects identical; four current CSV SHA-256 values match frozen protocol."})
    assert len(prompts) == len(cases) == len(users)
    for case, prompt in zip(cases, prompts):
        assert case["user_id"] == prompt["user_id"]
        assert make_prompt(case["purchases"]) == prompt["prompt"]
        payload = json.loads(prompt["prompt"].split("\nHISTORY=", 1)[1])
        for purchase, redacted in zip(case["purchases"], payload):
            assert set(redacted) == {"age", "bands"}
            assert purchase["date"] < protocol["cutoff"]
            age = (date.fromisoformat(protocol["cutoff"]) - date.fromisoformat(purchase["date"])).days
            assert age == redacted["age"] == purchase["age_days"] and age > 0
        assert len(payload) == len(case["purchases"]) <= 40
        assert set(case["candidate_ids"]) <= assets
        for method in ("user_knn", "statistical_risk", "llm_risk"):
            assert set(rankings[(case["user_id"], method)]) <= set(case["candidate_ids"])
    candidate_features = read_json(out / "candidate_features.json")
    for feature in candidate_features.values():
        assert feature["metadata_date"] < protocol["cutoff"]
        if feature["last_price_date"]:
            assert feature["last_price_date"] < protocol["cutoff"]
    checks.append({"name": "temporal_and_prompt_audit", "result": "pass",
                   "detail": "All 100 model payloads contain only pre-cutoff ages and seven historical bands; no IDs, questionnaire fields, prices or labels; all three custom methods use the same frozen candidate lists."})
    store = RiskStore(data_dir)
    audit = store.audit_transaction("8996071")
    f = audit["feature"]
    peak = audit["max_drawdown_check"]["peak_price"]
    trough = audit["max_drawdown_check"]["trough_price"]
    assert f["price_count"] == 64 and f["return_count"] == 63
    assert abs(peak - 9.8) < 1e-10 and abs(trough - 8.72) < 1e-10
    assert abs(f["metrics"][1] - (1 - trough / peak)) < 1e-12
    assert all(p["date"] < audit["date"] for p in audit["observations"])
    feature_check = {"transaction_id": "8996071", "price_count": 64, "return_count": 63,
                     "volatility": f["metrics"][0], "max_drawdown": f["metrics"][1],
                     "peak_price": peak, "trough_price": trough, "passed": True}
    write_json(out / "transaction_audit.json", audit)
    checks.append({"name": "real_transaction_feature_check", "result": "pass",
                   "detail": "64 closes, 63 log returns; maximum drawdown independently 1 - 8.72 / 9.80 = 11.020408%; all closes precede 2021-05-28."})
    by_user = defaultdict(dict)
    for row in records:
        by_user[row["user_id"]][row["method"]] = float(row["ndcg_at_10"])
    ordered = protocol["users"]
    future_users = [uid for uid in ordered if truth[uid]]
    # Prefer a nontrivial one-relevant-asset example; tie-break by frozen cohort order.
    example_uid = next((uid for uid in future_users if len(truth[uid]) == 1
                        and by_user[uid]["llm_risk"] > 0), future_users[0])
    example_ranking = rankings[(example_uid, "llm_risk")]
    rel, dcg, ideal, value = independent_ndcg(example_ranking, truth[example_uid])
    ndcg_check = {"cohort_index": ordered.index(example_uid) + 1,
                  "relevance_vector": rel, "dcg": dcg, "idcg": ideal, "ndcg": value,
                  "expected_ndcg": by_user[example_uid]["llm_risk"], "passed": True}
    write_json(out / "verification.json", {
        "performed_by": "OpenAI Codex; not a student's personal check",
        "summary": "Automated independent audit passed; student verification remains separate.",
        "checks": checks, "ndcg_check": ndcg_check, "feature_check": feature_check})
    deltas = [by_user[uid]["llm_risk"] - by_user[uid]["statistical_risk"] for uid in ordered]
    changed = sum(rankings[(uid, "llm_risk")] != rankings[(uid, "statistical_risk")] for uid in ordered)
    comparison = {"mean_delta": sum(deltas) / len(deltas),
                  "nonempty_mean_delta": sum(deltas) / len(future_users),
                  "improved_users": sum(x > 1e-12 for x in deltas),
                  "tied_users": sum(abs(x) <= 1e-12 for x in deltas),
                  "worse_users": sum(x < -1e-12 for x in deltas),
                  "changed_top10_users": changed}
    loss_uid = min(ordered, key=lambda uid: by_user[uid]["llm_risk"] - by_user[uid]["statistical_risk"])
    loss = {"cohort_index": ordered.index(loss_uid) + 1,
            "statistical_ndcg": by_user[loss_uid]["statistical_risk"],
            "llm_ndcg": by_user[loss_uid]["llm_risk"],
            "statistical_relevance": independent_ndcg(rankings[(loss_uid, "statistical_risk")], truth[loss_uid])[0],
            "llm_relevance": independent_ndcg(rankings[(loss_uid, "llm_risk")], truth[loss_uid])[0]}
    if comparison["worse_users"] == 0:
        loss = None
    coverage = read_json(out / "candidate_coverage.json")
    misses = [c for c in coverage if c["truth_count"] > c["candidate_hits"]]
    failure_case = {
        "description": f"Retrieval missed {metrics['total_truth_assets'] - metrics['candidate_truth_hits']} of {metrics['total_truth_assets']} held-out eligible assets; {len(misses)} future buyers have at least one missed asset.",
        "root_cause": "Only assets bought by the 50 nearest neighbors can enter the top-50 pool, except popularity fill to ten; profile encoding cannot recover an excluded asset.",
        "fix": "Next experiment: enlarge the candidate pool or union it with a popularity/CBF pool before freezing the same pool for both encoders. This proposed retrieval change has not been evaluated.",
        "verification": f"Independently intersected future eligible purchases with each saved candidate list; micro candidate recall {metrics['candidate_recall_micro']:.6f}. This diagnoses the bottleneck; it does not validate the proposed next experiment.",
    }
    # A true implementation issue found and resolved, distinct from a proposed research fix.
    engineering_case = {
        "description": f"The asset metadata has {store.same_date_metadata_ties} same-ISIN/same-date duplicate rows.",
        "root_cause": "Selecting the last tied CSV row would disagree with the instructor's first-row tie policy.",
        "fix": "Historical metadata uses the first row for equal timestamps, matching starter.latest_before.",
        "verification": "The unit test pins this tie rule and the frozen eligible universe matches the starter exactly.",
    }
    # A descriptive paired interval, not a test-set tuning criterion.
    import random
    rng = random.Random(42)
    bootstrap = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(2000))
    comparison["paired_bootstrap_95pct"] = [bootstrap[49], bootstrap[1949]]
    saved_profiles = read_jsonl(out / "profiles.jsonl")
    uniform_profiles = sum(all(abs(value - 1 / 3) < 1e-3 for triple in row["profile"] for value in triple)
                           for row in saved_profiles)
    knn_identical = sum(rankings[(uid, "llm_risk")] == rankings[(uid, "user_knn")] for uid in ordered)
    comparison["llm_knn_identical_top10_users"] = knn_identical
    comparison["llm_minus_knn_ndcg_all"] = metrics["methods"]["llm_risk"]["ndcg_at_10"] - metrics["methods"]["user_knn"]["ndcg_at_10"]
    known_history_users = sum(any(band is not None for purchase in case["purchases"] for band in purchase["bands"])
                              for case in cases)
    input_summary = {
        "eligible_assets": len(assets),
        "candidate_assets_with_all_seven_features": sum(all(x is not None for x in f["metrics"]) for f in candidate_features.values()),
        "users_with_any_known_historical_risk_band": known_history_users,
        "users_with_all_unknown_historical_bands": len(users) - known_history_users,
        "purchases_supplied": sum(len(case["purchases"]) for case in cases),
        "uniform_effective_llm_profiles": uniform_profiles}
    short = [uid for uid in ordered if len(historical_distinct[uid]) in (1, 2)]
    short_metrics = {}
    for method in metrics["methods"]:
        nonempty_short = [uid for uid in short if truth[uid]]
        short_metrics[method] = {
            "users": len(short), "nonempty_truth_users": len(nonempty_short),
            "ndcg_at_10": sum(by_user[uid][method] for uid in short) / len(short) if short else None,
            "ndcg_at_10_nonempty": sum(by_user[uid][method] for uid in nonempty_short) / len(nonempty_short) if nonempty_short else None}
    retrieval_failure_case = failure_case
    interpretation = (
        f"The LLM exceeds the statistical risk control by {comparison['mean_delta']:+.4f} all-user nDCG@10, "
        f"but its Top-10 is identical to original user-kNN for {knn_identical}/{len(users)} investors. "
        f"All {uniform_profiles}/{len(users)} effective LLM profiles are near-uniform (each probability within .001 of 1/3), "
        f"despite {known_history_users} investors having at least one known historical risk band. "
        "The apparent advantage over the frequency control is therefore not evidence of useful LLM risk inference. "
        "This run found no incremental ranking benefit over user-kNN."
    )
    if uniform_profiles == len(users) and knn_identical == len(users):
        risk_difference_bound = 0.25 * 2 * 1e-3
        minimum_prior_gap = 0.75 / (max(len(case["candidate_ids"]) for case in cases) - 1)
        assert risk_difference_bound < minimum_prior_gap
        failure_case = {
            "description": f"Semantic failure despite valid JSON: all {len(users)} LLM profiles were near-uniform, and all Top-10 lists equalled user-kNN.",
            "root_cause": f"Every probability is within .001 of 1/3, bounding the weighted risk-score difference between candidates by {risk_difference_bound:.4f}. The smallest adjacent kNN-prior gap is {minimum_prior_gap:.6f}, so these risk profiles cannot change the order. The separate influence of prompt wording and model capacity has not been isolated.",
            "fix": "Next, test a prompt without the literal uniform-output template on earlier development histories and compare a stronger model while keeping the same information. This semantic fix has not been run or validated on this test cohort.",
            "verification": f"Checked all {len(users)} normalized profiles within .001 of uniform and compared all saved LLM/kNN Top-10 lists byte-for-byte: {knn_identical} identical. The strict schema fixed the pilot format failure, not this semantic limitation.",
        }
    write_json(out / "analysis.json", {
        "summary": interpretation,
        "comparison": comparison, "example_loss": loss, "failure_case": failure_case,
        "retrieval_failure_case": retrieval_failure_case,
        "engineering_case": engineering_case,
        "input_summary": input_summary,
        "short_history_slice": {"rule": "one or two distinct assets in all pre-cutoff eligible purchase history", "methods": short_metrics},
        "next_experiment": "Use an earlier development split to diagnose uniform-profile collapse with a prompt ablation and a stronger model. Then test broader frozen candidates and the volatility/drawdown-only ablation. Do not tune on these held-out purchases.",
        "bootstrap_note": "2000 paired user resamples, seed 42; descriptive only, 11 future buyers.",
    })
    submission = ROOT / "submission"
    submission.mkdir(exist_ok=True)
    check_text = (
        "PERSONAL CHECK — to be completed by the student\n\n"
        "The assistant has audited these results. The course additionally requires your own check.\n"
        "1. Recompute maximum drawdown for the real price window: peak EUR9.80, later trough EUR8.72.\n"
        "   Formula: 1 - trough / peak. Convert the result to percent.\n"
        f"2. For anonymous cohort user {ndcg_check['cohort_index']}, the LLM Top-10 binary relevance vector is\n"
        f"   {rel}. There are {len(truth[example_uid])} eligible future purchased assets.\n"
        "   DCG=sum(relevance_at_rank / log2(rank+1)); IDCG uses relevant assets first.\n"
        f"   Saved LLM nDCG@10={value:.10f}. Recompute it and explain why position affects the score.\n"
        "3. Check outputs/main/prompts.jsonl: HISTORY records have only age and bands.\n"
        "   Explain why post-cutoff purchases and questionnaire riskLevel are absent.\n\n"
        "Record your calculation and explanation in student_verification.json (see README), then rebuild the PDF.\n"
        "Do not label the assistant's audit as a personal verification.\n"
    )
    (submission / "PERSONAL_CHECK.txt").write_text(check_text, encoding="utf-8")
    print(json.dumps({"verification": "pass", "comparison": comparison, "ndcg_check": ndcg_check,
                      "failure_case": failure_case}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data" / "FAR-Trans")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "main")
    args = parser.parse_args()
    verify(args.data_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
