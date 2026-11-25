"""Reproducible FAR-Trans HW2: seven-feature investor-risk Encoder comparison.

The unchanged instructor starter supplies all classical baselines, the cohort,
eligibility and evaluation. Model inputs contain only ages and historical bands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

from risk_features import METRIC_NAMES, RiskStore
from risk_method import build_candidates, score_candidates, statistical_profile
from vendor import starter

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "799e54c0493cf7e2b014152548ae5c18ee24a8f5"
PROMPT_VERSION = "risk-profile-v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path, values):
    with Path(path).open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def make_prompt(purchases):
    payload = [{"age": p["age_days"], "bands": p["bands"]} for p in purchases]
    return (
        "You encode an investor's observed asset purchases into a risk-band preference profile. "
        "Use ONLY the supplied purchase ages and computed price-risk bands. Do not infer motives, "
        "psychological risk tolerance, asset identities or future returns. "
        "The history is chronological, oldest first, at most 40 purchases. Age is days before prediction. "
        "For each of seven metrics, bands 0,1,2 mean relatively low, middle, high measured price risk "
        "among the same asset type on the purchase date; null means unknown. "
        "Summarize repeated choices and shifts toward recent choices. "
        "Output ONLY a JSON object with key profile: seven arrays of three nonnegative probabilities "
        "for bands [0,1,2], each summing to 1. Use [0.333333,0.333333,0.333334] when no values are known. "
        "The metric order is " + json.dumps(list(METRIC_NAMES)) + ". "
        "There are no demonstrations.\nHISTORY=" + json.dumps(payload, separators=(",", ":"))
    )


def validate_profile(text, purchases):
    """Strict structure; allow <=.02 rounding error per triple, then normalize."""
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"profile"}:
        raise ValueError("Expected an object with only profile")
    profile = obj["profile"]
    if not isinstance(profile, list) or len(profile) != 7:
        raise ValueError("Expected seven metric triples")
    checked = []
    for index, triple in enumerate(profile):
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError("Expected a three-band array")
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) or x < 0 or x > 1 for x in triple):
            raise ValueError("Probabilities must be finite numbers in [0,1]")
        total = sum(triple)
        if abs(total - 1) > 0.0200000001:
            raise ValueError("Probabilities must sum to one within .02 rounding tolerance")
        if not any(p["bands"][index] is not None for p in purchases):
            checked.append([1 / 3] * 3)
        else:
            checked.append([x / total for x in triple])
    return checked


def prepare(data_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    baseline = out / "baseline"
    command = [sys.executable, str(ROOT / "vendor" / "starter.py"), "run",
               "--data-dir", str(data_dir), "--output-dir", str(baseline)]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, encoding="utf-8", capture_output=True, check=True)
    elapsed = time.perf_counter() - started
    (out / "baseline_run.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    protocol = read_json(baseline / "protocol.json")
    write_json(out / "protocol.json", protocol)
    print(completed.stdout, flush=True)
    timings = {"classical_baselines_whole_run_seconds": elapsed,
               "baseline_timing_scope": "one subprocess: data loading and all four baselines together"}
    started = time.perf_counter()
    assets, _, history, _, _, _ = starter.load_data(data_dir, protocol["cutoff"], 30)
    pop = starter.popularity_scores(history, assets)
    store = RiskStore(data_dir)
    features = store.features(protocol["cutoff"])
    cases, prompts = [], []
    for number, uid in enumerate(protocol["users"], 1):
        purchases = store.purchase_context(uid, protocol["cutoff"], limit=40)
        candidates = build_candidates(uid, history, pop, set(assets))
        starter.validate_ranking(candidates[:10], assets)
        profile = statistical_profile(purchases)
        stat_top = score_candidates(candidates, profile, features)
        starter.validate_ranking(stat_top, candidates)
        case = {"user_id": uid, "cutoff": protocol["cutoff"], "purchases": purchases,
                "candidate_ids": candidates, "statistical_profile": profile,
                "user_knn": candidates[:10], "statistical_risk": stat_top}
        cases.append(case)
        prompt = make_prompt(purchases)
        prompts.append({"user_id": uid, "protocol_hash": protocol["protocol_hash"],
                        "prompt_version": PROMPT_VERSION, "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        print(f"Prepared {number}/{len(protocol['users'])}", flush=True)
    timings["risk_preparation_whole_run_seconds"] = time.perf_counter() - started
    write_json(out / "candidate_features.json", features)
    write_jsonl(out / "cases.jsonl", cases)
    write_jsonl(out / "prompts.jsonl", prompts)
    write_json(out / "timings.json", timings)
    write_json(out / "experiment.json", {
        "upstream_repository": "https://github.com/dryjins/RecSys-LLMs",
        "upstream_commit": UPSTREAM_COMMIT, "prompt_version": PROMPT_VERSION,
        "metrics_order": list(METRIC_NAMES), "price_window_calendar_days": 90,
        "min_valid_returns": 40, "max_price_gap_days": 7, "max_price_staleness_days": 7,
        "risk_band_percentiles": [33, 67], "band_reference": "same asset category as of feature date",
        "history_limit_purchases": 40, "statistical_weight": "2 ** (-age_days / 90)",
        "neighbors": 50, "max_candidates": 50, "knn_similarity": "binary purchase cosine, exclude self",
        "rank_score": "1 - zero_based_candidate_rank / (candidate_count - 1)",
        "final_score": ".75 * rank_score + .25 * mean(seven risk probabilities)",
        "unknown_candidate_probability": 1 / 3,
        "invalid_output": "statistical risk profile; retain and count user",
        "normalization_tolerance": .02,
        "format_pilot": "first three frozen users; no future labels; 2/3 unconstrained outputs invalid, so strict shape-only JSON schema fixed before main run",
        "structured_generation": "seven arrays of three numbers in [0,1]; sums and meaning validated separately; no semantic repair",
        "changes_from_hw1": "official Qwen GGUF Q8_0 quantization and llama.cpp Vulkan for a 4GB GPU; main feature and ranking settings retained",
        "input_privacy": "model sees ages and seven risk bands only; no customer, asset or questionnaire fields",
        "python": sys.version, "platform": platform.platform(),
    })


def infer(out, model_path, server_url, limit=0):
    from llm_runtime import LocalGenerator
    generator = LocalGenerator(model_path=model_path, server_url=server_url)
    metadata = dict(generator.metadata, max_new_tokens=256)
    write_json(out / "model.json", metadata)
    prompts = read_jsonl(out / "prompts.jsonl")
    existing = read_jsonl(out / "responses.jsonl")
    by_user = {r["user_id"]: r for r in existing}
    if len(by_user) != len(existing):
        raise ValueError("Duplicate saved responses")
    settings_digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    started = time.perf_counter()
    calls = 0
    with (out / "responses.jsonl").open("a", encoding="utf-8") as stream:
        for number, row in enumerate(prompts, 1):
            if row["user_id"] in by_user:
                old = by_user[row["user_id"]]
                if old["prompt_sha256"] != row["prompt_sha256"] or old["settings_sha256"] != settings_digest:
                    raise ValueError("Saved response belongs to different prompt or model settings")
                continue
            record = dict(row, settings_sha256=settings_digest)
            try:
                record["response"] = generator.generate(row["prompt"], max_new_tokens=256)
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            calls += 1
            print(f"Model call {number}/{len(prompts)}: "
                  f"{record.get('response', {}).get('seconds', 0):.2f}s "
                  f"{record.get('error', '')}", flush=True)
            if limit and calls >= limit:
                break
    timings = read_json(out / "timings.json")
    timings.setdefault("inference_invocations", []).append({
        "new_calls": calls, "wall_seconds": time.perf_counter() - started,
        "saved_responses_reused": len(existing)})
    write_json(out / "timings.json", timings)


def evaluate(data_dir, out):
    protocol = read_json(out / "protocol.json")
    baseline_protocol = read_json(out / "baseline" / "protocol.json")
    if protocol != baseline_protocol:
        raise ValueError("Baseline and custom protocols differ")
    assets, _, history, truth, end, _ = starter.load_data(data_dir, protocol["cutoff"], 30)
    if sorted(assets) != protocol["eligible_assets"] or end != protocol["end_exclusive"]:
        raise ValueError("Evaluation data differs from frozen protocol")
    for filename, digest in protocol["source_sha256"].items():
        if hashlib.sha256((data_dir / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Data changed since preparation: " + filename)
    cases = read_jsonl(out / "cases.jsonl")
    response_rows = read_jsonl(out / "responses.jsonl")
    responses = {r["user_id"]: r for r in response_rows}
    prompt_rows = read_jsonl(out / "prompts.jsonl")
    prompts = {r["user_id"]: r for r in prompt_rows}
    settings_digest = hashlib.sha256(json.dumps(read_json(out / "model.json"), sort_keys=True).encode()).hexdigest()
    users = protocol["users"]
    if ([c["user_id"] for c in cases] != users or set(responses) != set(users)
            or len(responses) != len(response_rows) or set(prompts) != set(users)
            or len(prompts) != len(prompt_rows)):
        raise ValueError("Need exactly one actual model attempt for each frozen user")
    features = read_json(out / "candidate_features.json")
    baseline_metrics = read_json(out / "baseline" / "metrics.json")
    with (out / "baseline" / "per_user.csv").open(encoding="utf-8", newline="") as stream:
        per_user = list(csv.DictReader(stream))
    with (out / "baseline" / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    for r in per_user:
        r["relevant_count"] = int(r["relevant_count"])
        r["ndcg_at_10"] = float(r["ndcg_at_10"])
    failures, profiles, coverage = [], [], []
    rank_times = {"user_knn": 0.0, "statistical_risk": 0.0, "llm_risk": 0.0}
    for case in cases:
        uid = case["user_id"]
        response = responses[uid]
        if response["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("Model response protocol mismatch")
        expected_prompt = make_prompt(case["purchases"])
        expected_sha = hashlib.sha256(expected_prompt.encode()).hexdigest()
        if (response.get("prompt") != expected_prompt or prompts[uid]["prompt"] != expected_prompt
                or response.get("prompt_sha256") != expected_sha
                or response.get("settings_sha256") != settings_digest):
            raise ValueError("Model response prompt or settings mismatch")
        fallback = False
        try:
            if "error" in response:
                raise ValueError(response["error"])
            profile = validate_profile(response["response"]["text"], case["purchases"])
        except (ValueError, TypeError, KeyError) as error:
            fallback = True
            failures.append({"user_id": uid, "error": str(error)})
            profile = case["statistical_profile"]
        started = time.perf_counter()
        llm_top = score_candidates(case["candidate_ids"], profile, features)
        rank_times["llm_risk"] += time.perf_counter() - started
        relevant = truth.get(uid, set()) & set(assets)
        coverage.append({"user_id": uid, "truth_count": len(relevant),
                         "candidate_hits": len(relevant & set(case["candidate_ids"])),
                         "candidate_count": len(case["candidate_ids"])})
        profiles.append({"user_id": uid, "profile": profile, "fallback": fallback})
        for name, ranking in (("user_knn", case["user_knn"]),
                              ("statistical_risk", case["statistical_risk"]), ("llm_risk", llm_top)):
            starter.validate_ranking(ranking, case["candidate_ids"])
            per_user.append({"method": name, "user_id": uid, "relevant_count": len(relevant),
                             "ndcg_at_10": starter.ndcg10(ranking, relevant)})
            predictions.extend({"method": name, "user_id": uid, "rank": i, "ISIN": isin}
                               for i, isin in enumerate(ranking, 1))
    methods = dict(baseline_metrics["methods"])
    for name in ("user_knn", "statistical_risk", "llm_risk"):
        methods[name] = starter.aggregate([r for r in per_user if r["method"] == name])
    for name in ("popularity", "cbf", "latest_buy", "latest_buy_or_sell"):
        computed = starter.aggregate([r for r in per_user if r["method"] == name])
        for key, value in computed.items():
            if value != methods[name][key]:
                raise AssertionError("Upstream metric mismatch")
    total_truth = sum(c["truth_count"] for c in coverage)
    metrics = {"protocol_hash": protocol["protocol_hash"], "methods": methods,
               "llm_failures": failures, "llm_attempts": len(responses),
               "candidate_recall_micro": sum(c["candidate_hits"] for c in coverage) / total_truth if total_truth else None,
               "candidate_truth_hits": sum(c["candidate_hits"] for c in coverage),
               "total_truth_assets": total_truth,
               "generation_seconds": sum(r.get("response", {}).get("seconds", 0) for r in responses.values()),
               "input_tokens": sum(r.get("response", {}).get("input_tokens", 0) for r in responses.values()),
               "output_tokens": sum(r.get("response", {}).get("output_tokens", 0) for r in responses.values()),
               "api_cost": 0, "api_cost_note": "local inference; electricity and hardware cost not measured",
               "llm_scoring_seconds": rank_times["llm_risk"]}
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "profiles.jsonl", profiles)
    write_json(out / "candidate_coverage.json", coverage)
    for filename, values, fields in (
        ("per_user.csv", per_user, ["method", "user_id", "relevant_count", "ndcg_at_10"]),
        ("predictions.csv", predictions, ["method", "user_id", "rank", "ISIN"])):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer", "evaluate", "all"])
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data" / "FAR-Trans")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "main")
    parser.add_argument("--model-path", type=Path, default=ROOT.parent / "models" / "Qwen3-1.7B-Q8_0.gguf")
    parser.add_argument("--server-url", default="http://127.0.0.1:8089")
    parser.add_argument("--limit", type=int, default=0, help="smoke test only; evaluation requires all 100")
    args = parser.parse_args()
    data_dir, out = args.data_dir.resolve(), args.output_dir.resolve()
    if args.stage in ("prepare", "all"):
        prepare(data_dir, out)
    if args.stage in ("infer", "all"):
        infer(out, args.model_path.resolve(), args.server_url, args.limit)
    if args.stage in ("evaluate", "all"):
        evaluate(data_dir, out)


if __name__ == "__main__":
    main()
