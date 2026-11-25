# FAR-Trans HW8 / Session 2 — Investor Risk-Profile Encoder

This is the implementation of Pyotr Alekseyevich Vorozhtsov's HW7 question:
does an LLM profile of observed purchase risk bands improve purchase ranking over
a recency-frequency profile built from the same information?

The experiment retains the instructor's four classical baselines and evaluates
three additional variants: user-kNN, statistical risk profile, and an actual local
Qwen3-1.7B risk Encoder. All 100 users remain in the evaluation, including failures
and users without future purchases. Model outputs are saved, so evaluation can be
replayed without downloading or running the model.

## Measured outcome

All methods use 100 users, including 11 future buyers.

| Method | nDCG@10 all | nDCG@10 nonempty |
|---|---:|---:|
| Popularity | 0.0286 | 0.2599 |
| CBF | 0.0201 | 0.1827 |
| Latest buy | 0.0277 | 0.2520 |
| Latest buy or sell | 0.0271 | 0.2464 |
| User-kNN | 0.0285 | 0.2593 |
| Statistical risk | 0.0267 | 0.2429 |
| LLM risk | 0.0285 | 0.2593 |

The 100 actual main-run generations took 706.58 seconds and required no format
fallbacks. All effective LLM profiles were near-uniform; every LLM Top-10 exactly
matched user-kNN. The gain over the statistical control therefore provides no
evidence of useful LLM risk inference in this run. The negative finding is
reported explicitly rather than hidden behind the matched-control difference.

## Read the results

- `submission/vorozhtsov_petr_session2_report.pdf`: individual report.
- `outputs/main/metrics.json`: full-precision aggregate results.
- `outputs/main/per_user.csv`, `predictions.csv`: every evaluated user and recommendation.
- `outputs/main/protocol.json`: frozen cohort, dates, eligible assets and data hashes.
- `outputs/main/model.json`, `experiment.json`: exact inference and method settings.
- `outputs/main/prompts.jsonl`, `responses.jsonl`: actual model inputs and responses.
- `outputs/main/verification.json`, `transaction_audit.json`: assistant-performed checks.
- `outputs/main/analysis.json`: paired differences, retrieval coverage and failure analysis.
- `submission/PERSONAL_CHECK.txt`: the student's separate verification exercise.

The source and results use the teaching split, not the official final project test.
Empty truth gives zero in the primary mean. Always distinguish the all-user mean
from the mean over future buyers.

## Reproduce

Python 3.10+ is sufficient for the experiment and its tests; Python 3.12.14 was
used. `reportlab` is needed only to rebuild the PDF. The inference runtime is a
pinned official Windows x64 llama.cpp Vulkan binary; no paid API or Python ML
package is required. On another OS, use an equivalent llama.cpp server with the
same model, template and generation settings, and record the changed backend.

Run these commands **from this folder** with your Python interpreter:

```powershell
# Verify the existing data, or fetch the pinned official archive.
python setup_data.py

# Replay the saved 100 genuine model responses; no LLM download is needed.
python run_experiment.py evaluate
python verify_results.py

# Tests: project tests plus the instructor's complete suite.
python -m unittest test_risk test_experiment test_export -v
python -m unittest discover -s vendor -p test_starter.py -v
```

For a fresh experiment, download about 1.9 GB of public model/runtime assets once:

```powershell
python download_runtime.py
python llm_runtime.py start
python run_experiment.py prepare --output-dir outputs/fresh
python run_experiment.py infer --output-dir outputs/fresh
python run_experiment.py evaluate --output-dir outputs/fresh
python verify_results.py --output-dir outputs/fresh
```

`llm_runtime.py start` launches a hidden loopback-only server on port 8089. The
server remains alive for the inference stage. Its PID and logs are saved in
`../runtime/`; stop that exact process after use. On the author's hybrid laptop,
the server is restricted to the NVIDIA Vulkan driver in its child environment;
see `model.json` and `llm_runtime.py` for the measured configuration.

`infer` appends one response per user and safely resumes interrupted runs. It
rejects a saved response if the prompt, model or generation settings changed.
Use a new output directory for a changed experiment. The main comparison uses
zero-shot prompts; parameters were not tuned against test purchases.

### Bundled Codex Python on this computer

The shell alias `python` may be a Microsoft Store launcher. The interpreter used
for the delivered run is:

```powershell
$experimentPython = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe'
& $experimentPython run_experiment.py evaluate
& $experimentPython verify_results.py
& $experimentPython build_report.py
```

## Method

`risk_features.py` computes seven price measures in `[asof-90 days, asof)`:
sample log-return volatility, maximum drawdown, downside log-return RMS, worst
five-observation-interval return, current drawdown, share of prices in drawdown
strictly greater than 5%, and last-20 volatility minus full-window volatility.
Only positive closes are used. Returns crossing gaps greater than seven days are
excluded; at least 40 valid changes and a close no more than seven days old are
required. Flat windows are unknown. Five-interval windows require every gap to
pass the same rule.

Bands use linearly interpolated 33rd/67th percentiles of valid measures for the
same asset category on the relevant date. Ties join the lower band; the sign of
worst five-interval return is reversed for banding. Metadata and prices are
strictly earlier than the relevant date, including each historical purchase.

The last 40 observed purchases are supplied as chronological `(age, seven bands)`
records. The statistical control sums `2 ** (-age/90)` within each metric's bands.
The LLM receives exactly these records, without asset names, IDs, customer fields,
questionnaires or future labels, and returns seven probability triples. Unknown
history dimensions use uniform triples for both methods.

Both use the same binary user-kNN retrieval (50 neighbors, excluding self, up to
50 assets; popularity fills only if fewer than ten), with stable identifier ties.
Candidate score is `0.75 * (1-rank/(n-1)) + 0.25 * mean(risk probability)`.
An unknown candidate dimension contributes 1/3. Final score ties preserve kNN
order. Malformed model JSON, invalid values, shape or row sums fall back to the
statistical profile; each failure is retained and logged. Rounding deviations
of at most .02 in a row sum are normalized.

The unmodified instructor code in `vendor/` supplies loading, user selection,
all four classical methods and nDCG evaluation. The risk-score harness is separate
because the starter's simple Encoder seam accepts only past-asset weights.
`UPSTREAM.json` pins the instructor commit. The data are CC BY 4.0, University of
Glasgow DOI 10.5525/gla.researchdata.1658; raw CSVs and model binaries are outside
the source repository. `setup_data.py` verifies exact archive/CSV hashes.

The development format pilot is preserved in `outputs/format_pilot/`. Two of
three unconstrained outputs violated the required structure, so a shape-only
JSON schema was fixed before the full run. The main run uses exactly seven
triples; schema constraints do not enforce row sums or semantic correctness.
All main failures still use the declared statistical fallback.

## Student verification and identity

The automated checks are explicitly attributed to Codex. The course separately
requires a personal check. Complete `submission/PERSONAL_CHECK.txt` and save:

```json
{
  "confirmed": true,
  "statement": "Your own calculation and explanation, in your own words."
}
```

as `student_verification.json` only after actually doing the check. Rebuild the
PDF with `python build_report.py`. The report states that confirmation is pending
until such a statement is supplied. Enter your email and your own GitHub
repository in `identity.json`; no email or repository is invented. The exact
experiment source commit is recorded there after the measured run.

The documented AI-session derivative includes the coursework portion of the root
chat and implementation subagent sessions. It preserves visible messages,
generated code, tool calls and measured outputs with source line references and
hashes. Internal instructions, private reasoning and runtime bookkeeping are
omitted. See `submission/ai_sessions/README.txt` and `manifest.json` for the exact
export policy. `package_submission.py` builds the deliverable archive. The
experiment scripts do not submit work through the course channel.
