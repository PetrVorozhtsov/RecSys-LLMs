# HW5 - From Ratings to Recommendations

**Student:** Petr Vorozhtsov. **Course:** LLM4RecSys, HSE University.

The six supplied matrix-factorization functions are implemented using TensorFlow.js 4.22.0. The lecture's last slide defines the principal protocol: 16D user/movie vectors, **dot-product prediction**, chronological splitting, validation checkpoint selection, and training-only retrieval candidates.

## Run

From the repository root:

```sh
python -m http.server 8000
```

Open `http://localhost:8000/week5/`. The first load needs the pinned TensorFlow.js CDN runtime. The selected backend is displayed. Run **Run tests** before training, then **Train** with the provided defaults: 20 epochs, learning rate 0.01, L2 regularization 0.05, batch size 64. Select users 1 and 2 and press **Recommend Top-10**, then **Evaluate Recall@10**. Use the evidence download control to export selected factors and results. Saved factors permit independent recalculation without retraining.

## Data and evaluation

The supplied `u.data` and `u.item` are unchanged, byte-identical to Weeks 2 and 3. Titles are decoded as Latin-1/Windows-1252 to preserve accents. Timestamp splitting and genre alignment are retained.

- 100,000 distinct user/movie ratings; 943 users; 1,682 movies; no malformed or duplicate rows.
- Actual matrix: 943 x 1,682; missing fraction: **93.6953306358%**. Allocated raw-ID arrays include an unused zero-index row/column. Padding is excluded from the sparsity denominator. Storage zeros mark unobserved pairs; they are never training targets.
- Per-user sort by `(timestamp, movieId)`, `floor(0.8*n)` train, `floor(0.1*n)` validation, remainder test. The supplied splitter excludes users with fewer than five ratings; no users are excluded.
- Raw split: **79,619 / 9,596 / 10,785** ratings.
- Validation/test RMSE only on rows whose user AND movie occur in training: **9,561 / 10,732**. Exclusions: **35 / 53** rows, **32 / 46** distinct movies, zero unseen users. Baseline and MF use identical eligible rows.
- Training-only mean: **285,084 / 79,619 = 3.5806026199776437**. Eligible-test baseline RMSE: **1.2257193843347933**. Original all-row baseline: **1.2289001371836488**, with a different denominator.
- Adam and batched tensor operations minimize observed squared error. L2 is the batch mean of gathered user/item squared norms. No test data enters training or checkpoint selection.
- Both factor matrices are restored at the earliest minimum validation RMSE.
- Candidates: **1,611** globally training-seen movies, excluding the user's training and validation movies. Test movies remain eligible if observed for another user in training.
- Relevant test rating >=4. Of 5,155 relevant rows, 5,142 are eligible. Excluded: 13 distinct unseen relevant films and 82 users without an eligible relevant film. Macro Recall@10 averages hits/relevant over **861** evaluable users.

## Requirements and evidence

| Requirement | Implementation / evidence |
|---|---|
| Six supplied functions | script.js: predictRating, initializeFactors, trainMF, selectBestCheckpoint, recommendTopK, recallAtK |
| Checkpoint weights | snapshots and restoration of both factor matrices |
| Counts, sparsity, exclusions | page; evidence/data-audit.json; independent audit_data.py |
| Baseline / MF test RMSE | page; selected model export; report |
| Two lists and past-profile analysis | users 1 and 2; report and real browser screenshots |
| Candidate pool and macro recall | verify.cjs; model export; independent recalculation |
| Original self-checks | preserved verbatim; original and final JSON logs |
| Editable/PDF report | report.md and report.pdf |
| Original agent journal | separate raw JSONL deliverables, outside GitHub |

Run an independent data audit from week5:

```sh
python audit_data.py --repo . --output evidence
```

For additional contract checks use Node 18+ and the same official TF.js distribution (`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js`) saved locally as `tf.min.cjs`:

```sh
HW5_TFJS_PATH=/absolute/path/to/tf.min.cjs node verify.cjs
```

```powershell
$env:HW5_TFJS_PATH = 'C:\path\to\tf.min.cjs'
node verify.cjs
```

Additional checks report the original harness separately. They do not replace it or alter its expected assertions.

## Starter inconsistencies

Starter source: upstream commit `00c032fb58ca5f7bec05218b5bebee01e462e532`. Only Week 5 was imported, replacing the fork's obsolete PageRank files; earlier homework remains intact.

1. Lecture prediction is `p dot q`; starter describes `mu + p dot q` with a [1,5] clamp. This solution follows the lecture, with no offset or clipping for known pairs. The training mean remains a baseline and unknown-ID fallback. The original prediction assertion expects 5 for a dot product of 11.
2. The original recommendation fixture expects movie 3, although it never appears in training.
3. The original rank-11 fixture trains only films 1, 2, 3. After excluding film 1, relevant film 2 necessarily fits in Top-10. The additional rank-11 check makes every ranking candidate observed in another user's training data.
4. Selecting a history row alone does not restore its weights. Actual factors are now restored.
5. Suggested `optimizer.minimize(..., false).dispose()` can dereference null. Training explicitly manages tensor lifetime.

The original harness remains verbatim. Initial result: **4 pass, 0 fail, 7 pending**. The three incompatible assertions are reported honestly, with final measured results in evidence and the PDF.

The cited `docs/homework-guidelines/guidelines.md` is absent from the pinned upstream tree. The report uses an IEEE-inspired two-column layout and does not claim access to missing guidelines.

## Sources and AI use

- [Pinned starter](https://github.com/dryjins/RecSys-LLMs/tree/00c032fb58ca5f7bec05218b5bebee01e462e532/week5)
- [MovieLens release](https://grouplens.org/datasets/movielens/100k/)
- [Exact data README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
- [TF.js API](https://js.tensorflow.org/api/4.22.0/)
- F. Maxwell Harper and Joseph A. Konstan (2015), The MovieLens Datasets: History and Context, ACM TiiS 5(4), Article 19. [DOI 10.1145/2827872](https://doi.org/10.1145/2827872).

Codex implemented the code, ran browser/automated checks, calculated independent audits and drafted the report. Personal verification by Petr is not claimed. Retrospective relevance does not establish causal effects on streaming-platform retention or revenue.

## Measured default run

WebGL, seed 202605, 20 epochs, learning rate 0.01, regularization 0.05, batch size 64. Selected checkpoint: epoch 4. Validation RMSE **1.035942**; test RMSE **1.080030**, versus eligible-test mean baseline **1.225719**. Macro Recall@10 **0.0077765642** over 861 users. Retrieval is weak despite improved rating RMSE. Original checks: **8 pass / 3 conflicting failures / 0 pending**; additional contract checks: **11 pass / 0 fail / 0 skipped**.

Saved selected factors: `evidence/browser-run.json`. Independent NumPy recalculation: `evidence/independent-model-check.json`. To reproduce it from week5 (requires NumPy):

```sh
python verify_export.py evidence/browser-run.json --repo . --out evidence/independent-model-check.json
```

The report describes implementation commit `0066ff4776dd5ec32f90cd9bfb282b108f4daa31`; the later report commit does not change the algorithm.

## Published demonstration

[Open HW5 on GitHub Pages](https://petrvorozhtsov.github.io/RecSys-LLMs/week5/). Hosting uses the student fork's main branch and repository root. The successful deployment and MovieLens loading were checked; the model and report describe the same implementation. The raw coding-agent journal remains a separate submission file and is not hosted here.
