# Collaborative Filtering Movie Recommender

This Week 3 project implements both User-Based and Item-Based collaborative filtering using the same MovieLens 100K `u.item` and `u.data` files as Week 2. The files are byte-identical across the two exercises. The site uses vanilla HTML, CSS, and JavaScript and has no build step.

## What I implemented

- `data.js` parses 1,682 movie titles and 100,000 ratings from 943 users, then builds a user-item matrix. A zero means "not rated"; actual MovieLens ratings range from 1 to 5.
- `script.js` computes User-Based CF with the Top-20 nearest users and Item-Based CF with similarities between movies. Both exclude movies the active user already rated and show Top-5 predicted ratings plus evidence counts.
- The assignment's sparse-rating strategy is **weighted by the number of common ratings**. For the set `C` of positions rated on both sides, `similarity = cosine(a_C, b_C) * |C| / (|C| + 10)`. At least two common ratings are required. This applies to both user and item pairs; missing entries never become dislikes.
- A prediction is `(sum(similarity * rating) + training_mean) / (sum(similarity) + 1)`. The training mean is a small prior that prevents one strong rating from determining an extreme score. Candidates need at least two contributing neighbors or rated movies; Item-Based candidates also need five community ratings.
- Item rating vectors and pairwise item similarities are cached for reuse. The caches are cleared whenever a new matrix is loaded.
- The UI discusses model efficiency, missing-value trade-offs, and cold start. The reported prediction score is **not** an accuracy metric.

## Reproduce the quality and efficiency comparison

```bash
node evaluate.js
node --test recommender.test.js
```

`evaluate.js` uses only Node's built-in modules. It deterministically samples 50 users (IDs `1, 20, ..., 932`) and hides five ratings per user, selecting from movies with at least ten original ratings. All similarities, item counts, and the global rating prior are rebuilt from the remaining 99,750 ratings. MAE and RMSE compare predictions with the 250 hidden true ratings; coverage is the fraction for which a method can predict. Paired errors compare the same hidden ratings for both methods. This is a small warm-start sample, not a full-dataset ranking evaluation.

In the reference run, User-Based covered 248/250 ratings and Item-Based covered 250/250. On the 248 paired cases, User-Based MAE/RMSE were **0.838/1.050**, while Item-Based were **0.790/0.962**. These numbers are evidence about predicted rating accuracy on this sample; they do not establish Top-5 relevance for every user.

The script also times the real Top-5 functions for Users 1, 100, 196, 300, and 500 on the full dataset. It reports median User-Based, cold Item-Based, and warm Item-Based time. Cold item timing clears the caches before each user; warm timing repeats the same user's request. The saved reference run on Windows with Node 24.19.0 gave approximately **13 ms / 351 ms / 56 ms** respectively. Timings depend on the machine and cache state. In MovieLens 100K there are more items (1,682) than users (943), so a complete user-pair model has fewer pairs than a complete item-pair model. The opposite is generally true when users outnumber items.

## Trade-offs and limitations

- **Mean imputation:** easy and dense, but filling missing ratings with averages can hide individual taste and introduce bias.
- **Common-rating weighting (chosen):** simple and less trusting of one or two overlaps, but can still miss useful relationships in a sparse matrix.
- **Matrix factorization:** can learn latent preferences from sparse data, but requires training, tuning, and more computation; it is harder to explain directly.
- **Cold start:** a user without ratings or an item without community ratings has no collaborative signal. The minimum-evidence rules intentionally leave some candidates without a prediction.
- **Evaluation scope:** the deterministic holdout has 250 ratings from 50 existing users; it does not test new users, new items, or ranking quality on unobserved preferences.

## Run locally

From this `week3` directory, start any static HTTP server (browsers restrict `fetch()` from `file://` pages):

```bash
python -m http.server 8000
```

Open `http://localhost:8000/`, select a user, and click **Get Recommendations**.

## Files

- `index.html` - active-user selector, two Top-5 panels, and model analysis.
- `style.css` - responsive layout.
- `data.js` - MovieLens loading, parsing, and rating matrix.
- `script.js` - weighted similarities, predictions, ranking, caching, and rendering.
- `evaluate.js` - deterministic holdout and timing comparison.
- `evaluation-results.json` - one saved reference run of the evaluation.
- `recommender.test.js` - sparse-data, cold-start, and result checks.
- `u.item`, `u.data` - unchanged MovieLens 100K files.
- `Petr_Vorozhtsov_Week3_Report.pdf` - student report with screenshots and results.
