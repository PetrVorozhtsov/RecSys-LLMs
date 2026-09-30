# Collaborative Filtering Movie Recommender

This Week 3 project implements the collaborative filtering assignment from the lecture using the same MovieLens 100K files as Week 2.

## What is implemented

- `data.js` loads `u.item` and `u.data`, parses movies and ratings, and builds a user-item rating matrix.
- Missing ratings are stored as `0`; MovieLens ratings are `1-5`, so this is unambiguous.
- `script.js` uses cosine similarity over co-rated non-zero entries only.
- **User-Based CF** compares the active user with all other users, keeps the Top-20 positive neighbors, and predicts unseen movies with a similarity-weighted average.
- **Item-Based CF** compares unseen candidate movies with the movies already rated by the active user and aggregates similarities weighted by the user's ratings.
- Both methods return Top-5 recommendation lists with predicted scores.
- The UI displays both approaches side by side and includes the requested business and algorithmic analysis: efficiency, missing-value trade-offs, cold start, and sparsity.

## Run locally

Because browsers restrict `fetch()` from `file://` pages, serve this directory with any static HTTP server, for example:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/`, select a user, and click **Get Recommendations**.

## Files

- `index.html` — interface with user selection and two recommendation sections.
- `style.css` — responsive presentation.
- `data.js` — MovieLens loading, parsing, and rating matrix construction.
- `script.js` — cosine similarity, User-Based CF, Item-Based CF, rendering, and analysis.
- `u.item`, `u.data` — MovieLens 100K metadata and ratings.
