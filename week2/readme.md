# Content-Based Movie Recommender

This Week 2 project implements the content-based filtering assignment from the lecture using the MovieLens 100K dataset.

## What is implemented

- Movie metadata is represented by an 18-dimensional binary genre vector (the MovieLens `unknown` field is excluded).
- Cosine similarity replaces the original Jaccard similarity baseline.
- **Item-to-item** recommendations compare one active movie with every other movie.
- **Profile-based** recommendations average the vectors of up to three liked movies and compare the resulting profile with the catalog.
- Both approaches return Top-5 recommendations and exclude the movies used as input.
- The page compares genre coverage and long-tail discovery. A long-tail item is one with fewer ratings than the catalog median.
- The UI explains why cosine normalization reduces the dominance of multi-genre blockbusters.

## Run locally

Because browsers restrict `fetch()` from `file://` pages, serve this directory with any static HTTP server, for example:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/` and choose an active movie plus three liked movies.

## Files

- `index.html` — interface for the active item and three-movie profile.
- `data.js` — MovieLens loading and parsing.
- `script.js` — cosine similarity, profile aggregation, ranking, and analysis.
- `style.css` — responsive presentation.
- `u.item`, `u.data` — MovieLens 100K metadata and ratings.
