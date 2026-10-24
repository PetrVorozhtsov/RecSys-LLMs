# SWEET.TV TV-program recommender

**Student:** Пётр Ворожцов  
**Kaggle:** PetrVorozhtsovLMS  
**Repository:** https://github.com/PetrVorozhtsov/RecSys-LLMs/tree/exam-sweettv/exam_sweettv

## Data and protocol

The view log was joined to labelled schedule rows by channel and interval
overlap. A view was counted when overlap was at least 60 seconds and at least
10% of the scheduled programme duration. Rows with `tv_show_id = 0` were
excluded. Weeks 11–20 were training, weeks 21–30 were temporal validation,
and weeks 31–42 were the blind test period. No test-period user views entered
training or features.

## Methods

1. Popularity reference: unique training viewers per programme.
2. Content-based filtering: category, genres, year bucket, director and actor
   tokens; duration-weighted user profile; cosine ranking.
3. Association recommender: user-history programme pairs ranked by confidence
   and lift-weighted co-view evidence.

All methods used the same candidate pool and removed programmes already seen
by the user. Users without usable history received the same popularity fallback.

## Validation

| Method | MAP@5 |
|---|---:|
| Popularity | 0.162263 |
| Content-based | 0.008688 |
| Association | 0.157865 |

The final method was selected using validation MAP@5 only.

## Two-user comparison

Use the same heavy/light or cold-start users for every method. Report programme
IDs, titles, categories, and comparison with each user's training history.

## Strategy and limitations

Discuss the selected method using validation MAP@5, computational cost and the
two recommendation lists. State the limitations supported by the data: only a
minority of schedule rows have programme IDs, the view log covers platform
behaviour only, cold-start cases exist, and no A/B or revenue claim is made.

## Reproducibility

Record runtime, Python version, threshold values, random seed (if any), join
counts, candidate counts, and the exact Kaggle submission screenshot.

The supplied Kaggle screenshot records Private Score 0.01454 and Public Score
0.01740 for `submission.csv`.
