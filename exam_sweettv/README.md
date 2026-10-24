# SWEET.TV midterm

This folder contains the exam implementation. The pipeline uses only labelled
schedule rows, joins a view to the schedule by channel and interval overlap,
uses weeks 11–20 for training and weeks 21–30 for temporal validation, then
re-trains on weeks 11–30 for the weeks 31–42 submission.

`run.py` evaluates the required popularity reference plus content-based and
association recommenders. It writes `outputs/submission.csv`, validation
MAP@5, audit counts, and two example users. `tv_show_id = 0` rows are dropped.
The join threshold is at least 60 seconds of overlap and at least 10% of the
scheduled programme duration. When several labelled rows overlap, the row with
the largest overlap wins.

Do not commit the Kaggle archive or extracted CSVs. Keep the unmodified AI
session export with the final report as required by the course.
