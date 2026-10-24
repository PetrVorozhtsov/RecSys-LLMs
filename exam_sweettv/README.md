# SWEET.TV midterm

This folder contains the exam implementation. The pipeline uses only labelled
schedule rows, joins a view to the schedule by channel and interval overlap,
uses weeks 11–20 for training and weeks 21–30 for temporal validation. For
the weeks 31–42 submission it combines weeks 11–30 user histories and fallback
popularity, but retains the association pairs fitted on weeks 11–20.

`run.py` evaluates the required popularity reference plus content-based and
association recommenders. It writes `outputs/submission.csv`, validation
MAP@5, audit counts, and two example users. `tv_show_id = 0` rows are dropped.
The join threshold is at least 60 seconds of overlap and at least 10% of the
scheduled programme duration. When several labelled rows overlap, the row with
the largest overlap wins.

Do not commit the Kaggle archive or extracted CSVs. Keep the unmodified AI
session export with the final report as required by the course.

The detailed English report is `report_v2.pdf`. Regenerate it with
`python make_report_v2.py`; the English content modules and the source data
must be available. `python verify_report.py` checks the submitted CSV and the
reported example lists without changing them. `python timed_reproduce.py`
runs the unchanged model into a separate output directory and records timing
and byte comparisons; it never overwrites the submitted output files.

The submitted implementation has known limitations documented in the report:
content cold start does not use the common popularity fallback, validation-only
users receive empty lists in local evaluation, content metadata includes later
schedule attributes, and the lift denominator uses summed item supports.
The submitted association method was selected against content, while popularity
has a higher recorded local MAP@5. No claim of full implementation compliance
is made.
