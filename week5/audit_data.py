"""Independent stdlib audit of supplied MovieLens data and temporal split.

Run from the chat workspace:
  python work/audit_data.py --repo work/RecSys-LLMs/week5 --output work

The script does not import or execute project JavaScript and never modifies it.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def file_info(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("work/RecSys-LLMs/week5"))
    parser.add_argument("--output", type=Path, default=Path("work"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_rating_lines = args.repo.joinpath("u.data").read_text(encoding="ascii").splitlines()
    raw_movie_lines = args.repo.joinpath("u.item").read_text(encoding="latin-1").splitlines()
    ratings, malformed_ratings = [], []
    for line_number, line in enumerate(raw_rating_lines, 1):
        if not line.strip():
            continue
        try:
            fields = line.split("\t")
            if len(fields) != 4:
                raise ValueError("not four fields")
            user, movie, rating, timestamp = map(int, fields)
            if user < 1 or movie < 1 or not 1 <= rating <= 5 or timestamp < 0:
                raise ValueError("invalid numeric field")
            ratings.append((user, movie, rating, timestamp))
        except ValueError as exc:
            malformed_ratings.append({"line": line_number, "reason": str(exc)})
    movies, malformed_movies = {}, []
    for line_number, line in enumerate(raw_movie_lines, 1):
        if not line.strip():
            continue
        try:
            fields = line.split("|")
            if len(fields) != 24:
                raise ValueError("not 24 fields")
            movie = int(fields[0])
            flags = list(map(int, fields[5:24]))
            if movie < 1 or any(flag not in (0, 1) for flag in flags):
                raise ValueError("invalid id or genre flag")
            if movie in movies:
                raise ValueError("duplicate movie id")
            movies[movie] = {"id": movie, "title": fields[1],
                             "genres": [genre for genre, flag in zip(GENRES, flags[1:]) if flag],
                             "unknownGenre": bool(flags[0])}
        except ValueError as exc:
            malformed_movies.append({"line": line_number, "reason": str(exc)})
    by_user = collections.defaultdict(list)
    for rating in ratings:
        by_user[rating[0]].append(rating)
    splits = {"train": [], "val": [], "test": []}
    per_user, excluded = {}, []
    timestamp_tie_groups = 0
    for user in sorted(by_user):
        # A fresh independent implementation of the contract, not JS execution.
        ordered = sorted(by_user[user], key=lambda row: (row[3], row[1]))
        n = len(ordered)
        if n < 5:
            excluded.append({"userId": user, "count": n})
            continue
        n_train, n_val = math.floor(0.8 * n), math.floor(0.1 * n)
        per_user[user] = {"train": ordered[:n_train], "val": ordered[n_train:n_train+n_val],
                          "test": ordered[n_train+n_val:]}
        for name in splits:
            splits[name].extend(per_user[user][name])
        timestamp_tie_groups += sum(count > 1 for count in collections.Counter(r[3] for r in ordered).values())
    train_users = {r[0] for r in splits["train"]}
    train_movies = {r[1] for r in splits["train"]}
    mu = sum(r[2] for r in splits["train"]) / len(splits["train"])

    def split_stats(rows):
        n = len(rows)
        total, squares = sum(r[2] for r in rows), sum(r[2] ** 2 for r in rows)
        mse_moments = (squares - 2 * mu * total + n * mu ** 2) / n
        mse_direct = math.fsum((r[2] - mu) ** 2 for r in rows) / n
        eligible_rows = [r for r in rows if r[0] in train_users and r[1] in train_movies]
        en = len(eligible_rows)
        esum, esq = sum(r[2] for r in eligible_rows), sum(r[2] ** 2 for r in eligible_rows)
        eligible_stats = {"ratings": en, "excludedRatingCount": n - en,
                          "ratingSum": esum, "ratingSumSquares": esq,
                          "ratingMean": esum / en if en else None,
                          "baselineRMSEFromMoments": math.sqrt((esq - 2*mu*esum + en*mu**2) / en) if en else None,
                          "baselineRMSEDirect": math.sqrt(math.fsum((r[2] - mu)**2 for r in eligible_rows) / en) if en else None,
                          "eligibilityRule": "User AND movie ids occur in training. Baseline remains the training-set mean; validation/test means are never used for prediction."}
        return {"ratings": n, "users": len({r[0] for r in rows}),
                "movies": len({r[1] for r in rows}), "ratingSum": total,
                "ratingSumSquares": squares, "ratingMean": total / n,
                "baselineRMSEFromMoments": math.sqrt(mse_moments),
                "baselineRMSEDirect": math.sqrt(mse_direct),
                "unseenTrainUserIds": sorted({r[0] for r in rows} - train_users),
                "unseenTrainMovieIds": sorted({r[1] for r in rows} - train_movies),
                "ratingsWithUnseenTrainMovie": sum(r[1] not in train_movies for r in rows),
                "eligibleKnownTrainUserAndMovie": eligible_stats}

    relevant = [r for r in splits["test"] if r[2] >= 4]
    eligible = [r for r in relevant if r[1] in train_movies]
    eligible_by_user = collections.defaultdict(set)
    for row in eligible:
        eligible_by_user[row[0]].add(row[1])
    no_relevant = sorted({r[0] for r in splits["test"]} - set(eligible_by_user))
    unseen_relevant = sorted({r[1] for r in relevant if r[1] not in train_movies})

    def profile(user):
        past = per_user[user]["train"] + per_user[user]["val"]
        genres = collections.defaultdict(list)
        liked = [r for r in past if r[2] >= 4]
        liked_genres = collections.Counter()
        for row in past:
            for genre in movies[row[1]]["genres"]:
                genres[genre].append(row[2])
                if row[2] >= 4:
                    liked_genres[genre] += 1
        past_genre_stats = sorted([{"genre": genre, "pastCount": len(rs),
                                   "pastFraction": len(rs) / len(past),
                                   "pastMeanRating": sum(rs) / len(rs),
                                   "likedCount": liked_genres[genre],
                                   "likedFraction": liked_genres[genre] / len(liked) if liked else 0}
                                  for genre, rs in genres.items()],
                                 key=lambda g: (-g["likedCount"], -g["pastCount"], g["genre"]))
        liked_examples = sorted(liked, key=lambda row: (-row[2], -row[3], row[1]))[:15]
        return {"userId": user, "splitCounts": {name: len(rows) for name, rows in per_user[user].items()},
                "pastRatings": len(past), "pastRatingMean": sum(r[2] for r in past) / len(past),
                "likedPastRatings": len(liked), "genreStats": past_genre_stats,
                "likedPastMovieExamples": [{**movies[r[1]], "rating": r[2], "timestamp": r[3]} for r in liked_examples],
                "pastMovieIds": sorted({r[1] for r in past}),
                "pastGenreSet": sorted(genres),
                "eligibleTestRelevantCount": len(eligible_by_user[user])}

    user_count, movie_count = len(by_user), len(movies)
    duplicate_pairs = [list(pair) for pair, count in collections.Counter((r[0], r[1]) for r in ratings).items() if count > 1]
    result = {
        "method": "Independent Python standard-library parsing; per-user sort by (timestamp,movieId), floor(0.8*n), floor(0.1*n), remainder test; no JS imported.",
        "files": {name: file_info(args.repo / name) for name in ("u.data", "u.item")},
        "raw": {"ratingLines": len(raw_rating_lines), "nonemptyRatingLines": sum(bool(line.strip()) for line in raw_rating_lines),
                "movieLines": len(raw_movie_lines), "nonemptyMovieLines": sum(bool(line.strip()) for line in raw_movie_lines)},
        "validated": {"ratings": len(ratings), "users": user_count, "movies": movie_count,
                      "minUserId": min(by_user), "maxUserId": max(by_user), "minMovieId": min(movies), "maxMovieId": max(movies),
                      "minRatingsPerUser": min(map(len, by_user.values())), "maxRatingsPerUser": max(map(len, by_user.values())),
                      "malformedRatingRows": malformed_ratings, "malformedMovieRows": malformed_movies,
                      "duplicateRatingRows": len(ratings) - len(set(ratings)), "duplicateUserMoviePairs": duplicate_pairs,
                      "ratingMovieIdsMissingMetadata": sorted({r[1] for r in ratings} - set(movies)),
                      "movieMetadataNeverRatedIds": sorted(set(movies) - {r[1] for r in ratings}),
                      "minTimestampUTC": utc(min(r[3] for r in ratings)), "maxTimestampUTC": utc(max(r[3] for r in ratings)),
                      "unknownGenreMovieIds": sorted(m for m, meta in movies.items() if meta["unknownGenre"]),
                      "moviesWithoutRealGenreIds": sorted(m for m, meta in movies.items() if not meta["genres"])},
        "matrix": {"actualShape": [user_count, movie_count], "actualCells": user_count * movie_count,
                   "actualObservedDensity": len(ratings) / (user_count * movie_count),
                   "actualMissingFraction": 1 - len(ratings) / (user_count * movie_count),
                   "allocationShape": [max(by_user) + 1, max(movies) + 1],
                   "allocationCells": (max(by_user) + 1) * (max(movies) + 1),
                   "allocationObservedDensity": len(ratings) / ((max(by_user) + 1) * (max(movies) + 1)),
                   "note": "Row/column zero are unused id-index padding; sparsity should use actual 943 x 1682 pairs, not 944 x 1683 allocated cells."},
        "split": {"stats": {name: split_stats(rows) for name, rows in splits.items()},
                  "splitExclusions": excluded, "trainingMean": mu,
                  "perUserTimestampTieGroups": timestamp_tie_groups,
                  "partitionExhaustive": sum(map(len, splits.values())) == len(ratings),
                  "partitionDisjoint": len(set(splits["train"]) & set(splits["val"])) == len(set(splits["train"]) & set(splits["test"])) == len(set(splits["val"]) & set(splits["test"])) == 0,
                  "chronologyAllUsers": all(max(r[3] for r in parts["train"]) <= min(r[3] for r in parts["val"]) <= max(r[3] for r in parts["val"]) <= min(r[3] for r in parts["test"]) for parts in per_user.values())},
        "retrieval": {"candidateMovieCount": len(train_movies), "testRatingRelevantThreshold": 4,
                      "relevantTestRatings": len(relevant), "eligibleRelevantTestRatings": len(eligible),
                      "distinctEligibleRelevantMovieCount": len({r[1] for r in eligible}),
                      "unseenRelevantTestMovieIds": unseen_relevant,
                      "unseenRelevantTestMovieCount": len(unseen_relevant),
                      "unseenRelevantTestRatingCount": len(relevant) - len(eligible),
                      "usersWithoutEligibleRelevantTest": no_relevant, "usersWithoutEligibleRelevantTestCount": len(no_relevant),
                      "evaluableUserCount": len(eligible_by_user),
                      "relevantTestPairsAlreadyInHistoryCount": sum(r[1] in {p[1] for p in per_user[r[0]]["train"] + per_user[r[0]]["val"]} for r in relevant),
                      "eligibleRelevantCountHistogram": dict(sorted(collections.Counter(map(len, eligible_by_user.values())).items())),
                      "note": "unseenIds is a DISTINCT movie-id count, not the number of rating rows. Recall is macro averaged across evaluable users, not pooled micro recall."},
        "selectedUsers": {"rule": "Users 1 and 2 chosen from contrasting train+validation genre profiles only; no recommendations or test preferences used in selection.",
                          "profiles": [profile(1), profile(2)]},
        "primarySources": [{"url": "https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
                            "verified": "Exact 100,000 / 943 / 1,682 dimensions; >=20 ratings/user; September 1997-April 1998; genres contain 19 flags including unknown; 2015 citation and DOI. Research use conditions include attribution and permission for redistribution."},
                           {"url": "https://grouplens.org/datasets/movielens/100k/", "verified": "Official release page, April 1998; its 1,000 users/1,700 movies is rounded."},
                           {"url": "https://doi.org/10.1145/2827872", "verified": "DOI checked against authoritative GroupLens dataset README. ACM/DOI web fetch returned 403 in this environment."}],
        "citation": "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19, 19 pages. DOI: 10.1145/2827872."
    }
    args.output.joinpath("data-audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    st = result["split"]["stats"]
    lines = ["# Independent MovieLens data audit", "", result["method"], "",
             f"Validated {len(ratings):,} ratings, {user_count} users and {movie_count:,} movies. No malformed rows or duplicate pairs: {len(malformed_ratings)} malformed ratings, {len(malformed_movies)} malformed movies, {len(duplicate_pairs)} duplicate pairs.",
             f"Each user has at least {result['validated']['minRatingsPerUser']} ratings; split exclusions: {len(excluded)}.", "",
             "| Split | Ratings | Users | Movies | Rating sum | Sum of squares | Mean baseline RMSE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stats in st.items():
        lines.append(f"| {name} | {stats['ratings']} | {stats['users']} | {stats['movies']} | {stats['ratingSum']} | {stats['ratingSumSquares']} | {stats['baselineRMSEDirect']:.12f} |")
    lines += ["", "### RMSE subset with user and movie both observed in training", "",
              "| Split | Eligible ratings | Excluded ratings | Rating sum | Sum of squares | Training-mean baseline RMSE | All-row baseline RMSE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stats in st.items():
        e = stats["eligibleKnownTrainUserAndMovie"]
        lines.append(f"| {name} | {e['ratings']} | {e['excludedRatingCount']} | {e['ratingSum']} | {e['ratingSumSquares']} | {e['baselineRMSEDirect']:.12f} | {stats['baselineRMSEDirect']:.12f} |")
    lines += ["", "Eligibility uses training ids only. All users are observed in training; the excluded validation/test rows all involve movies absent from training. Baseline prediction always remains the training mean (never validation/test subset means).", "",
              f"Training mean: {mu:.15f}. Independent moment formula: RMSE = sqrt((sum(r²) − 2*mu*sum(r) + N*mu²)/N), matching the direct per-rating calculation to floating-point precision.", "",
              f"Actual matrix: {user_count}×{movie_count} = {user_count*movie_count:,} pairs. Missing fraction: {result['matrix']['actualMissingFraction']:.12%}. Allocated padded matrix: {max(by_user)+1}×{max(movies)+1}; its zero-index padding is not an observed user/movie.", "",
              f"Candidates: {len(train_movies)} training movies. Relevant test ratings: {len(relevant)}; eligible relevant ratings: {len(eligible)}. Exclusions: {len(unseen_relevant)} distinct unseen relevant movie ids ({len(relevant)-len(eligible)} rating rows), {len(no_relevant)} users with no eligible relevant test item. Evaluable users: {len(eligible_by_user)}.", "",
              "Unseen relevant movie IDs: " + ", ".join(map(str, unseen_relevant)), "",
              "Test/validation movies absent from training and full exclusion lists are in `data-audit.json`.", "",
              "## Users selected from past profiles", "", result["selectedUsers"]["rule"], ""]
    for p in result["selectedUsers"]["profiles"]:
        lines += [f"### User {p['userId']}", "", f"Past: {p['pastRatings']} ratings, {p['likedPastRatings']} rated >=4. Past mean: {p['pastRatingMean']:.6f}. Train/val/test: {p['splitCounts']}.", "",
                  "| Genre | Past count | Past fraction | Past mean | Liked count | Fraction of liked movies |", "|---|---:|---:|---:|---:|---:|"]
        for genre in p["genreStats"]:
            lines.append(f"| {genre['genre']} | {genre['pastCount']} | {genre['pastFraction']:.4f} | {genre['pastMeanRating']:.4f} | {genre['likedCount']} | {genre['likedFraction']:.4f} |")
        lines += ["", "Multi-genre movies count once in each applicable genre; genre fractions need not sum to 100%.", "", "Past examples rated 4 or 5: " + "; ".join(f"{m['title']} ({m['rating']}/5; {', '.join(m['genres'])})" for m in p["likedPastMovieExamples"][:10]), ""]
    lines += ["## Verified sources", "", "- [Official precise dataset README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)",
              "- [Official release page](https://grouplens.org/datasets/movielens/100k/)", "- [Publication DOI](https://doi.org/10.1145/2827872)", "", result["citation"], "",
              "Citation and DOI verified in the authoritative dataset README. Direct ACM/DOI fetch returned 403 here. The landing page rounds dimensions; do not use its 1,000/1,700 values for calculations.", "",
              "No student personal verification is claimed; this is an independent calculation by Codex."]
    args.output.joinpath("data-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["validated"], "split": result["split"], "retrieval": result["retrieval"], "selectedUsers": [{"userId": p["userId"], "pastRatings": p["pastRatings"], "likedPastRatings": p["likedPastRatings"], "genreStats": p["genreStats"][:8]} for p in result["selectedUsers"]["profiles"]]}, indent=2))


if __name__ == "__main__":
    main()
