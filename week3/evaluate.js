// Assignment point: reproducible held-out accuracy and efficiency comparison.
// Run with `node evaluate.js` from any directory; no external packages are used.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { performance } = require('node:perf_hooks');

const directory = __dirname;
const context = vm.createContext({ console, window: {}, document: {}, performance });
for (const name of ['data.js', 'script.js']) {
    vm.runInContext(fs.readFileSync(path.join(directory, name), 'utf8'), context);
}

context.itemText = fs.readFileSync(path.join(directory, 'u.item'), 'latin1');
context.ratingText = fs.readFileSync(path.join(directory, 'u.data'), 'utf8');
vm.runInContext(`
    parseItemData(itemText);
    parseRatingData(ratingText);
    numUsers = ratings.reduce((max, row) => Math.max(max, row.userId), 0);
    numMovies = movies.reduce((max, movie) => Math.max(max, movie.id), 0);
`, context);

const allRatings = vm.runInContext('ratings', context);
const userRatings = new Map();
const itemCounts = new Map();
for (const row of allRatings) {
    if (!userRatings.has(row.userId)) userRatings.set(row.userId, []);
    userRatings.get(row.userId).push(row);
    itemCounts.set(row.itemId, (itemCounts.get(row.itemId) || 0) + 1);
}

// Assignment point: reserve five reproducible warm-start ratings per sampled user.
// The active user's hidden ratings are removed before any similarity or prior is built.
const evaluationUsers = Array.from({ length: 50 }, (_, index) => 1 + index * 19);
const hidden = [];
for (const userId of evaluationUsers) {
    const eligible = userRatings.get(userId)
        .filter(row => itemCounts.get(row.itemId) >= 10)
        .sort((a, b) => {
            const rankA = (Math.imul(a.itemId, 2654435761) ^
                Math.imul(userId, 1013904223)) >>> 0;
            const rankB = (Math.imul(b.itemId, 2654435761) ^
                Math.imul(userId, 1013904223)) >>> 0;
            return rankA - rankB || a.itemId - b.itemId;
        });
    if (userRatings.get(userId).length < 20 || eligible.length < 5) {
        throw new Error(`User ${userId} has too few eligible ratings for evaluation`);
    }
    hidden.push(...eligible.slice(0, 5));
}

const hiddenKeys = new Set(hidden.map(row => `${row.userId}:${row.itemId}`));
context.trainingRatings = allRatings.filter(row =>
    !hiddenKeys.has(`${row.userId}:${row.itemId}`));
vm.runInContext('ratings = trainingRatings; buildRatingMatrix(); resetRecommendationCaches();', context);
if (process.env.EVAL_PROGRESS) console.error('Training matrix ready');

context.hiddenJson = JSON.stringify(hidden);
context.evaluationUsersJson = JSON.stringify(evaluationUsers);
// Execute model calls inside the same VM as the browser code to avoid cross-realm overhead.
const totals = vm.runInContext(`(() => {
    const hiddenRows = JSON.parse(hiddenJson);
    const userIds = JSON.parse(evaluationUsersJson);
    const result = {
        userBased: { evaluated: 0, absoluteError: 0, squaredError: 0 },
        itemBased: { evaluated: 0, absoluteError: 0, squaredError: 0 },
        pairedUserBased: { evaluated: 0, absoluteError: 0, squaredError: 0 },
        pairedItemBased: { evaluated: 0, absoluteError: 0, squaredError: 0 }
    };
    for (const userId of userIds) {
        const neighbors = getTopNeighbors(userId);
        const ratedMovieIds = getRatedMovieIds(userId);
        for (const row of hiddenRows.filter(item => item.userId === userId)) {
            const predictions = {
                userBased: predictUserBasedRating(row.itemId, neighbors),
                itemBased: predictItemBasedRating(userId, row.itemId, ratedMovieIds)
            };
            for (const [method, prediction] of Object.entries(predictions)) {
                if (!prediction) continue;
                const error = prediction.score - row.rating;
                result[method].evaluated += 1;
                result[method].absoluteError += Math.abs(error);
                result[method].squaredError += error * error;
                if (predictions.userBased && predictions.itemBased) {
                    const paired = result[method === 'userBased' ?
                        'pairedUserBased' : 'pairedItemBased'];
                    paired.evaluated += 1;
                    paired.absoluteError += Math.abs(error);
                    paired.squaredError += error * error;
                }
            }
        }
    }
    return result;
})()`, context);
if (process.env.EVAL_PROGRESS) console.error('Held-out ratings scored');

const round = value => Math.round(value * 1000) / 1000;
const summarize = total => ({
    evaluated: total.evaluated,
    coverage: round(total.evaluated / hidden.length),
    mae: total.evaluated ? round(total.absoluteError / total.evaluated) : null,
    rmse: total.evaluated ? round(Math.sqrt(total.squaredError / total.evaluated)) : null
});

// Assignment point: time the actual Top-5 methods on the full MovieLens matrix.
// Cold item timings clear both caches; warm timings reuse the same user's item pairs.
context.fullRatings = allRatings;
vm.runInContext('ratings = fullRatings; buildRatingMatrix(); resetRecommendationCaches();', context);
const benchmarkUsers = [1, 100, 196, 300, 500];
context.benchmarkUsersJson = JSON.stringify(benchmarkUsers);
const times = vm.runInContext(`(() => {
    const users = JSON.parse(benchmarkUsersJson);
    const result = { userBased: [], itemBasedCold: [], itemBasedWarm: [] };
    for (const userId of users) {
        resetRecommendationCaches();
        let start = performance.now();
        getUserBasedRecommendations(userId);
        result.userBased.push(performance.now() - start);

        start = performance.now();
        getItemBasedRecommendations(userId);
        result.itemBasedCold.push(performance.now() - start);

        start = performance.now();
        getItemBasedRecommendations(userId);
        result.itemBasedWarm.push(performance.now() - start);
    }
    return result;
})()`, context);
if (process.env.EVAL_PROGRESS) console.error('Full-dataset benchmark complete');

const median = numbers => {
    const sorted = [...numbers].sort((a, b) => a - b);
    return round(sorted[Math.floor(sorted.length / 2)]);
};

console.log(JSON.stringify({
    dataset: { users: 943, movies: 1682, ratings: allRatings.length },
    model: { minimumCommonRatings: 2, overlapShrinkage: 10,
        predictionPriorWeight: 1, minimumItemRatings: 5 },
    split: { sampledUsers: evaluationUsers.length, heldOutRatings: hidden.length,
        trainingRatings: context.trainingRatings.length,
        ratingsPerUser: 5, minimumOriginalItemRatings: 10 },
    quality: {
        userBased: summarize(totals.userBased),
        itemBased: summarize(totals.itemBased),
        pairedUserBased: summarize(totals.pairedUserBased),
        pairedItemBased: summarize(totals.pairedItemBased)
    },
    timing: { users: benchmarkUsers,
        runtime: `${process.platform} / Node ${process.version}`,
        medianMilliseconds: {
            userBased: median(times.userBased),
            itemBasedCold: median(times.itemBasedCold),
            itemBasedWarm: median(times.itemBasedWarm)
        }
    }
}, null, 2));
