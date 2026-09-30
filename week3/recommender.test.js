// Assignment point: verify sparse similarities, cold start, and Top-5 behavior.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = vm.createContext({ window: {}, document: {}, console });
for (const name of ['data.js', 'script.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, name), 'utf8'), context);
}
context.itemText = fs.readFileSync(path.join(__dirname, 'u.item'), 'latin1');
context.ratingText = fs.readFileSync(path.join(__dirname, 'u.data'), 'utf8');
vm.runInContext(`
    parseItemData(itemText);
    parseRatingData(ratingText);
    numUsers = ratings.reduce((max, row) => Math.max(max, row.userId), 0);
    numMovies = movies.reduce((max, movie) => Math.max(max, movie.id), 0);
    buildRatingMatrix();
    resetRecommendationCaches();
`, context);

test('MovieLens loads into the expected user-item matrix', () => {
    const counts = vm.runInContext(`({
        users: numUsers, movies: movies.length, ratings: ratings.length,
        rows: ratingMatrix.length, columns: ratingMatrix[1].length
    })`, context);
    assert.deepEqual(JSON.parse(JSON.stringify(counts)), {
        users: 943, movies: 1682, ratings: 100000, rows: 944, columns: 1683
    });
});

test('weighted cosine rejects tiny overlaps and favors stronger evidence', () => {
    const scores = vm.runInContext(`({
        none: cosineSimilarity([0, 5, 0], [0, 0, 5]),
        one: cosineSimilarity([0, 5, 0], [0, 4, 5]),
        two: cosineSimilarity([0, 5, 5], [0, 4, 4]),
        four: cosineSimilarity([0, 5, 5, 5, 5], [0, 4, 4, 4, 4])
    })`, context);
    assert.equal(scores.none, 0);
    assert.equal(scores.one, 0);
    assert.ok(scores.two > 0 && scores.two < scores.four && scores.four < 1);
});

test('a new user without ratings receives no collaborative recommendations', () => {
    const result = vm.runInContext(`(() => {
        ratingMatrix[numUsers + 1] = Array(numMovies + 1).fill(0);
        const userBased = getUserBasedRecommendations(numUsers + 1);
        const itemBased = getItemBasedRecommendations(numUsers + 1);
        ratingMatrix.pop();
        return { userBased, itemBased };
    })()`, context);
    assert.equal(result.userBased.length, 0);
    assert.equal(result.itemBased.length, 0);
});

test('both Top-5 lists are finite, supported, and exclude rated movies', () => {
    const result = vm.runInContext(`(() => {
        const userId = 196;
        const lists = [getUserBasedRecommendations(userId),
            getItemBasedRecommendations(userId)];
        const titleToId = new Map(movies.map(movie => [movie.title, movie.id]));
        return lists.map(list => list.map(item => ({
            ...item, alreadyRated: ratingMatrix[userId][titleToId.get(item.title)] > 0
        })));
    })()`, context);
    for (const list of result) {
        assert.equal(list.length, 5);
        for (const item of list) {
            assert.ok(Number.isFinite(item.score) && item.score >= 1 && item.score <= 5);
            assert.ok(item.support >= 2);
            assert.equal(item.alreadyRated, false);
        }
    }
});

test('cached item similarities preserve recommendation results', () => {
    const result = vm.runInContext(`(() => {
        const first = getItemBasedRecommendations(1);
        resetRecommendationCaches();
        const second = getItemBasedRecommendations(1);
        return { first, second };
    })()`, context);
    assert.deepEqual(JSON.parse(JSON.stringify(result.first)),
        JSON.parse(JSON.stringify(result.second)));
});
