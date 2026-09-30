// Assignment point: both recommendation strategies must return a Top-5 list.

const TOP_K = 5;
const NEIGHBOR_COUNT = 20;
const MIN_COMMON_RATINGS = 2;
const OVERLAP_SHRINKAGE = 10;
const PRIOR_WEIGHT = 1;
const MIN_ITEM_RATINGS = 5;

// Assignment point: use the weighted-by-common-ratings strategy from HW3.
// Zero means "not rated". Cosine uses observed pairs and shrinks small overlaps;
// predictions also shrink sparse evidence toward the training-set mean.

window.onload = async function initialize() {
    setStatus('Loading MovieLens data...', 'loading');
    try {
        await loadData();
        resetRecommendationCaches();
        populateUserDropdown();
        setStatus(`Loaded ${numUsers.toLocaleString()} users, ${movies.length.toLocaleString()} movies, and ${ratings.length.toLocaleString()} ratings.`, 'success');
        clearRecommendationPanels('Select a user and generate recommendations.');
    } catch (error) {
        console.error('Initialization error:', error);
        setStatus('Could not load MovieLens data.', 'error');
    }
};

function setStatus(message, className = '') {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status ${className}`.trim();
}

function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');
    selectElement.innerHTML = '<option value="" disabled selected>Select a user</option>';

    for (let userId = 1; userId <= numUsers; userId += 1) {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        selectElement.appendChild(option);
    }

    if (numUsers > 0) selectElement.value = '1';
}

// Assignment point: weighted cosine similarity between two sparse rating vectors.

function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    let commonCount = 0;
    const length = Math.min(a.length, b.length);

    for (let index = 0; index < length; index += 1) {
        if (a[index] === 0 || b[index] === 0) continue;
        dot += a[index] * b[index];
        normA += a[index] * a[index];
        normB += b[index] * b[index];
        commonCount += 1;
    }

    // One shared rating cannot establish a reliable user or item relationship.
    if (commonCount < MIN_COMMON_RATINGS) return 0;
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : (dot / denominator) *
        (commonCount / (commonCount + OVERLAP_SHRINKAGE));
}

function getMovieTitle(movieId) {
    const movie = movies.find(item => item.id === movieId);
    return movie ? movie.title : `Movie ${movieId}`;
}

function getRatedMovieIds(userId) {
    const vector = ratingMatrix[userId] || [];
    const result = [];
    for (let movieId = 1; movieId <= numMovies; movieId += 1) {
        if (vector[movieId] > 0) result.push(movieId);
    }
    return result;
}

// Assignment point: User-Based CF with reliability-weighted neighbors.

function getTopNeighbors(activeUserId) {
    const activeVector = ratingMatrix[activeUserId];
    if (!activeVector) return [];

    const neighbors = [];
    for (let userId = 1; userId <= numUsers; userId += 1) {
        if (userId === activeUserId) continue;
        const similarity = cosineSimilarity(activeVector, ratingMatrix[userId]);
        if (similarity > 0) neighbors.push({ userId, similarity });
    }

    return neighbors
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, NEIGHBOR_COUNT);
}

function predictUserBasedRating(movieId, topNeighbors) {
    let weightedSum = 0;
    let similaritySum = 0;
    let support = 0;
    topNeighbors.forEach(({ userId, similarity }) => {
        const neighborRating = ratingMatrix[userId][movieId];
        if (neighborRating > 0) {
            weightedSum += similarity * neighborRating;
            similaritySum += similarity;
            support += 1;
        }
    });

    if (support < 2) return null;
    return {
        score: (weightedSum + PRIOR_WEIGHT * globalMeanRating) /
            (similaritySum + PRIOR_WEIGHT),
        support
    };
}

function getUserBasedRecommendations(activeUserId, topK = TOP_K) {
    const activeVector = ratingMatrix[activeUserId];
    if (!activeVector) return [];
    const topNeighbors = getTopNeighbors(activeUserId);

    const candidates = [];
    for (let movieId = 1; movieId <= numMovies; movieId += 1) {
        if (activeVector[movieId] > 0) continue;

        const prediction = predictUserBasedRating(movieId, topNeighbors);
        if (prediction) {
            candidates.push({
                title: getMovieTitle(movieId),
                ...prediction
            });
        }
    }

    return candidates
        .sort((a, b) => b.score - a.score || b.support - a.support ||
            a.title.localeCompare(b.title))
        .slice(0, topK);
}

const itemVectorCache = new Map();
const itemSimilarityCache = new Map();

function resetRecommendationCaches() {
    itemVectorCache.clear();
    itemSimilarityCache.clear();
}

function getItemVector(movieId) {
    if (itemVectorCache.has(movieId)) return itemVectorCache.get(movieId);

    const vector = Array(numUsers + 1).fill(0);
    for (let userId = 1; userId <= numUsers; userId += 1) {
        vector[userId] = ratingMatrix[userId][movieId] || 0;
    }

    itemVectorCache.set(movieId, vector);
    return vector;
}

// Assignment point: reuse item similarities across active users.
function getItemSimilarity(firstId, secondId) {
    if (!itemSimilarityCache.has(firstId)) {
        const row = new Float64Array(numMovies + 1);
        row.fill(Number.NaN);
        itemSimilarityCache.set(firstId, row);
    }
    const row = itemSimilarityCache.get(firstId);
    if (!Number.isNaN(row[secondId])) return row[secondId];

    const similarity = cosineSimilarity(getItemVector(firstId), getItemVector(secondId));
    row[secondId] = similarity;
    const reverseRow = itemSimilarityCache.get(secondId);
    if (reverseRow) reverseRow[firstId] = similarity;
    return similarity;
}

// Assignment point: Item-Based CF from movies the active user already rated.
function predictItemBasedRating(activeUserId, candidateId, ratedMovieIds) {
    if (movieRatingCounts[candidateId] < MIN_ITEM_RATINGS) return null;
    const activeVector = ratingMatrix[activeUserId];
    let weightedSum = 0;
    let similaritySum = 0;
    let support = 0;

    ratedMovieIds.forEach(ratedMovieId => {
        const similarity = getItemSimilarity(candidateId, ratedMovieId);
        if (similarity > 0) {
            weightedSum += similarity * activeVector[ratedMovieId];
            similaritySum += similarity;
            support += 1;
        }
    });

    if (support < 2) return null;
    return {
        score: (weightedSum + PRIOR_WEIGHT * globalMeanRating) /
            (similaritySum + PRIOR_WEIGHT),
        support
    };
}

function getItemBasedRecommendations(activeUserId, topK = TOP_K) {
    const activeVector = ratingMatrix[activeUserId];
    if (!activeVector) return [];

    const ratedMovieIds = getRatedMovieIds(activeUserId);
    if (ratedMovieIds.length === 0) return [];

    const candidates = [];
    for (let candidateId = 1; candidateId <= numMovies; candidateId += 1) {
        if (activeVector[candidateId] > 0) continue;

        const prediction = predictItemBasedRating(activeUserId, candidateId, ratedMovieIds);
        if (prediction) {
            candidates.push({
                title: getMovieTitle(candidateId),
                ...prediction
            });
        }
    }

    return candidates
        .sort((a, b) => b.score - a.score || b.support - a.support ||
            a.title.localeCompare(b.title))
        .slice(0, topK);
}

function getRecommendations() {
    const selectElement = document.getElementById('user-select');
    const userId = Number.parseInt(selectElement.value, 10);

    if (Number.isNaN(userId)) {
        setStatus('Select a user first.', 'error');
        clearRecommendationPanels('Please select a user first.');
        return;
    }

    setStatus(`Calculating collaborative filtering recommendations for User ${userId}...`, 'loading');

    const userBased = getUserBasedRecommendations(userId);
    const itemBased = getItemBasedRecommendations(userId);

    renderList(
        'user-based-result',
        userBased,
        null,
        'Because this user is similar to other users, we recommend:'
    );
    renderList(
        'item-based-result',
        itemBased,
        null,
        'Because this user liked similar movies, we recommend:'
    );
    renderAnalysis(userId, userBased, itemBased);
    setStatus(`Recommendations generated for User ${userId}.`, 'success');
}

function clearRecommendationPanels(message) {
    renderList('user-based-result', [], message);
    renderList('item-based-result', [], message);
}

function renderList(elementId, items, message, intro = '') {
    const element = document.getElementById(elementId);
    element.replaceChildren();

    if (message) {
        const paragraph = document.createElement('p');
        paragraph.textContent = message;
        element.appendChild(paragraph);
        return;
    }

    if (!items || items.length === 0) {
        const paragraph = document.createElement('p');
        paragraph.textContent = 'No recommendations could be generated for this user. Try a user with more ratings.';
        element.appendChild(paragraph);
        return;
    }

    if (intro) {
        const paragraph = document.createElement('p');
        paragraph.textContent = intro;
        element.appendChild(paragraph);
    }

    const list = document.createElement('ol');
    list.className = 'recommendations';
    items.forEach(item => {
        const entry = document.createElement('li');
        const title = document.createElement('span');
        const evidence = document.createElement('div');
        const score = document.createElement('strong');
        const support = document.createElement('small');
        title.textContent = item.title;
        score.textContent = item.score.toFixed(3);
        support.textContent = `${item.support} supporting matches`;
        evidence.className = 'score-evidence';
        evidence.append(score, support);
        entry.append(title, evidence);
        list.appendChild(entry);
    });
    element.appendChild(list);
}

// Assignment point: expose the requested business and algorithmic analysis.

function renderAnalysis(userId, userBased, itemBased) {
    const ratedCount = getRatedMovieIds(userId).length;
    const commonTitles = userBased.filter(item =>
        itemBased.some(other => other.title === item.title)).length;

    document.getElementById('analysis').innerHTML = `
        <p><strong>Two methods:</strong> user-based CF compares User ${userId} with ${numUsers - 1} other users; item-based CF compares unseen movies with this user's ${ratedCount} rated movies. Their Top-5 lists share ${commonTitles} title(s). Predicted scores alone do not measure accuracy; the held-out evaluation is documented in the README.</p>
        <p><strong>Efficiency:</strong> for a full pairwise model, user comparisons grow roughly as U² × I and item comparisons as I² × U. Here U=${numUsers} and I=${numMovies}, so user-based has fewer possible pairs. If users outnumber items, item-based can be cheaper; this implementation caches item similarities for reuse.</p>
        <p><strong>Missing ratings:</strong> the selected strategy weights cosine similarity by the number of common ratings, requires at least ${MIN_COMMON_RATINGS} common ratings, and shrinks predictions toward the training mean. It does not turn missing ratings into dislikes. Mean imputation is simple but can bias scores; matrix factorization can learn latent tastes but costs more to train and explain.</p>
        <p><strong>Cold start:</strong> a new user or movie without ratings has no collaborative evidence. Item candidates need at least ${MIN_ITEM_RATINGS} community ratings, and displayed results show how many neighbors or rated movies support them.</p>
    `;
}
