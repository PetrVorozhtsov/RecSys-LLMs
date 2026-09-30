// Assignment point: both recommendation strategies must return a Top-5 list.

const TOP_K = 5;
const NEIGHBOR_COUNT = 20;

// Missing-value strategy: use co-rated entries only.
// In ratingMatrix, 0 means "not rated"; cosineSimilarity ignores positions
// where either side is 0, so missing ratings are not treated as negative taste.

window.onload = async function initialize() {
    setStatus('Loading MovieLens data...', 'loading');
    try {
        await loadData();
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

// Assignment point: cosine similarity between two sparse rating vectors.

function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let index = 0; index < a.length; index += 1) {
        if (a[index] === 0 || b[index] === 0) continue;
        dot += a[index] * b[index];
        normA += a[index] * a[index];
        normB += b[index] * b[index];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dot / denominator;
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

// Assignment point: User-Based CF with a similarity-weighted neighbor average.

function getUserBasedRecommendations(activeUserId, topK = TOP_K) {
    const activeVector = ratingMatrix[activeUserId];
    if (!activeVector) return [];

    const neighbors = [];
    for (let userId = 1; userId <= numUsers; userId += 1) {
        if (userId === activeUserId) continue;
        const similarity = cosineSimilarity(activeVector, ratingMatrix[userId]);
        if (similarity > 0) neighbors.push({ userId, similarity });
    }

    const topNeighbors = neighbors
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, NEIGHBOR_COUNT);

    const candidates = [];
    for (let movieId = 1; movieId <= numMovies; movieId += 1) {
        if (activeVector[movieId] > 0) continue;

        let weightedSum = 0;
        let similaritySum = 0;
        topNeighbors.forEach(({ userId, similarity }) => {
            const neighborRating = ratingMatrix[userId][movieId];
            if (neighborRating > 0) {
                weightedSum += similarity * neighborRating;
                similaritySum += similarity;
            }
        });

        if (similaritySum > 0) {
            candidates.push({
                title: getMovieTitle(movieId),
                score: weightedSum / similaritySum
            });
        }
    }

    return candidates
        .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title))
        .slice(0, topK);
}

const itemVectorCache = new Map();

function getItemVector(movieId) {
    if (itemVectorCache.has(movieId)) return itemVectorCache.get(movieId);

    const vector = Array(numUsers + 1).fill(0);
    for (let userId = 1; userId <= numUsers; userId += 1) {
        vector[userId] = ratingMatrix[userId][movieId] || 0;
    }

    itemVectorCache.set(movieId, vector);
    return vector;
}

// Assignment point: Item-Based CF from movies the active user already rated.

function getItemBasedRecommendations(activeUserId, topK = TOP_K) {
    const activeVector = ratingMatrix[activeUserId];
    if (!activeVector) return [];

    const ratedMovieIds = getRatedMovieIds(activeUserId);
    if (ratedMovieIds.length === 0) return [];

    const candidates = [];
    for (let candidateId = 1; candidateId <= numMovies; candidateId += 1) {
        if (activeVector[candidateId] > 0) continue;

        const candidateVector = getItemVector(candidateId);
        let weightedSum = 0;
        let similaritySum = 0;

        ratedMovieIds.forEach(ratedMovieId => {
            const similarity = cosineSimilarity(candidateVector, getItemVector(ratedMovieId));
            if (similarity > 0) {
                weightedSum += similarity * activeVector[ratedMovieId];
                similaritySum += similarity;
            }
        });

        if (similaritySum > 0) {
            candidates.push({
                title: getMovieTitle(candidateId),
                score: weightedSum / similaritySum
            });
        }
    }

    return candidates
        .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title))
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
        const score = document.createElement('strong');
        title.textContent = item.title;
        score.textContent = item.score.toFixed(3);
        entry.append(title, score);
        list.appendChild(entry);
    });
    element.appendChild(list);
}

// Assignment point: expose the requested business and algorithmic analysis.

function renderAnalysis(userId, userBased, itemBased) {
    const ratedCount = getRatedMovieIds(userId).length;
    const avgScore = recommendations => {
        if (!recommendations.length) return 'n/a';
        const total = recommendations.reduce((sum, item) => sum + item.score, 0);
        return (total / recommendations.length).toFixed(3);
    };

    document.getElementById('analysis').innerHTML = `
        <p><strong>User-based vs. item-based:</strong> user-based CF compares the active user with ${numUsers - 1} other users and then uses the Top-${NEIGHBOR_COUNT} neighbors. Item-based CF compares candidate movie columns against the ${ratedCount} movies this user already rated. With ${numUsers} users and ${numMovies} movies, item similarities are often better for reuse and caching, while user similarities adapt directly to the active user's neighborhood.</p>
        <p><strong>Missing-value strategy:</strong> this implementation uses co-rated entries only. It is simple and avoids treating unknown ratings as dislikes. Mean imputation would make vectors denser but can bias scores toward averages; matrix factorization can model latent taste patterns better, but costs more to train and explain.</p>
        <p><strong>Cold start and sparsity:</strong> collaborative filtering needs rating history. A new user, a new movie, or a pair with very few co-rated items produces unreliable similarity because there is too little evidence.</p>
        <p><strong>Observed scores:</strong> User-Based average Top-5 score: ${avgScore(userBased)}. Item-Based average Top-5 score: ${avgScore(itemBased)}.</p>
    `;
}
