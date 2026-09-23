const TOP_K = 5;

window.onload = async function initialize() {
    setStatus('Loading MovieLens data...', 'loading');
    try {
        await loadData();
        populateMovieSelects();
        setStatus(`Loaded ${movies.length.toLocaleString()} movies and ${ratings.length.toLocaleString()} ratings.`, 'success');
    } catch (error) {
        console.error('Initialization error:', error);
    }
};

function setStatus(message, className = '') {
    const result = document.getElementById('result');
    result.textContent = message;
    result.className = className;
}

function populateMovieSelects() {
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    document.querySelectorAll('.movie-select').forEach(select => {
        select.innerHTML = '<option value="">Select a movie</option>';
        sortedMovies.forEach(movie => {
            const option = document.createElement('option');
            option.value = movie.id;
            option.textContent = movie.title;
            select.appendChild(option);
        });
    });
    [1, 2, 3].forEach((id, index) => {
        const select = document.getElementById(`profile-movie-${index + 1}`);
        if (select && movies.some(movie => movie.id === id)) select.value = String(id);
    });
}

function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i += 1) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dot / denominator;
}

function averageVectors(movieList) {
    const result = Array(genreNames.length).fill(0);
    movieList.forEach(movie => movie.vector.forEach((value, index) => {
        result[index] += value / movieList.length;
    }));
    return result;
}

function scoreCandidates(queryVector, excludedIds) {
    return movies
        .filter(movie => !excludedIds.has(movie.id))
        .map(movie => ({ movie, score: cosineSimilarity(queryVector, movie.vector) }))
        .sort((a, b) => b.score - a.score || a.movie.title.localeCompare(b.movie.title))
        .slice(0, TOP_K);
}

function getSelectedMovie(id) {
    return movies.find(movie => movie.id === Number.parseInt(id, 10));
}

function getRecommendations() {
    const activeMovie = getSelectedMovie(document.getElementById('active-movie').value);
    const profileMovies = [1, 2, 3]
        .map(index => getSelectedMovie(document.getElementById(`profile-movie-${index}`).value))
        .filter(Boolean);

    if (!activeMovie) {
        setStatus('Select an active movie first.', 'error');
        return;
    }
    if (profileMovies.length === 0) {
        setStatus('Select at least one movie for the user profile.', 'error');
        return;
    }

    setStatus('Calculating cosine similarities...', 'loading');
    const itemRecommendations = scoreCandidates(activeMovie.vector, new Set([activeMovie.id]));
    const profileIds = new Set(profileMovies.map(movie => movie.id));
    const profileRecommendations = scoreCandidates(averageVectors(profileMovies), profileIds);

    renderRecommendations('item-recommendations', itemRecommendations);
    renderRecommendations('profile-recommendations', profileRecommendations);
    renderAnalysis(itemRecommendations, profileRecommendations, profileMovies);
    setStatus('Recommendations generated with cosine similarity.', 'success');
}

function renderRecommendations(elementId, recommendations) {
    const element = document.getElementById(elementId);
    element.innerHTML = recommendations.map(({ movie, score }) => `
        <li><span>${movie.title}</span><strong>${score.toFixed(3)}</strong></li>
    `).join('');
}

function getPopularityMap() {
    const popularity = new Map();
    ratings.forEach(rating => popularity.set(rating.itemId, (popularity.get(rating.itemId) || 0) + 1));
    return popularity;
}

function median(values) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

function renderAnalysis(itemRecommendations, profileRecommendations, profileMovies) {
    const popularity = getPopularityMap();
    const catalogMedian = median(movies.map(movie => popularity.get(movie.id) || 0));
    const summarize = recommendations => {
        const longTail = recommendations.filter(({ movie }) => (popularity.get(movie.id) || 0) < catalogMedian).length;
        const genres = new Set(recommendations.flatMap(({ movie }) => movie.genres));
        return `${longTail}/${recommendations.length} long-tail items; ${genres.size} genres covered`;
    };

    document.getElementById('analysis').innerHTML = `
        <p><strong>Profile:</strong> ${profileMovies.map(movie => movie.title).join('; ')}</p>
        <p><strong>Cosine normalization:</strong> vectors are compared by angle, so a film with many genres does not automatically win because it has a larger vector magnitude.</p>
        <p><strong>Catalog discovery:</strong> item-to-item: ${summarize(itemRecommendations)}. Profile-based: ${summarize(profileRecommendations)}. “Long-tail” means below the catalog median number of ratings (${catalogMedian}).</p>
    `;
}
