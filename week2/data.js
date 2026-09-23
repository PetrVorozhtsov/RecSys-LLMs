// Assignment: represent MovieLens content as feature vectors for content-based filtering.
// The parsed data is shared by the cosine similarity and profile-based recommenders.
let movies = [];
let ratings = [];

// Assignment point: create content features from metadata.
// MovieLens has an extra `unknown` flag; the task requires the 18 named genres from Action to Western.

const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Assignment point: load the provided MovieLens files before running either recommendation strategy.

async function loadData() {
    try {
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) throw new Error(`Failed to load u.item (${moviesResponse.status})`);
        parseItemData(await moviesResponse.text());

        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) throw new Error(`Failed to load u.data (${ratingsResponse.status})`);
        parseRatingData(await ratingsResponse.text());

        if (movies.length === 0) throw new Error('No movies were found in u.item.');
    } catch (error) {
        console.error('Error loading MovieLens data:', error);
        const result = document.getElementById('result');
        if (result) {
            result.textContent = `Could not load MovieLens data: ${error.message}`;
            result.className = 'error';
        }
        throw error;
    }
}

// Assignment point: parse each movie into an ID, title, genre labels, and a binary feature vector.

function parseItemData(text) {
    movies = [];
    for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue;
        const fields = line.split('|');
        if (fields.length < 24) continue;
        const id = Number.parseInt(fields[0], 10);
        const title = fields[1].trim();
        // Exclude the MovieLens `unknown` flag so the vector contains exactly the 18 task features.

        const vector = fields.slice(6, 24).map(value => Number.parseInt(value, 10) || 0);
        const genres = genreNames.filter((_, index) => vector[index] === 1);
        if (Number.isInteger(id) && title) movies.push({ id, title, genres, vector });
    }
}

// Assignment point: parse ratings so the app can measure catalog popularity for the long-tail analysis.

function parseRatingData(text) {
    ratings = [];
    for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue;
        const fields = line.split('\t');
        if (fields.length < 4) continue;
        const userId = Number.parseInt(fields[0], 10);
        const itemId = Number.parseInt(fields[1], 10);
        const rating = Number.parseFloat(fields[2]);
        const timestamp = Number.parseInt(fields[3], 10);
        if ([userId, itemId, rating, timestamp].every(Number.isFinite)) {
            ratings.push({ userId, itemId, rating, timestamp });
        }
    }
}
