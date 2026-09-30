// Assignment: load MovieLens ratings for collaborative filtering.
// The same u.item/u.data files are reused from Week 2; only the algorithm changes.

let movies = [];
let ratings = [];

// Collaborative filtering structures populated by buildRatingMatrix().
let numUsers = 0;          // highest user id found in u.data
let numMovies = 0;         // highest movie id found in u.item
let ratingMatrix = null;   // (numUsers + 1) x (numMovies + 1); 0 = "not rated"

// Genre names as defined in the u.item file, excluding the MovieLens "unknown" flag.
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

async function loadData() {
    try {
        movies = [];
        ratings = [];

        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        parseItemData(await moviesResponse.text());

        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        parseRatingData(await ratingsResponse.text());

        numUsers = ratings.reduce((max, rating) => Math.max(max, rating.userId), 0);
        numMovies = movies.reduce((max, movie) => Math.max(max, movie.id), 0);
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        const errorTarget = document.getElementById('user-based-result');
        if (errorTarget) {
            errorTarget.innerHTML = `<p class="error">Error: ${error.message}. Please make sure u.item and u.data are in the correct location.</p>`;
        }
        throw error;
    }
}

function parseItemData(text) {
    for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue;

        const fields = line.split('|');
        if (fields.length < 24) continue;

        const id = Number.parseInt(fields[0], 10);
        const title = fields[1].trim();

        // Fields 6-23 are the 18 named genres; field 5 is the "unknown" flag.
        const genreValues = fields.slice(6, 24).map(value => Number.parseInt(value, 10) || 0);
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        if (Number.isInteger(id) && title) {
            movies.push({ id, title, genres });
        }
    }
}

function parseRatingData(text) {
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

// Assignment point: build a sparse-friendly user-item matrix.
// MovieLens ratings are 1-5, so 0 is an unambiguous "not rated" marker.

function buildRatingMatrix() {
    ratingMatrix = Array.from({ length: numUsers + 1 }, () => Array(numMovies + 1).fill(0));

    ratings.forEach(({ userId, itemId, rating }) => {
        if (userId <= numUsers && itemId <= numMovies) {
            ratingMatrix[userId][itemId] = rating;
        }
    });
}
