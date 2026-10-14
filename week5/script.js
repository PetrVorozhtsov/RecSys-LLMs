/**
 * HW5: 16-factor TensorFlow.js matrix factorization on observed ratings.
 * The lecture defines r_hat(u, i) = p_u dot q_i, without biases or clipping.
 * Training uses mean squared error plus mean batch-local squared factor norms.
 * The supplied self-checks below remain unchanged, including their three
 * assertions that conflict with the lecture's scoring/candidate contracts.
 */

const NUM_FACTORS = 16;
const RANDOM_SEED = 202605;
let trainSet = [];
let valSet = [];
let testSet = [];
let model = null;
let trainingHistory = [];
let bestCheckpoint = null;
let tfBackend = null;
let selfCheckResult = null;
let trainingInProgress = false;

async function initTf() {
  const statusEl = document.getElementById("tf-status");
  if (typeof tf === "undefined") {
    const message = "TensorFlow.js failed to load. Check the CDN connection and reload.";
    if (statusEl) { statusEl.textContent = message; statusEl.className = "error"; }
    return null;
  }
  try {
    if (!(await tf.setBackend("webgl"))) await tf.setBackend("cpu");
  } catch (error) { await tf.setBackend("cpu"); }
  await tf.ready();
  tfBackend = tf.getBackend();
  if (statusEl) statusEl.textContent = `TensorFlow.js backend: ${tfBackend}`;
  return tfBackend;
}

function isTfModel(candidate) {
  return !!candidate && !!candidate.userFactors &&
    typeof candidate.userFactors.dataSync === "function" &&
    !!candidate.itemFactors && typeof candidate.itemFactors.dataSync === "function";
}

function fmt(x, d = 4) {
  if (x === null || x === undefined || !Number.isFinite(Number(x))) return "\u2014";
  return Number(x).toFixed(d);
}

function movieTitle(id) {
  const movie = movies.find((m) => m.id === Number(id));
  return movie ? movie.title : `Movie #${id}`;
}

function escapeHtml(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function clamp(value, lo, hi) { return Math.min(hi, Math.max(lo, value)); }

/** Train-only entity vocabulary. Fixture models obtain it from trainSet. */
function trainingEntities() {
  if (model && model.trainUserIds && model.trainMovieIds) {
    return { users: model.trainUserIds, items: model.trainMovieIds };
  }
  return {
    users: new Set(trainSet.map((r) => r.userId)),
    items: new Set(trainSet.map((r) => r.movieId)),
  };
}

/** Exclude cold-start rows from RMSE; count every excluded rating once. */
function eligibleSplit(split) {
  const { users, items } = trainingEntities();
  const eligible = [], unknownUsers = new Set(), unknownMovies = new Set();
  for (const r of split || []) {
    const userKnown = users.has(r.userId), movieKnown = items.has(r.movieId);
    if (!userKnown) unknownUsers.add(r.userId);
    if (!movieKnown) unknownMovies.add(r.movieId);
    if (userKnown && movieKnown) eligible.push(r);
  }
  return {
    rows: eligible,
    total: (split || []).length,
    evaluated: eligible.length,
    excluded: (split || []).length - eligible.length,
    unknownUserIds: [...unknownUsers].sort((a, b) => a - b),
    unknownMovieIds: [...unknownMovies].sort((a, b) => a - b),
  };
}

function splitAudit(split) {
  const { rows, ...audit } = eligibleSplit(split);
  return audit;
}

/** Batched pure-dot RMSE; no global mean addition and no clipping. */
function computeRMSEFor(split) {
  const rows = eligibleSplit(split).rows;
  if (!rows.length) return 0;
  if (isTfModel(model)) {
    return tf.tidy(() => {
      const users = tf.tensor1d(rows.map((r) => r.userId), "int32");
      const items = tf.tensor1d(rows.map((r) => r.movieId), "int32");
      const actual = tf.tensor1d(rows.map((r) => r.rating));
      const prediction = tf.gather(model.userFactors, users)
        .mul(tf.gather(model.itemFactors, items)).sum(1);
      return Math.sqrt(prediction.sub(actual).square().mean().dataSync()[0]);
    });
  }
  let squaredError = 0;
  for (const r of rows) squaredError += (predictRating(r.userId, r.movieId) - r.rating) ** 2;
  return Math.sqrt(squaredError / rows.length);
}

/** Training mean baseline, evaluated on the same eligible rows as the model. */
function computeBaselineRMSE(split) {
  const rows = eligibleSplit(split).rows;
  if (!rows.length) return 0;
  const mean = model ? model.globalMean :
    (trainSet.length ? trainSet.reduce((sum, r) => sum + r.rating, 0) / trainSet.length : 0);
  return Math.sqrt(rows.reduce((sum, r) => sum + (mean - r.rating) ** 2, 0) / rows.length);
}

/** Pure latent dot product; unknown training IDs use the train mean. */
function predictRating(userId, movieId) {
  if (!isTfModel(model)) return 0;
  const { users, items } = trainingEntities();
  if (!Number.isInteger(userId) || !Number.isInteger(movieId) ||
      !users.has(userId) || !items.has(movieId) || userId < 0 || movieId < 0 ||
      userId >= model.userFactors.shape[0] || movieId >= model.itemFactors.shape[0]) {
    return model.globalMean;
  }
  return tf.tidy(() => {
    const width = model.userFactors.shape[1];
    const user = model.userFactors.slice([userId, 0], [1, width]);
    const item = model.itemFactors.slice([movieId, 0], [1, width]);
    return user.mul(item).sum().dataSync()[0];
  });
}

function exclusionMap(train, val) {
  const excluded = new Map();
  for (const r of [...train, ...val]) {
    if (!excluded.has(r.userId)) excluded.set(r.userId, new Set());
    excluded.get(r.userId).add(r.movieId);
  }
  return excluded;
}

/** Seeded normal initialization; dispose the previous model on repeated Train. */
function initializeFactors(train, numFactors = NUM_FACTORS) {
  if (!Number.isInteger(numFactors) || numFactors < 1) throw new Error("Invalid factor count.");
  if (model) tf.dispose([model.userFactors, model.itemFactors]);
  const maxUserId = train.reduce((max, r) => Math.max(max, r.userId), Number(numUsers) || 0);
  const maxMovieId = train.reduce((max, r) => Math.max(max, r.movieId),
    movies.reduce((max, movie) => Math.max(max, movie.id), 0));
  const variables = tf.tidy(() => ({
    userFactors: tf.variable(tf.randomNormal([maxUserId + 1, numFactors], 0, 0.05, "float32", RANDOM_SEED)),
    itemFactors: tf.variable(tf.randomNormal([maxMovieId + 1, numFactors], 0, 0.05, "float32", RANDOM_SEED + 1)),
  }));
  model = {
    ...variables,
    globalMean: train.length ? train.reduce((sum, r) => sum + r.rating, 0) / train.length : 0,
    trainUserIds: new Set(train.map((r) => r.userId)),
    trainMovieIds: new Set(train.map((r) => r.movieId)),
    excludedByUser: exclusionMap(train, valSet),
    seed: RANDOM_SEED,
    factorCount: numFactors,
    backend: tf.getBackend(),
    initializedAt: new Date().toISOString(),
    selectedEpoch: null,
  };
  trainingHistory = [];
  bestCheckpoint = null;
  return model;
}

/** Mulberry32 drives reproducible Fisher-Yates batch order. */
function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function normalizedOptions(opts = {}) {
  const options = {
    epochs: opts.epochs === undefined ? 20 : Number(opts.epochs),
    lr: opts.lr === undefined ? 0.01 : Number(opts.lr),
    reg: opts.reg === undefined ? 0.05 : Number(opts.reg),
    batchSize: opts.batchSize === undefined ? 64 : Number(opts.batchSize),
    seed: opts.seed === undefined ? RANDOM_SEED : Number(opts.seed),
  };
  if (!Number.isInteger(options.epochs) || options.epochs < 1 ||
      !Number.isFinite(options.lr) || options.lr <= 0 ||
      !Number.isFinite(options.reg) || options.reg < 0 ||
      !Number.isInteger(options.batchSize) || options.batchSize < 1 ||
      !Number.isInteger(options.seed)) throw new Error("Invalid training hyperparameters.");
  return options;
}

function startTraining(train, val, opts) {
  if (!isTfModel(model)) throw new Error("Initialize factors before training.");
  const options = normalizedOptions(opts);
  model.options = options;
  model.excludedByUser = exclusionMap(train, val);
  trainingHistory = [];
  bestCheckpoint = null;
  const order = Array.from({ length: train.length }, (_, i) => i);
  return {
    train, val, options, order,
    random: seededRandom(options.seed),
    optimizer: tf.train.adam(options.lr),
    bestWeights: null,
    started: performance.now(),
  };
}

function shuffleTrainingOrder(state) {
  for (let i = state.order.length - 1; i > 0; i--) {
    const j = Math.floor(state.random() * (i + 1));
    [state.order[i], state.order[j]] = [state.order[j], state.order[i]];
  }
}

/** One observed-rating batch: MSE + lambda * mean(||p||² + ||q||²). */
function trainBatch(state, offset) {
  const batch = state.order.slice(offset, offset + state.options.batchSize)
    .map((index) => state.train[index]);
  tf.tidy(() => {
    const users = tf.tensor1d(batch.map((r) => r.userId), "int32");
    const items = tf.tensor1d(batch.map((r) => r.movieId), "int32");
    const actual = tf.tensor1d(batch.map((r) => r.rating));
    state.optimizer.minimize(() => {
      const user = tf.gather(model.userFactors, users);
      const item = tf.gather(model.itemFactors, items);
      const predicted = user.mul(item).sum(1);
      const mse = predicted.sub(actual).square().mean();
      const l2 = user.square().sum(1).add(item.square().sum(1)).mean();
      return mse.add(l2.mul(state.options.reg));
    }, false, [model.userFactors, model.itemFactors]);
  });
}

function finishEpoch(state, epoch, started) {
  const row = {
    epoch,
    trainRMSE: computeRMSEFor(state.train),
    valRMSE: computeRMSEFor(state.val),
    durationMs: performance.now() - started,
  };
  if (!Number.isFinite(row.trainRMSE) || !Number.isFinite(row.valRMSE)) {
    throw new Error("Training diverged; reduce the learning rate.");
  }
  trainingHistory.push(row);
  if (!bestCheckpoint || row.valRMSE < bestCheckpoint.valRMSE) {
    bestCheckpoint = row;
    // CPU snapshots retain only the current best epoch, not every epoch's tensors.
    state.bestWeights = {
      user: new Float32Array(model.userFactors.dataSync()),
      item: new Float32Array(model.itemFactors.dataSync()),
    };
  }
}

function finishTraining(state) {
  if (state.bestWeights) {
    tf.tidy(() => {
      model.userFactors.assign(tf.tensor2d(state.bestWeights.user, model.userFactors.shape));
      model.itemFactors.assign(tf.tensor2d(state.bestWeights.item, model.itemFactors.shape));
    });
  }
  model.selectedEpoch = bestCheckpoint ? bestCheckpoint.epoch : null;
  model.durationMs = performance.now() - state.started;
  model.trainedAt = new Date().toISOString();
  model.metricExclusions = { validation: splitAudit(state.val), test: splitAudit(testSet) };
  return trainingHistory;
}

/** Synchronous assignment API, also used by the original self-check harness. */
function trainMF(train, val, opts) {
  if (!train.length) { trainingHistory = []; return trainingHistory; }
  const state = startTraining(train, val, opts);
  try {
    for (let epoch = 1; epoch <= state.options.epochs; epoch++) {
      const started = performance.now();
      shuffleTrainingOrder(state);
      for (let offset = 0; offset < train.length; offset += state.options.batchSize) trainBatch(state, offset);
      finishEpoch(state, epoch, started);
    }
    return finishTraining(state);
  } finally { state.optimizer.dispose(); }
}

/** The UI uses the same batches but yields so rendering and controls stay live. */
async function trainMFAsync(train, val, opts, onEpoch) {
  if (!train.length) { trainingHistory = []; return trainingHistory; }
  const state = startTraining(train, val, opts);
  try {
    for (let epoch = 1; epoch <= state.options.epochs; epoch++) {
      const started = performance.now();
      shuffleTrainingOrder(state);
      let lastYield = performance.now();
      for (let offset = 0; offset < train.length; offset += state.options.batchSize) {
        trainBatch(state, offset);
        if (performance.now() - lastYield >= 50) {
          await tf.nextFrame();
          lastYield = performance.now();
        }
      }
      finishEpoch(state, epoch, started);
      if (onEpoch) onEpoch(trainingHistory[trainingHistory.length - 1]);
      await tf.nextFrame();
    }
    return finishTraining(state);
  } finally { state.optimizer.dispose(); }
}

/** Smallest finite validation RMSE; ties select the earliest epoch. */
function selectBestCheckpoint(history) {
  let best = null;
  for (const row of history || []) {
    if (!Number.isFinite(row.valRMSE)) continue;
    if (!best || row.valRMSE < best.valRMSE ||
        (row.valRMSE === best.valRMSE && row.epoch < best.epoch)) best = row;
  }
  return best;
}

/** Train-seen candidate items only; exclude this user's train + validation. */
function recommendTopK(userId, K = 10) {
  if (!isTfModel(model) || !Number.isFinite(K) || K <= 0) return [];
  const { users, items } = trainingEntities();
  if (!users.has(userId) || !Number.isInteger(userId) || userId >= model.userFactors.shape[0]) return [];
  const excluded = (model.excludedByUser || exclusionMap(trainSet, valSet)).get(userId) || new Set();
  const candidateIds = movies.map((m) => m.id).filter((id) =>
    items.has(id) && !excluded.has(id) && id >= 0 && id < model.itemFactors.shape[0])
    .sort((a, b) => a - b);
  if (!candidateIds.length) return [];
  return tf.tidy(() => {
    const user = model.userFactors.slice([userId, 0], [1, model.userFactors.shape[1]]);
    const item = tf.gather(model.itemFactors, tf.tensor1d(candidateIds, "int32"));
    const scores = item.matMul(user, false, true).reshape([candidateIds.length]);
    const top = tf.topk(scores, Math.min(Math.floor(K), candidateIds.length), true);
    const indices = top.indices.dataSync(), values = top.values.dataSync();
    return Array.from(indices, (index, rank) => ({
      movieId: candidateIds[index], title: movieTitle(candidateIds[index]), score: values[rank],
    }));
  });
}

/** Macro Recall@K, counting distinct relevant (rating >= 4) test movies. */
function recallAtK(K = 10) {
  const { users, items } = trainingEntities();
  const relevantByUser = new Map(), testUsers = new Set(), unseen = new Set(), unknownUsers = new Set();
  const exclusions = model && model.excludedByUser || exclusionMap(trainSet, valSet);
  const metadataIds = new Set(movies.map((m) => m.id));
  const unavailable = new Set(), previouslyRated = new Set();
  for (const r of testSet) {
    testUsers.add(r.userId);
    if (!users.has(r.userId)) unknownUsers.add(r.userId);
    if (r.rating < 4) continue;
    if (!items.has(r.movieId)) { unseen.add(r.movieId); continue; }
    if (!metadataIds.has(r.movieId)) { unavailable.add(r.movieId); continue; }
    if (!users.has(r.userId)) continue;
    if (exclusions.get(r.userId)?.has(r.movieId)) { previouslyRated.add(r.movieId); continue; }
    if (!relevantByUser.has(r.userId)) relevantByUser.set(r.userId, new Set());
    relevantByUser.get(r.userId).add(r.movieId);
  }
  const perUser = [];
  for (const userId of [...relevantByUser.keys()].sort((a, b) => a - b)) {
    const relevant = relevantByUser.get(userId);
    const recommended = new Set(recommendTopK(userId, K).map((r) => r.movieId));
    let hit = 0;
    for (const movieId of relevant) if (recommended.has(movieId)) hit++;
    perUser.push({ userId, relevant: relevant.size, hit });
  }
  return {
    overallRecall: perUser.length ? perUser.reduce((sum, r) => sum + r.hit / r.relevant, 0) / perUser.length : 0,
    perUser,
    excluded: {
      unseenIds: [...unseen].sort((a, b) => a - b),
      usersWithoutRelevant: [...testUsers].filter((id) => !relevantByUser.has(id)).sort((a, b) => a - b),
      unknownUserIds: [...unknownUsers].sort((a, b) => a - b),
      unavailableMovieIds: [...unavailable].sort((a, b) => a - b),
      previouslyRatedMovieIds: [...previouslyRated].sort((a, b) => a - b),
    },
  };
}

function renderDatasetSummary(container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;
  const uniqueUsers = new Set(ratings.map((r) => r.userId)).size;
  const uniqueMovies = new Set(movies.map((m) => m.id)).size;
  const observedPairs = new Set(ratings.map((r) => `${r.userId}:${r.movieId}`)).size;
  const cells = uniqueUsers * uniqueMovies;
  const sparsity = cells ? 100 * (1 - observedPairs / cells) : 0;
  target.innerHTML = `<dl class="summary-list">
    <dt>Movies</dt><dd>${uniqueMovies.toLocaleString("en-US")}</dd>
    <dt>Ratings</dt><dd>${ratings.length.toLocaleString("en-US")}</dd>
    <dt>Users</dt><dd>${uniqueUsers.toLocaleString("en-US")}</dd>
    <dt>Rating matrix</dt><dd>${uniqueUsers} &times; ${uniqueMovies} (observed IDs)</dd>
    <dt>Sparsity</dt><dd>${fmt(sparsity, 2)}% missing entries</dd>
    <dt>Prediction</dt><dd>p<sub>u</sub> &middot; q<sub>i</sub>; ${NUM_FACTORS} factors</dd>
  </dl>`;
}

function renderSplitSummary(container) {
  const target = container || document.getElementById("split-summary-body");
  if (!target) return;
  const total = trainSet.length + valSet.length + testSet.length;
  const pct = (count) => total ? (100 * count / total).toFixed(1) : "0.0";
  const exclusions = typeof SPLIT_EXCLUSIONS === "undefined" ? [] : SPLIT_EXCLUSIONS;
  const rows = [["Train", trainSet], ["Validation", valSet], ["Test", testSet]].map(([label, split]) => {
    const audit = splitAudit(split);
    return `<tr><td>${label}</td><td class="num">${split.length}</td><td class="num">${pct(split.length)}%</td><td class="num">${audit.evaluated}</td><td class="num">${audit.excluded}</td></tr>`;
  }).join("");
  const trained = model && model.selectedEpoch !== null;
  target.innerHTML = `<table class="data-table">
    <thead><tr><th>Split</th><th>Ratings</th><th>Share</th><th>RMSE eligible</th><th>Cold-start excluded</th></tr></thead><tbody>${rows}</tbody>
    </table><p class="hint">${exclusions.length} user(s) skipped while splitting. Baseline test RMSE ${fmt(computeBaselineRMSE(testSet))}${trained ? `; selected-model test RMSE ${fmt(computeRMSEFor(testSet))}` : ""}. Both metrics use the same eligible test rows.</p>`;
}

function renderTrainingCurve(container) {
  const target = container || document.getElementById("training-curve");
  if (!target) return;
  if (!trainingHistory.length) {
    target.innerHTML = '<p class="empty-state">No training history yet. Press &ldquo;Train&rdquo;.</p>';
    return;
  }
  const best = selectBestCheckpoint(trainingHistory);
  const rows = trainingHistory.map((row) => `<tr${row === best ? ' class="best-epoch"' : ""}>
    <td class="num">${row.epoch}</td><td class="num">${fmt(row.trainRMSE)}</td><td class="num">${fmt(row.valRMSE)}</td><td class="num">${fmt(row.durationMs / 1000, 2)}</td></tr>`).join("");
  target.innerHTML = `<table class="data-table"><thead><tr><th>Epoch</th><th>Train RMSE</th><th>Val RMSE</th><th>Seconds</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderTopK(userId, container) {
  const target = container || document.getElementById("recommendations");
  if (!target) return;
  if (!model || model.selectedEpoch === null) {
    target.innerHTML = '<p class="empty-state">Train the model before requesting recommendations.</p>';
    return;
  }
  const recs = recommendTopK(Number(userId), 10);
  target.innerHTML = recs.length ? '<ol class="recommendation-list">' + recs.map((r) =>
    `<li><span class="rec-title">${escapeHtml(r.title)}</span> <span class="rec-score">${fmt(r.score)}</span></li>`).join("") + "</ol>" :
    '<p class="empty-state">No eligible recommendations for this user.</p>';
}

function renderRecall(K = 10, container) {
  const target = container || document.getElementById("recall-results");
  if (!target) return;
  if (!model || model.selectedEpoch === null) {
    target.innerHTML = '<p class="empty-state">Train the model before evaluating Recall.</p>';
    return;
  }
  const result = recallAtK(K);
  const rows = result.perUser.map((u) => `<tr><td>${u.userId}</td><td class="num">${u.relevant}</td><td class="num">${u.hit}</td><td class="num">${fmt(u.hit / u.relevant)}</td></tr>`).join("");
  target.innerHTML = `<p><strong>Recall@${K}:</strong> <span class="num">${fmt(result.overallRecall)}</span> (${result.perUser.length} users)</p>
    <table class="data-table"><thead><tr><th>User</th><th>Relevant</th><th>Hits</th><th>Recall</th></tr></thead><tbody>${rows || '<tr><td colspan="4">No evaluable users.</td></tr>'}</tbody></table>
    <p class="hint">Excluded: ${result.excluded.unseenIds.length} unseen relevant test item(s); ${result.excluded.usersWithoutRelevant.length} user(s) without an eligible relevant candidate; ${result.excluded.unknownUserIds.length} unknown training user(s).</p>`;
}

function readHyperparameters() {
  const read = (id, fallback) => {
    const el = document.getElementById(id);
    return !el || el.value.trim() === "" ? fallback : Number(el.value);
  };
  return normalizedOptions({ epochs: read("epochs", 20), lr: read("lr", 0.01),
    reg: read("reg", 0.05), batchSize: read("batch-size", 64), seed: RANDOM_SEED });
}

function populateUserSelect() {
  const select = document.getElementById("user-select");
  if (!select) return;
  const ids = [...new Set(trainSet.map((r) => r.userId))].sort((a, b) => a - b);
  select.innerHTML = ids.length ? ids.map((id) => `<option value="${id}">User ${id}</option>`).join("") : '<option value="">No users</option>';
}

/** Execute the unchanged harness safely before fitting the dataset model. */
function runSelfChecksSafely() {
  if (trainingInProgress || model && model.selectedEpoch !== null) return selfCheckResult;
  const existing = new Set(Object.keys(tf.engine().registeredVariables));
  try { selfCheckResult = tf.tidy(() => runTests()); }
  finally {
    for (const [name, variable] of Object.entries(tf.engine().registeredVariables)) {
      if (!existing.has(name)) variable.dispose();
    }
  }
  const log = document.getElementById("testLog");
  if (log && selfCheckResult && selfCheckResult.failed) {
    log.textContent += "\n\nLecture/scaffold conflicts: pure dot products are not clipped; candidates must be seen in train, so the supplied candidate and rank-11 fixtures expect unavailable movies. The original assertions are preserved.";
  }
  return selfCheckResult;
}

/** Dense selected weights and their raw-ID mapping make the run auditable. */
function exportRun() {
  if (!model || model.selectedEpoch === null || trainingInProgress) return null;
  const width = model.userFactors.shape[1];
  const userWeights = Array.from(model.userFactors.dataSync());
  const itemWeights = Array.from(model.itemFactors.dataSync());
  const auditPair = eligibleSplit(testSet).rows[0] || eligibleSplit(trainSet).rows[0];
  const userFactors = auditPair ? userWeights.slice(auditPair.userId * width, (auditPair.userId + 1) * width) : [];
  const itemFactors = auditPair ? itemWeights.slice(auditPair.movieId * width, (auditPair.movieId + 1) * width) : [];
  return {
    formatVersion: 1,
    dataset: "MovieLens 100k",
    split: "per-user timestamp order; first 80%, next 10%, final 10%; train vocabulary only",
    predictionFormula: "sum_f userFactors[userId,f] * itemFactors[movieId,f]",
    lossFormula: "mean((rating - dot)^2) + reg * mean(sum_f(p^2 + q^2))",
    seed: model.seed,
    backend: model.backend,
    options: { ...model.options },
    selectedEpoch: model.selectedEpoch,
    trainMean: model.globalMean,
    initializedAt: model.initializedAt,
    trainedAt: model.trainedAt,
    durationMs: model.durationMs,
    history: trainingHistory.map((row) => ({ ...row })),
    metrics: { selectedTrainRMSE: computeRMSEFor(trainSet), selectedValRMSE: computeRMSEFor(valSet),
      selectedTestRMSE: computeRMSEFor(testSet), baselineTestRMSE: computeBaselineRMSE(testSet),
      exclusions: { validation: splitAudit(valSet), test: splitAudit(testSet) }, recallAt10: recallAtK(10) },
    recommendations: { user1: recommendTopK(1, 10), user2: recommendTopK(2, 10) },
    factorMatrices: {
      indexing: "row number equals raw ID; unused rows are not eligible for scoring",
      factorCount: width,
      userShape: [...model.userFactors.shape], itemShape: [...model.itemFactors.shape],
      trainUserIds: [...model.trainUserIds].sort((a, b) => a - b),
      trainMovieIds: [...model.trainMovieIds].sort((a, b) => a - b),
      userWeights, itemWeights,
    },
    independentPredictionAudit: auditPair ? {
      userId: auditPair.userId, movieId: auditPair.movieId, actual: auditPair.rating,
      userFactors, itemFactors,
      javascriptDot: userFactors.reduce((sum, value, f) => sum + value * itemFactors[f], 0),
      tensorPrediction: predictRating(auditPair.userId, auditPair.movieId),
    } : null,
    suppliedSelfChecks: selfCheckResult,
  };
}

function downloadRun() {
  const run = exportRun();
  if (!run) return;
  const json = JSON.stringify(run, null, 2);
  // A copyable fallback also works in browsers that block file downloads.
  let copyPanel = document.getElementById("copy-run-panel");
  if (!copyPanel) {
    copyPanel = document.createElement("details");
    copyPanel.id = "copy-run-panel";
    const summary = document.createElement("summary");
    summary.textContent = "Copy run JSON if the download is unavailable";
    const text = document.createElement("textarea");
    text.id = "run-json";
    text.readOnly = true;
    text.rows = 8;
    text.style.width = "100%";
    text.setAttribute("aria-label", "Run JSON");
    copyPanel.append(summary, text);
    document.getElementById("download-run").parentElement.after(copyPanel);
  }
  document.getElementById("run-json").value = json;
  const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `hw5-run-seed-${run.seed}-epoch-${run.selectedEpoch}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function runPipeline() {
  const status = document.getElementById("status");
  if (trainingInProgress) return;
  if (!movies.length || !trainSet.length) {
    if (status) status.textContent = "Data not loaded yet — reload the page.";
    return;
  }
  const controls = ["train", "run-tests", "recommend-btn", "recall-btn", "download-run"]
    .map((id) => document.getElementById(id)).filter(Boolean);
  try {
    const opts = readHyperparameters();
    if (!selfCheckResult && !model) runSelfChecksSafely();
    trainingInProgress = true;
    for (const control of controls) control.disabled = true;
    if (status) status.textContent = "Initializing the seeded factor model…";
    await tf.nextFrame();
    initializeFactors(trainSet, NUM_FACTORS);
    await trainMFAsync(trainSet, valSet, opts, (row) => {
      renderTrainingCurve();
      if (status) status.textContent = `Epoch ${row.epoch}/${opts.epochs}; train RMSE ${fmt(row.trainRMSE)}; validation RMSE ${fmt(row.valRMSE)}.`;
    });
    bestCheckpoint = selectBestCheckpoint(trainingHistory);
    renderTrainingCurve();
    renderSplitSummary();
    const select = document.getElementById("user-select");
    renderTopK(select ? Number(select.value) : 1);
    if (status) status.textContent = `Trained ${opts.epochs} epochs on ${model.backend}; restored epoch ${model.selectedEpoch}, validation RMSE ${fmt(bestCheckpoint.valRMSE)}, test RMSE ${fmt(computeRMSEFor(testSet))}. Seed ${model.seed}; ${fmt(model.durationMs / 1000, 1)} seconds.`;
  } catch (error) {
    if (status) status.textContent = `Training failed: ${error.message}`;
  } finally {
    trainingInProgress = false;
    for (const control of controls) control.disabled = false;
    const testButton = document.getElementById("run-tests");
    if (testButton && model && model.selectedEpoch !== null) {
      testButton.disabled = true;
      testButton.title = "The supplied fixture harness ran before training. Reload to rerun it.";
    }
    const downloadButton = document.getElementById("download-run");
    if (downloadButton) downloadButton.disabled = !model || model.selectedEpoch === null;
  }
}

function init() {
  populateUserSelect();
  renderDatasetSummary();
  renderSplitSummary();
  renderTrainingCurve();
  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", runSelfChecksSafely);
  const trainButton = document.getElementById("train");
  if (trainButton) {
    trainButton.addEventListener("click", runPipeline);
    const download = document.createElement("button");
    download.id = "download-run"; download.type = "button"; download.className = "secondary";
    download.textContent = "Download run JSON"; download.disabled = true;
    download.addEventListener("click", downloadRun);
    trainButton.parentElement.appendChild(download);
  }
  const recommendButton = document.getElementById("recommend-btn");
  if (recommendButton) recommendButton.addEventListener("click", () => {
    const select = document.getElementById("user-select");
    renderTopK(select ? Number(select.value) : 1);
  });
  const recallButton = document.getElementById("recall-btn");
  if (recallButton) recallButton.addEventListener("click", () => renderRecall(10));
  runSelfChecksSafely();
  const status = document.getElementById("status");
  if (status) status.textContent = `Dataset ready: ${movies.length.toLocaleString("en-US")} movies, ${trainSet.length.toLocaleString("en-US")} train ratings on ${tfBackend || "tf"}. Press “Train”. Self-checks: ${selfCheckResult.passed} pass, ${selfCheckResult.failed} lecture/scaffold conflicts.`;
}

// ---------------------------------------------------------------------------
// Test harness (supplied and preserved below)
// ---------------------------------------------------------------------------


const TODO_MARKER = "TODO(hw5)";

/** A five-rating toy training set used by several checks. */
const TOY_RATINGS = [
  { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
  { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
  { userId: 1, movieId: 3, rating: 3, timestamp: 3 },
  { userId: 2, movieId: 1, rating: 2, timestamp: 1 },
  { userId: 2, movieId: 2, rating: 5, timestamp: 2 },
];

/** Movies matching `TOY_RATINGS`. */
const TOY_MOVIES = [
  { id: 1, title: "Toy One", genres: [] },
  { id: 2, title: "Toy Two", genres: [] },
  { id: 3, title: "Toy Three", genres: [] },
];

/** A tiny train/val/test fixture for the recommendation checks. */
const REC_FIXTURE = {
  movies: [
    { id: 1, title: "Alpha", genres: [] },
    { id: 2, title: "Bravo", genres: [] },
    { id: 3, title: "Charlie", genres: [] },
    { id: 4, title: "Delta", genres: [] },
    { id: 5, title: "Echo", genres: [] },
  ],
  train: [{ userId: 1, movieId: 1, rating: 5, timestamp: 1 }],
  val: [{ userId: 1, movieId: 2, rating: 4, timestamp: 2 }],
  test: [{ userId: 1, movieId: 3, rating: 5, timestamp: 3 }],
};

/**
 * Build a TF.js factor model from sparse score maps.
 *
 * `userScores[id]` / `itemScores[id]` may be a number (written into factor 0,
 * so the dot product is just the product) or an array (written across factors).
 * Rows are dense and indexed by raw id, exactly like the real model, so the
 * recommendation / recall fixtures exercise the same tensor path as training.
 *
 * @param {Object<number, number|number[]>} userScores
 * @param {Object<number, number|number[]>} itemScores
 * @param {number} [globalMean=0]
 * @returns {Model}
 */
function fixtureModel(userScores, itemScores, globalMean = 0) {
  const F = NUM_FACTORS;
  const userIds = Object.keys(userScores).map(Number);
  const itemIds = Object.keys(itemScores).map(Number);

  // A number goes into factor 0 (so the dot product is the product); an array
  // is copied across the first factors.
  const rowsFor = (value) => {
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return Array.from(value);
    const row = new Array(F).fill(0);
    row[0] = Number(value);
    return row;
  };

  const maxUser = Math.max(numUsers, ...userIds);
  const maxMovieFromMovies = movies.reduce((m, x) => Math.max(m, x.id), 0);
  const maxItem = Math.max(maxMovieFromMovies, ...itemIds);

  const uData = new Float32Array((maxUser + 1) * F);
  for (const id of userIds) {
    const row = rowsFor(userScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) uData[id * F + f] = row[f];
  }
  const iData = new Float32Array((maxItem + 1) * F);
  for (const id of itemIds) {
    const row = rowsFor(itemScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) iData[id * F + f] = row[f];
  }

  return {
    userFactors: tf.variable(tf.tensor2d(uData, [maxUser + 1, F])),
    itemFactors: tf.variable(tf.tensor2d(iData, [maxItem + 1, F])),
    globalMean,
  };
}

/**
 * A recall fixture with 12 movies, one user whose single relevant test movie
 * (movie 2) is placed at a chosen rank by the hand-made model.
 *
 * @param {number} targetRank 1 to put movie 2 first, 11 to put it last.
 * @returns {{movies: Movie[], train: Rating[], val: Rating[], test: Rating[], model: Model}}
 */
function recallFixture(targetRank) {
  const movies = [];
  for (let id = 1; id <= 12; id++) {
    movies.push({ id, title: `Movie ${id}`, genres: [] });
  }
  const train = [
    { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 2, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 3, rating: 4, timestamp: 2 },
  ];
  const val = [];
  const test = [{ userId: 1, movieId: 2, rating: 5, timestamp: 2 }];
  const itemScores = { 1: 3.0 };
  if (targetRank === 1) {
    itemScores[2] = 5.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.0 + 0.01 * id;
  } else {
    itemScores[2] = 1.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.1 + 0.01 * id;
  }
  return { movies, train, val, test, model: fixtureModel({ 1: 1 }, itemScores) };
}

/**
 * Run `fn` with the module-level data globals temporarily replaced by a fixture,
 * then restore them. Keeps the checks independent of the loaded dataset.
 *
 * @param {{movies?: Movie[], train?: Rating[], val?: Rating[], test?: Rating[], model?: Model|null, history?: HistoryRow[]}} fixture
 * @param {() => any} fn
 * @returns {any}
 */
function withFixture(fixture, fn) {
  const saved = { movies, trainSet, valSet, testSet, model, trainingHistory };
  try {
    movies = fixture.movies !== undefined ? fixture.movies : movies;
    trainSet = fixture.train || [];
    valSet = fixture.val || [];
    testSet = fixture.test || [];
    model = fixture.model !== undefined ? fixture.model : null;
    trainingHistory = fixture.history || [];
    return fn();
  } finally {
    movies = saved.movies;
    trainSet = saved.trainSet;
    valSet = saved.valSet;
    testSet = saved.testSet;
    model = saved.model;
    trainingHistory = saved.trainingHistory;
  }
}

/**
 * Run the automated self-checks. Checks that call an unimplemented
 * `TODO(hw5)` stub are reported as PENDING rather than FAIL, so the harness is
 * useful before and after the assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {{passed: number, failed: number, pending: number, checks: Array<Object>}}
 */
function runTests(logElement) {
  const checks = [];
  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const check = (name, fn) => {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw5): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Shared helper: baseline RMSE on a toy split.
  check("computeBaselineRMSE: toy split matches the hand-computed value", () =>
    withFixture(
      {
        train: [
          { userId: 1, movieId: 1, rating: 2, timestamp: 1 },
          { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
        ],
      },
      () => {
        const rmse = computeBaselineRMSE(trainSet);
        if (!close(rmse, 1)) throw new Error(`rmse = ${rmse}, expected 1`);
        return `rmse=${fmt(rmse)}`;
      },
    ),
  );

  // 2. Shared helper: zero-denominator guard.
  check("computeBaselineRMSE: empty split returns 0 (no division by zero)", () =>
    withFixture({ train: [{ userId: 1, movieId: 1, rating: 3, timestamp: 1 }] }, () => {
      if (computeBaselineRMSE([]) !== 0) throw new Error("expected 0");
      return "guarded";
    }),
  );

  // 3. Formatting helper.
  check("fmt: formats numbers and guards non-finite input", () => {
    if (fmt(1.23456) !== "1.2346") throw new Error(`fmt(1.23456) = ${fmt(1.23456)}`);
    if (fmt(2, 2) !== "2.00") throw new Error(`fmt(2, 2) = ${fmt(2, 2)}`);
    if (fmt(Number.NaN) !== "\u2014") throw new Error("NaN must render as an em dash");
    return "ok";
  });

  // 4. Formatting helper.
  check("movieTitle: resolves ids and falls back for unknown ids", () =>
    withFixture({ movies: [{ id: 7, title: "Seven Samurai", genres: [] }] }, () => {
      if (movieTitle(7) !== "Seven Samurai") throw new Error("known id failed");
      if (movieTitle(999) !== "Movie #999") throw new Error(`fallback = ${movieTitle(999)}`);
      return "ok";
    }),
  );

  // 5. Student stub: prediction and clamp (TF.js gather + dot).
  check("predictRating: clamps to [1, 5] and falls back to the global mean", () =>
    withFixture(
      {
        movies: TOY_MOVIES,
        train: TOY_RATINGS,
        model: fixtureModel({ 1: [1, 2] }, { 3: [3, 4] }, 3),
      },
      () => {
        const pred = predictRating(1, 3);
        if (pred !== 5) throw new Error(`predictRating = ${pred}, expected 5 (3 + 11 clamped)`);
        const fallback = predictRating(100000, 3);
        if (fallback !== 3) throw new Error(`out-of-range user must return globalMean, got ${fallback}`);
        return `clamped=${pred}, fallback=${fallback}`;
      },
    ),
  );

  // 6. Student stub: TF.js variable allocation on the toy 5-rating set.
  check("initializeFactors: allocates tf.Variable factors and the global mean", () =>
    withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, model: null }, () => {
      const created = initializeFactors(TOY_RATINGS, 4);
      if (!created || !model) throw new Error("model was not assigned");
      if (!close(model.globalMean, 3.8)) {
        throw new Error(`globalMean = ${model.globalMean}, expected 3.8`);
      }
      if (!isTfModel(model)) throw new Error("userFactors/itemFactors are not tf tensors");
      if (model.userFactors.shape[1] !== 4 || model.itemFactors.shape[1] !== 4) {
        throw new Error(
          `factor width = ${model.userFactors.shape[1]} / ${model.itemFactors.shape[1]}, expected 4`,
        );
      }
      if (model.itemFactors.shape[0] !== 4) {
        throw new Error(`item rows = ${model.itemFactors.shape[0]}, expected 4 (max movie id 3 + 1)`);
      }
      if (model.userFactors.shape[0] < 3) {
        throw new Error(`user rows = ${model.userFactors.shape[0]}, expected at least 3`);
      }
      const row = model.userFactors.slice([1, 0], [1, 4]).dataSync();
      if (!Array.from(row).every(Number.isFinite)) {
        throw new Error("user factor row is not finite");
      }
      return `user [${model.userFactors.shape}], item [${model.itemFactors.shape}]`;
    }),
  );

  // 7. Student stub: batched training loop.
  check("trainMF: returns one finite history row per epoch", () =>
    withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, val: TOY_RATINGS, model: null }, () => {
      initializeFactors(TOY_RATINGS, 4);
      const history = trainMF(TOY_RATINGS, TOY_RATINGS, {
        epochs: 3,
        lr: 0.02,
        reg: 0.05,
        batchSize: 2,
      });
      if (!Array.isArray(history) || history.length !== 3) {
        throw new Error(`history length = ${history && history.length}, expected 3`);
      }
      history.forEach((row, i) => {
        if (row.epoch !== i + 1 && row.epoch !== i) {
          throw new Error(`row ${i} epoch = ${row.epoch}`);
        }
        if (!Number.isFinite(row.trainRMSE) || !Number.isFinite(row.valRMSE)) {
          throw new Error(`row ${i} RMSE is not finite`);
        }
      });
      return history.map((r) => `e${r.epoch}:${fmt(r.trainRMSE)}`).join(" ");
    }),
  );

  // 8. Student stub: checkpoint selection and tie-break.
  check("selectBestCheckpoint: smallest valRMSE wins, ties keep the earliest epoch", () => {
    const history = [
      { epoch: 2, trainRMSE: 0.9, valRMSE: 0.4 },
      { epoch: 1, trainRMSE: 1.0, valRMSE: 0.3 },
      { epoch: 3, trainRMSE: 0.8, valRMSE: 0.3 },
    ];
    const best = selectBestCheckpoint(history);
    if (!best || best.epoch !== 1) {
      throw new Error(`best epoch = ${best && best.epoch}, expected 1 (tie -> earliest)`);
    }
    return `best epoch ${best.epoch}`;
  });

  // 9. Student stub: recommendation filtering (tensor matMul + topk).
  check("recommendTopK: excludes train+val items but allows test-rated items", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: REC_FIXTURE.val,
        test: REC_FIXTURE.test,
        model: fixtureModel({ 1: 1 }, { 1: 5, 2: 4, 3: 3, 4: 2, 5: 1 }),
      },
      () => {
        const recs = recommendTopK(1, 10);
        const ids = recs.map((r) => r.movieId);
        if (ids.includes(1)) throw new Error("movie 1 is in train and must be excluded");
        if (ids.includes(2)) throw new Error("movie 2 is in val and must be excluded");
        if (!ids.includes(3)) throw new Error("movie 3 is test-rated and must be eligible");
        for (let i = 1; i < recs.length; i++) {
          if (recs[i - 1].score < recs[i].score) {
            throw new Error("results are not sorted by descending score");
          }
        }
        return ids.join(", ");
      },
    ),
  );

  // 10. Student stub: recall at rank 1 vs rank 11.
  check("recallAtK: relevant movie at rank 1 scores 1.0; at rank 11 scores 0.0", () => {
    const rankOne = withFixture({ ...recallFixture(1), history: [] }, () => recallAtK(10));
    if (!close(rankOne.overallRecall, 1)) {
      throw new Error(`rank-1 recall = ${rankOne.overallRecall}, expected 1`);
    }
    const rankEleven = withFixture({ ...recallFixture(11), history: [] }, () => recallAtK(10));
    if (!close(rankEleven.overallRecall, 0)) {
      throw new Error(`rank-11 recall = ${rankEleven.overallRecall}, expected 0`);
    }
    return `rank1=${fmt(rankOne.overallRecall)} rank11=${fmt(rankEleven.overallRecall)}`;
  });

  // 11. Student stub: empty-result guard.
  check("recallAtK: empty test set returns 0 without dividing by zero", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: [],
        test: [],
        model: fixtureModel({ 1: 1 }, { 1: 1, 2: 1, 3: 1, 4: 1, 5: 1 }),
      },
      () => {
        const result = recallAtK(10);
        if (result.overallRecall !== 0) {
          throw new Error(`expected 0, got ${result.overallRecall}`);
        }
        if (!Array.isArray(result.perUser) || result.perUser.length !== 0) {
          throw new Error("perUser must be empty");
        }
        if (!result.excluded) throw new Error("excluded report missing");
        return "guarded";
      },
    ),
  );

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW5 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` \u2014 ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/**
 * Select the TF.js backend, load the MovieLens data, split it chronologically,
 * and wire the page. `data.js` must be loaded before this script.
 *
 * If TensorFlow.js is missing, `initTf` renders a clear message and this
 * function stops without throwing.
 *
 * @returns {Promise<void>}
 */
window.onload = async function () {
  const status = document.getElementById("status");
  const backend = await initTf();
  if (!backend) return;

  try {
    if (status) status.textContent = "Loading data\u2026";
    await loadData();
    const split = splitByTimestamp();
    trainSet = split.train;
    valSet = split.val;
    testSet = split.test;
    init();
  } catch (error) {
    console.error("Initialization failed:", error);
    if (status) status.textContent = `Error: ${error.message}`;
  }
};
