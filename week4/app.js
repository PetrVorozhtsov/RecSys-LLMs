/* Week 4 interface for transaction-based association-rule exploration. */
(() => {
    "use strict";

    const state = {
        data: null,
        result: null,
        rejected: [],
        activeTab: "retained",
        selectedKey: null,
        running: false,
        rerunRequested: false,
        lastOptions: null,
    };
    const numberFormatter = new Intl.NumberFormat("en-US");
    const $ = (id) => document.getElementById(id);

    function escapeHTML(value) {
        return String(value).replace(/[&<>"']/g, (character) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
        })[character]);
    }

    function formatCount(value) {
        return numberFormatter.format(Number(value) || 0);
    }

    function formatPercent(value) {
        const percent = Number(value) * 100;
        return `${percent < 1 ? percent.toFixed(2) : percent.toFixed(1)}%`;
    }

    function formatLift(value) {
        return `${Number(value).toFixed(2)}×`;
    }

    function ruleKey(rule) {
        return `${JSON.stringify(rule.antecedent)}→${JSON.stringify(rule.consequent)}`;
    }

    function itemName(code) {
        const description = state.data && state.data.items && state.data.items[code];
        return description || String(code);
    }

    function itemNames(codes) {
        return codes.map(itemName).join(" + ");
    }

    function ruleCodes(rule) {
        return `${rule.antecedent.join(" + ")} → ${rule.consequent.join(" + ")}`;
    }

    function ruleTitle(rule) {
        return `${itemNames(rule.antecedent)} → ${itemNames(rule.consequent)}`;
    }

    function currentOptions() {
        return {
            minSupport: Number($("support-input").value) / 100,
            minConfidence: Number($("confidence-input").value) / 100,
            maxSize: 10,
        };
    }

    function setStatus(message, kind = "working") {
        $("status").textContent = message;
        $("data-indicator").className = `status-indicator ${kind === "ready" ? "is-ready" : kind === "error" ? "is-error" : ""}`;
    }

    // Assignment: make the chosen support and confidence cutoffs explicit and justified.
    function updateThresholdExplanation() {
        const options = currentOptions();
        const supportPct = formatPercent(options.minSupport);
        const confidencePct = formatPercent(options.minConfidence);
        $("support-value").textContent = supportPct;
        $("confidence-value").textContent = confidencePct;
        const minimum = state.data ? Math.max(1, Math.ceil(options.minSupport * state.data.baskets.length - 1e-10)) : null;
        const countText = minimum === null ? "" : ` (at least ${formatCount(minimum)} of ${formatCount(state.data.baskets.length)} invoices)`;
        $("threshold-rationale").textContent =
            `At ${supportPct} support${countText}, a rule needs enough joint transactions to be credible. ` +
            `At ${confidencePct} confidence, B must occur in at least that share of baskets containing A. ` +
            `Lower thresholds reveal rarer patterns but add noise and computation; higher thresholds require stronger evidence. ` +
            `Only lift above 1 is retained for further analysis.`;

        for (const button of document.querySelectorAll(".preset-button")) {
            const match = Number(button.dataset.support) === Number($("support-input").value) &&
                Number(button.dataset.confidence) === Number($("confidence-input").value);
            button.classList.toggle("is-active", match);
        }

        if (state.result && state.lastOptions &&
            (options.minSupport !== state.lastOptions.minSupport || options.minConfidence !== state.lastOptions.minConfidence)) {
            $("results-context").textContent = "Thresholds changed · press Run rules";
        }
    }

    function setActiveTab(tab) {
        state.activeTab = tab;
        for (const name of ["retained", "rejected"]) {
            const button = $(`${name}-tab`);
            const active = name === tab;
            button.classList.toggle("is-active", active);
            button.setAttribute("aria-selected", String(active));
        }
        $("rule-list").setAttribute("aria-labelledby", `${tab}-tab`);
        state.selectedKey = null;
        renderList();
        renderDetail();
    }

    function emptyState(title, description) {
        $("rule-list").innerHTML = `<div class="empty-state"><span class="empty-symbol" aria-hidden="true">◇</span><h3>${escapeHTML(title)}</h3><p>${escapeHTML(description)}</p></div>`;
        $("list-footnote").textContent = "";
    }

    function visibleRules() {
        return state.activeTab === "retained" ? state.result.rules : state.rejected;
    }

    function rejectionReason(rule) {
        const belowConfidence = rule.confidence + 1e-12 < state.lastOptions.minConfidence;
        const lowLift = rule.lift <= 1 + 1e-12;
        if (belowConfidence && lowLift) return "Below confidence threshold and no positive lift";
        if (belowConfidence) return "Below confidence threshold";
        if (lowLift) return "Lift does not exceed baseline";
        return "Does not pass the retained-rule filter";
    }

    // Assignment: show both lift>1 discoveries and rejected candidates for comparison.
    function renderList() {
        if (!state.result) return;
        const search = $("rule-search").value.trim().toLowerCase();
        const exactPair = search.match(/^([^\s]+)\s*(?:→|->)\s*([^\s]+)$/);
        const terms = search.split(/\s+/).filter(Boolean);
        const pool = visibleRules();
        const matching = exactPair ? pool.filter((rule) =>
            rule.antecedent.length === 1 && rule.consequent.length === 1 &&
            rule.antecedent[0].toLowerCase() === exactPair[1] &&
            rule.consequent[0].toLowerCase() === exactPair[2]) :
            terms.length ? pool.filter((rule) => {
                const searchable = `${ruleTitle(rule)} ${ruleCodes(rule)}`.toLowerCase();
                return terms.every((term) => searchable.includes(term));
            }) : pool;
        if (!matching.length) {
            emptyState(terms.length ? "No matching rules" : "No rules in this view",
                terms.length ? "Try another product name or StockCode." :
                    "Try a lower threshold or inspect the other rule category.");
            return;
        }

        const displayed = matching.slice(0, 60);
        $("rule-list").innerHTML = displayed.map((rule) => {
            const key = ruleKey(rule);
            const isSelected = key === state.selectedKey;
            const labelA = escapeHTML(itemNames(rule.antecedent));
            const labelB = escapeHTML(itemNames(rule.consequent));
            const reason = state.activeTab === "rejected" ? rejectionReason(rule) : "StockCode " + ruleCodes(rule);
            return `<button type="button" class="rule-row${isSelected ? " is-selected" : ""}" data-rule-key="${escapeHTML(key)}" aria-pressed="${isSelected}" title="${escapeHTML(ruleTitle(rule))}">` +
                `<span class="rule-name"><strong>${labelA}<span class="arrow">→</span>${labelB}</strong><small>${escapeHTML(reason)}</small></span>` +
                `<span class="rule-stat">${formatPercent(rule.support)}<small>support</small></span>` +
                `<span class="rule-stat">${formatPercent(rule.confidence)}<small>confidence</small></span>` +
                `<span class="rule-stat ${rule.lift > 1 ? "lift-good" : "lift-poor"}">${formatLift(rule.lift)}<small>lift</small></span></button>`;
        }).join("");
        $("list-footnote").textContent = `Showing ${formatCount(displayed.length)} of ${formatCount(matching.length)} ${state.activeTab} candidates. Select a row for the full calculation.`;
    }

    // Assignment: expose N, nA, nB, nAB and the formulas behind each rule metric.
    function renderDetail() {
        const tag = $("detail-tag");
        const detail = $("rule-detail");
        const selected = state.result && visibleRules().find((rule) => ruleKey(rule) === state.selectedKey);
        if (!selected) {
            tag.textContent = "No selection";
            tag.className = "detail-tag";
            detail.className = "detail-empty";
            detail.textContent = "Select a rule above to see the counts and formulas behind its metrics.";
            return;
        }

        const rejected = state.activeTab === "rejected";
        tag.textContent = rejected ? "Rejected candidate" : "Retained rule";
        tag.className = `detail-tag ${rejected ? "is-rejected" : "is-retained"}`;
        detail.className = "";
        const N = state.result.N;
        const baseline = selected.nB / N;
        const explanation = rejected ?
            `${rejectionReason(selected)}. This association should not be promoted as evidence for a cross-sell without further investigation.` :
            `B occurs in ${formatPercent(baseline)} of all baskets but in ${formatPercent(selected.confidence)} of baskets containing A. The positive lift is a discovery signal, not causal proof.`;
        detail.innerHTML =
            `<h3 class="detail-rule">${escapeHTML(itemNames(selected.antecedent))}<span class="arrow">→</span>${escapeHTML(itemNames(selected.consequent))}</h3>` +
            `<p class="detail-codes">StockCodes: ${escapeHTML(ruleCodes(selected))}</p>` +
            `<div class="detail-grid">` +
            `<div class="detail-count"><span>N · all invoices</span><strong>${formatCount(N)}</strong></div>` +
            `<div class="detail-count"><span>nA · contain A</span><strong>${formatCount(selected.nA)}</strong></div>` +
            `<div class="detail-count"><span>nB · contain B</span><strong>${formatCount(selected.nB)}</strong></div>` +
            `<div class="detail-count"><span>nAB · contain both</span><strong>${formatCount(selected.nAB)}</strong></div></div>` +
            `<div class="formula-grid">` +
            `<div class="formula-card"><span>Support</span><strong>${formatPercent(selected.support)}</strong><small>nAB / N = ${formatCount(selected.nAB)} / ${formatCount(N)}</small></div>` +
            `<div class="formula-card"><span>Confidence</span><strong>${formatPercent(selected.confidence)}</strong><small>nAB / nA = ${formatCount(selected.nAB)} / ${formatCount(selected.nA)}</small></div>` +
            `<div class="formula-card"><span>Lift</span><strong>${formatLift(selected.lift)}</strong><small>confidence / (nB / N)</small></div></div>` +
            `<p class="interpretation${rejected ? " is-rejected" : ""}">${escapeHTML(explanation)}</p>`;
    }

    function displayResult(result, options) {
        if (!result || !Array.isArray(result.rules) || !Array.isArray(result.candidates)) {
            throw new Error("The mining result does not include rule lists.");
        }
        state.result = result;
        state.lastOptions = options;
        const retainedKeys = new Set(result.rules.map(ruleKey));
        const lowLift = [];
        const lowConfidence = [];
        for (const candidate of result.candidates) {
            if (!retainedKeys.has(ruleKey(candidate))) {
                // High-confidence rules with lift<=1 make the misleading-rule lesson clearest.
                (candidate.confidence >= options.minConfidence && candidate.lift <= 1 ? lowLift : lowConfidence).push(candidate);
            }
        }
        lowLift.sort((left, right) => right.confidence - left.confidence || right.support - left.support);
        lowConfidence.sort((left, right) => right.support - left.support || right.confidence - left.confidence);
        state.rejected = lowLift.concat(lowConfidence);
        state.selectedKey = null;

        $("itemset-count").textContent = formatCount(result.itemsets.length);
        $("retained-count").textContent = formatCount(result.rules.length);
        $("rejected-count").textContent = formatCount(state.rejected.length);
        $("retained-tab-count").textContent = formatCount(result.rules.length);
        $("rejected-tab-count").textContent = formatCount(state.rejected.length);
        $("results-context").textContent = `${formatPercent(options.minSupport)} support · ${formatPercent(options.minConfidence)} confidence`;
        $("rule-search").value = "";
        state.activeTab = "retained";
        $("retained-tab").classList.add("is-active");
        $("retained-tab").setAttribute("aria-selected", "true");
        $("rejected-tab").classList.remove("is-active");
        $("rejected-tab").setAttribute("aria-selected", "false");
        $("rule-list").setAttribute("aria-labelledby", "retained-tab");
        renderList();
        renderDetail();
    }

    async function runRules() {
        if (state.running) {
            state.rerunRequested = true;
            return;
        }
        if (!state.data) return;
        if (!globalThis.AssociationRules || typeof AssociationRules.mine !== "function") {
            setStatus("Rule miner is unavailable. Check that apriori.js was loaded.", "error");
            return;
        }
        const options = currentOptions();
        state.running = true;
        $("run-rules").disabled = true;
        setStatus("Mining frequent itemsets and evaluating directional rules…");
        // Yield a frame so the busy state becomes visible before synchronous Apriori work.
        await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
        try {
            const started = performance.now();
            const result = AssociationRules.mine(state.data.baskets, options);
            displayResult(result, options);
            const seconds = ((performance.now() - started) / 1000).toFixed(1);
            setStatus(`Ready: ${formatCount(result.N)} invoices analysed in ${seconds}s. Select a rule to inspect its evidence.`, "ready");
        } catch (error) {
            console.error(error);
            setStatus(`Could not mine rules: ${error.message}`, "error");
        } finally {
            state.running = false;
            $("run-rules").disabled = false;
            if (state.rerunRequested) {
                state.rerunRequested = false;
                runRules();
            }
        }
    }

    // Assignment: self-check support, confidence, lift and lift>1 filtering on known baskets.
    function runTests() {
        if (!globalThis.AssociationRules || typeof AssociationRules.mine !== "function") {
            $("test-log").textContent = "FAIL · apriori.js is not available.";
            return;
        }
        const fixture = [
            ["bread", "milk", "jam"],
            ["bread", "milk", "jam"],
            ["bread", "milk", "eggs"],
            ["bread", "jam", "eggs"],
            ["bread"],
        ];
        const checks = [];
        function check(label, condition) { checks.push(`${condition ? "PASS" : "FAIL"} · ${label}`); }
        function close(actual, expected) { return Math.abs(actual - expected) < 1e-10; }
        try {
            const result = AssociationRules.mine(fixture, { minSupport: .2, minConfidence: .1, maxSize: 3 });
            const find = (a, b) => result.candidates.find((rule) =>
                rule.antecedent.length === 1 && rule.antecedent[0] === a &&
                rule.consequent.length === 1 && rule.consequent[0] === b);
            const breadMilk = find("bread", "milk");
            const milkJam = find("milk", "jam");
            const jamEggs = find("jam", "eggs");
            check("Five invoices are the denominator", result.N === 5);
            check("Bread → milk: 3/5 support, 3/5 confidence, lift 1",
                !!breadMilk && breadMilk.nA === 5 && breadMilk.nB === 3 && breadMilk.nAB === 3 &&
                close(breadMilk.support, .6) && close(breadMilk.confidence, .6) && close(breadMilk.lift, 1));
            check("Milk → jam: 2/5 support, 2/3 confidence, 10/9 lift",
                !!milkJam && milkJam.nA === 3 && milkJam.nB === 3 && milkJam.nAB === 2 &&
                close(milkJam.support, .4) && close(milkJam.confidence, 2 / 3) && close(milkJam.lift, 10 / 9));
            check("Jam → eggs: 1/5 support, 1/3 confidence, 5/6 lift",
                !!jamEggs && jamEggs.nAB === 1 && close(jamEggs.support, .2) &&
                close(jamEggs.confidence, 1 / 3) && close(jamEggs.lift, 5 / 6));
            check("Only positive-lift rules are retained",
                result.rules.some((rule) => ruleKey(rule) === ruleKey(milkJam)) &&
                !result.rules.some((rule) => ruleKey(rule) === ruleKey(breadMilk)));
        } catch (error) {
            checks.push(`FAIL · ${error.message}`);
        }
        const passing = checks.filter((line) => line.startsWith("PASS")).length;
        $("test-log").textContent = `${passing}/${checks.length} checks passed\n${checks.join("\n")}`;
        $("test-log").classList.toggle("has-failures", passing !== checks.length);
    }

    async function loadData() {
        try {
            const response = await fetch("data/baskets.json");
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (!data || !Array.isArray(data.baskets) || !data.baskets.length ||
                !data.items || typeof data.items !== "object") {
                throw new Error("baskets.json has an unexpected structure");
            }
            state.data = data;
            $("basket-count").textContent = formatCount(data.baskets.length);
            $("item-count").textContent = formatCount(Object.keys(data.items).length);
            $("source-label").textContent = data.source && data.source.name ? "UCI" : "Retail";
            const preprocessing = data.preprocessing || {};
            const dateRange = preprocessing.first_date && preprocessing.last_date ?
                ` The retained invoices span ${preprocessing.first_date} to ${preprocessing.last_date}.` : "";
            const method = typeof preprocessing === "string" ? preprocessing : preprocessing.method;
            $("method-copy").textContent = `${method || "Each InvoiceNo is one transaction. StockCode identifies an item, and Description labels it."}${dateRange} A high lift points to an unusual co-occurrence; it does not prove a promotion will increase sales.`;
            updateThresholdExplanation();
            $("run-rules").disabled = false;
            setStatus(`Loaded ${formatCount(data.baskets.length)} invoice baskets and ${formatCount(Object.keys(data.items).length)} product codes.`, "ready");
            // Default scenario is calculated immediately; presets make later parameter comparisons repeatable.
            runRules();
        } catch (error) {
            console.error(error);
            const hint = location.protocol === "file:" ? " Serve the repository over HTTP instead of opening file://." : "";
            setStatus(`Could not load data/baskets.json: ${error.message}.${hint}`, "error");
            emptyState("Dataset unavailable", "Serve the project over HTTP and confirm data/baskets.json exists.");
        }
    }

    function initialize() {
        $("support-input").addEventListener("input", updateThresholdExplanation);
        $("confidence-input").addEventListener("input", updateThresholdExplanation);
        for (const button of document.querySelectorAll(".preset-button")) {
            button.addEventListener("click", () => {
                $("support-input").value = button.dataset.support;
                $("confidence-input").value = button.dataset.confidence;
                updateThresholdExplanation();
                if (state.data) runRules();
            });
        }
        $("run-rules").addEventListener("click", runRules);
        $("run-tests").addEventListener("click", runTests);
        $("retained-tab").addEventListener("click", () => setActiveTab("retained"));
        $("rejected-tab").addEventListener("click", () => setActiveTab("rejected"));
        $("rule-search").addEventListener("input", renderList);
        $("rule-list").addEventListener("click", (event) => {
            const row = event.target.closest(".rule-row");
            if (!row || !state.result) return;
            state.selectedKey = row.dataset.ruleKey;
            renderList();
            renderDetail();
        });
        updateThresholdExplanation();
        loadData();
    }

    document.addEventListener("DOMContentLoaded", initialize);
})();
