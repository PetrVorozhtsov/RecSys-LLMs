"use strict";

// Optional reproduction helper. Start `python -m http.server 8000` in week4,
// install Playwright, then run this script with WEEK4_URL=http://localhost:8000/.
const assert = require("node:assert/strict");
const path = require("node:path");
const { chromium } = require("playwright");

const url = process.env.WEEK4_URL || "http://127.0.0.1:8000/";
const edge = process.env.EDGE_PATH || "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

async function capture() {
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 2 });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  try {
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => document.querySelector("#retained-count")?.textContent.trim() === "2,030", { timeout: 30000 });
    await page.locator("#run-tests").click();
    await page.waitForFunction(() => document.querySelector("#test-log")?.textContent.includes("5/5 checks passed"));

    async function select(tab, query, antecedent, consequent, expectedJoint) {
      await page.locator(`#${tab}-tab`).click();
      await page.locator("#rule-search").fill(query);
      await page.evaluate(({ antecedent, consequent }) => {
        const wanted = `${JSON.stringify([antecedent])}→${JSON.stringify([consequent])}`;
        const row = [...document.querySelectorAll("#rule-list .rule-row")]
          .find(element => element.dataset.ruleKey === wanted);
        if (!row) throw new Error(`Rule ${wanted} is not in this view`);
        row.click();
      }, { antecedent, consequent });
      const detail = await page.locator("#rule-detail").innerText();
      assert.ok(detail.includes(antecedent), `missing code ${antecedent}`);
      assert.ok(detail.includes(expectedJoint), `missing joint count ${expectedJoint}`);
      await page.evaluate(() => {
        const top = document.querySelector(".workspace-grid").getBoundingClientRect().top + scrollY;
        window.scrollTo(0, top - 16);
      });
    }

    await select("retained", "22386 → 85099B", "22386", "85099B", "825");
    await page.screenshot({ path: path.join(__dirname, "promising_rule.png") });

    await select("rejected", "22423 → 85099B", "22423", "85099B", "285");
    await page.screenshot({ path: path.join(__dirname, "rejected_rule.png") });

    await page.locator(".preset-button[data-support='2.0']").click();
    await page.waitForFunction(() => document.querySelector("#retained-count")?.textContent.trim() === "62", { timeout: 30000 });
    await select("retained", "22386 → 85099B", "22386", "85099B", "825");
    await page.screenshot({ path: path.join(__dirname, "high_evidence.png") });
    assert.deepEqual(errors, []);
    console.log("Captured three verified UI scenarios; built-in checks 5/5 passed.");
  } finally {
    await browser.close();
  }
}

capture().catch(error => { console.error(error); process.exitCode = 1; });
