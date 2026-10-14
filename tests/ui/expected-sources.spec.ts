import { expect, test } from "@playwright/test";

test("expected sources survive generation, review, publication and scored results", async ({ page }) => {
  const runtimeErrors: string[] = [];
  page.on("pageerror", (error) => runtimeErrors.push(error.message));
  const name = `Source expectations ${Date.now()}`;
  await page.goto("/evaluations");
  await page.getByRole("button", { name: /Datasets/ }).click();
  await page.getByLabel("Name").first().fill(name);
  await page.getByLabel("Dataset file").setInputFiles({
    name: "sources.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({schema_version: "qa-dataset-v2", items: [{
      id: "source-item", question: `What is the operations policy? ${name}`, answer: "Follow the policy.", time_sensitive: false,
      expected_sources: [" Operations Handbook.pdf ", "https://example.org/policy"],
    }]})),
  });
  await page.getByRole("button", { name: "Validate and import" }).click();
  await page.locator("#dataset-list [data-dataset-id]").filter({has: page.getByText(name, {exact: true})}).click();
  await page.getByRole("button", { name: "Generate Atoms", exact: true }).click();
  const sources = page.getByLabel("Expected sources for source-item", {exact: true});
  await expect(sources).toHaveValue(" Operations Handbook.pdf \nhttps://example.org/policy");
  await sources.fill("Operations Handbook.pdf\noperations handbook.pdf");
  await page.getByRole("button", {name: "Save as new dataset"}).click();
  await expect(page.getByRole("alert")).toContainText("Expected sources must be unique");
  await sources.fill(" Operations Handbook.pdf \nhttps://example.org/policy\nMissing <manual>.pdf");
  await page.getByRole("button", {name: "Add atom to source-item"}).click();
  await expect(sources).toHaveValue(" Operations Handbook.pdf \nhttps://example.org/policy\nMissing <manual>.pdf");
  await page.getByLabel("Atom text 2 for source-item").fill("Follow the policy.");
  const saveResponse = page.waitForResponse((response) => response.url().endsWith("/save") && response.request().method() === "POST");
  await page.getByRole("button", {name: "Save as new dataset"}).click();
  const child = (await (await saveResponse).json()).dataset;
  await expect(page.locator("#dataset-detail").getByRole("heading", {name: `${name} · reviewed`})).toBeVisible();
  await page.getByRole("button", {name: "Review Atoms", exact: true}).click();
  await expect(sources).toHaveValue(" Operations Handbook.pdf \nhttps://example.org/policy\nMissing <manual>.pdf");
  await page.getByRole("button", {name: "Close", exact: true}).click();
  await page.locator("#evaluate-dataset").click();
  await page.getByLabel("Evaluation name").fill(`${name} run`);
  await page.getByRole("button", {name: "Start evaluation"}).click();
  await expect(page.getByRole("heading", {name: `${name} run`})).toBeVisible({timeout: 20000});
  await page.locator(".question-result > summary").click();
  await page.locator(".attempt-result > summary").first().click();
  const evidence = page.getByRole("region", {name: "Expected source matches"});
  await expect(evidence).toBeVisible();
  await expect(evidence.locator(".source-judgment")).toHaveCount(3);
  await expect(evidence.getByText("Matched", {exact: true})).toHaveCount(2);
  await expect(evidence.getByText("Missing", {exact: true})).toHaveCount(1);
  await expect(evidence.getByText("fixture_document_search", {exact: true})).toHaveCount(2);
  await expect(evidence.getByText("Missing <manual>.pdf", {exact: true})).toBeVisible();
  await expect(evidence.getByText("66.7% recall", {exact: true})).toBeVisible();
  expect(await evidence.locator("manual").count()).toBe(0);
  await page.locator(".tool-call-disclosure > summary").click();
  await page.locator(".tool-call-detail > summary").click();
  await expect(page.locator(".tool-call-body")).toContainText("OPERATIONS HANDBOOK.PDF");
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.reload();
  await page.locator("#runs-body tr").filter({hasText: `${name} run`}).click();
  await expect(page.getByRole("heading", {name: `${name} run`})).toBeVisible();
  const catalog = await (await page.request.get("/api/evaluations/catalog")).json();
  expect(catalog.datasets.some((dataset: {id: string}) => dataset.id === child.id)).toBe(true);
  expect(runtimeErrors).toEqual([]);
});

test("source evidence distinguishes unavailable historical responses from unscored checks", async ({page}) => {
  const historyId = "historical-source-checks";
  await page.route("**/api/evaluations/runs?*", (route) => route.fulfill({json: {runs: [{id: historyId, valid: true, name: "Historical source checks", status: "scored", attempts: 2}]}}));
  await page.route(`**/api/evaluations/runs/${historyId}`, (route) => route.fulfill({json: {run: {
    manifest: {schema_version: "qa-v1", status: "scored", run_id: historyId},
    metadata: {name: "Historical source checks"},
    prepared_items: [{item_id: "historical", question: "Find the document", expected_sources: ["Historical.pdf"]}],
    answers: [1,2].map((ordinal) => ({item_id: "historical", attempt_id: `historical-${ordinal}`, ordinal, status: "answer_ready", answer: "Answer", tool_calls: [{ordinal: 1, name: "legacy_search", status: "success"}]})),
    evaluation_results: [{item_id: "historical", attempt_id: "historical-1", ordinal: 1, status: "scored", source_evaluation: {status: "unavailable", recall: null, matches: [{source: "Historical.pdf", outcome: "unavailable", matching_calls: []}]}}],
    report_available: false,
  }}}));
  await page.goto("/evaluations");
  await page.locator(`[data-run-id="${historyId}"]`).click();
  await page.locator(".question-result > summary").click();
  for (const attempt of await page.locator(".attempt-result").all()) await attempt.locator(":scope > summary").click();
  const cards = page.locator(".source-judgment");
  await expect(cards).toHaveCount(2);
  await expect(cards.nth(0).getByText("Unavailable", {exact: true})).toBeVisible();
  await expect(cards.nth(1).getByText("Not evaluated", {exact: true})).toBeVisible();
  await expect(page.locator(".source-evidence").getByText("Missing", {exact: true})).toHaveCount(0);
});
