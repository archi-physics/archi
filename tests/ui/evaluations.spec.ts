import { expect, test } from "@playwright/test";

test.describe("QA evaluation console", () => {
  test.describe.configure({ mode: "serial" });

  test("stacks dataset and profile imports into spaced form rows", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto("/evaluations");

    const expectStackedImportRows = async (formSelector: string) => {
      const rows = page.locator(`${formSelector} > label, ${formSelector} > button`);
      await expect(rows).toHaveCount(3);
      const boxes = await rows.evaluateAll((elements) => elements.map((element) => {
        const bounds = element.getBoundingClientRect();
        return {
          x: Math.round(bounds.x),
          y: Math.round(bounds.y),
          width: Math.round(bounds.width),
          height: Math.round(bounds.height),
        };
      }));

      expect(boxes[1].y).toBeGreaterThanOrEqual(boxes[0].y + boxes[0].height + 19);
      expect(boxes[2].y).toBeGreaterThanOrEqual(boxes[1].y + boxes[1].height + 19);
      expect(boxes[0].x).toBe(boxes[1].x);
      expect(boxes[2].height).toBeGreaterThanOrEqual(44);
    };

    await page.getByRole("button", { name: /Datasets/ }).click();
    await expectStackedImportRows("#dataset-import-form");

    await page.getByRole("button", { name: /Profiles/ }).click();
    await expectStackedImportRows("#profile-import-form");
  });

  test("shows the persisted live pre-check gate without exposing hidden truth", async ({ page }) => {
    const job = {
      id: "job-live-attention",
      kind: "evaluation",
      status: "attention_required",
      created_at: "2026-08-13T08:00:00+00:00",
      started_at: "2026-08-13T08:00:01+00:00",
      context: { dataset_id: "approved-live", workspace_id: "run-live" },
      result: {
        status: "attention_required",
        attention_required: {
          live_items: 2,
          checked_at: "2026-08-13T08:00:02+00:00",
          can_continue: true,
          no_agent_attempts_started: true,
          affected_items: [
            { item_id: "live-capacity", phase: "pre_run", reason: "answer_changed" },
          ],
        },
      },
    };
    await page.route("**/api/evaluations/catalog", (route) => route.fulfill({
      json: { datasets: [], profiles: [], agents: [], jobs: [job], permissions: { can_manage: false } },
    }));
    await page.route("**/api/evaluations/runs?*", (route) => route.fulfill({ json: { runs: [] } }));
    await page.route("**/api/evaluations/jobs/job-live-attention", (route) => route.fulfill({ json: { job } }));

    await page.goto("/evaluations");

    await expect(page.getByRole("heading", { name: "Live answers need attention" })).toBeVisible();
    await expect(page.getByText("No agent attempts have started.", { exact: false }).first()).toBeVisible();
    await expect(page.locator("#attention-items").getByText("live-capacity", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Continue with valid questions" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Refresh live snapshot" })).toHaveCount(0);
    await expect(page.getByText("Approved answer")).toHaveCount(0);
    await expect(page.getByText("Contact a dataset manager to refresh the approved live snapshot.", { exact: false })).toBeVisible();
    await expect(page.locator("#metric-job")).toHaveText("No");
  });

  test("preserves manager-only live evidence after refreshing the catalog", async ({ page }) => {
    let continueRequests = 0;
    let refreshRequests = 0;
    const sanitizedJob = {
      id: "job-manager-attention", kind: "evaluation", status: "attention_required",
      created_at: "2026-08-13T08:00:00+00:00", started_at: "2026-08-13T08:00:01+00:00",
      context: { dataset_id: "approved-live", workspace_id: "run-live" },
      result: { attention_required: {
        live_items: 1, checked_at: "2026-08-13T08:00:02+00:00", can_continue: true,
        affected_items: [{ item_id: "live-capacity", phase: "pre_run", reason: "answer_changed" }],
      } },
    };
    const detailedJob = {
      ...sanitizedJob,
      result: { attention_required: {
        ...sanitizedJob.result.attention_required,
        affected_items: [{
          item_id: "live-capacity", phase: "pre_run", reason: "answer_changed",
          question: "What is current capacity?", approved_answer: { capacity: 4 },
          current_answer: { capacity: 7 }, metadata: { region: "eu" },
          oracle: { calls: [] }, calls: [],
        }],
      } },
    };
    await page.route("**/api/evaluations/catalog", (route) => route.fulfill({
      json: { datasets: [], profiles: [], agents: [], jobs: [sanitizedJob], permissions: { can_run: false, can_manage: true } },
    }));
    await page.route("**/api/evaluations/runs?*", (route) => route.fulfill({ json: { runs: [] } }));
    await page.route("**/api/evaluations/jobs/job-manager-attention", (route) => route.fulfill({ json: { job: detailedJob } }));
    await page.route("**/api/evaluations/jobs/job-manager-attention/continue", (route) => {
      continueRequests += 1;
      route.fulfill({ json: { job: { ...detailedJob, status: "queued", result: undefined } } });
    });
    await page.route("**/api/evaluations/jobs/job-manager-attention/refresh-live", (route) => {
      refreshRequests += 1;
      route.fulfill({
        status: 202,
        json: {
          closed_evaluation: {
            job: { ...detailedJob, status: "canceled", result: undefined },
            history_id: "run-live",
          },
          job: {
            id: "job-live-refresh", kind: "generate_atoms", status: "completed",
            context: { dataset_id: "approved-live", refresh: true },
            result: { draft_id: "draft-live-refresh" },
          },
        },
      });
    });
    await page.route("**/api/evaluations/jobs/job-live-refresh", (route) => route.fulfill({ json: {
      job: {
        id: "job-live-refresh", kind: "generate_atoms", status: "completed",
        context: { dataset_id: "approved-live", refresh: true },
        result: { draft_id: "draft-live-refresh" },
      },
    } }));
    await page.route("**/api/evaluations/atom-drafts/draft-live-refresh", (route) => route.fulfill({ json: {
      draft: {
        id: "draft-live-refresh", status: "open", dataset_name: "Approved live",
        schema_version: "qa-dataset-v2", dataset_role: "approved_child",
        generation_scope: "refresh_live", items: [], failure_summary: {},
      },
    } }));

    await page.goto("/evaluations");

    await expect(page.getByText("What is current capacity?", { exact: true })).toBeVisible();
    await page.getByText("What is current capacity?", { exact: true }).click();
    await expect(page.getByText("Approved answer", { exact: true })).toBeVisible();
    await expect(page.getByText('"capacity": 4', { exact: false })).toBeVisible();
    await expect(page.getByRole("button", { name: "Refresh live snapshot" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Continue with valid questions" })).toHaveCount(0);
    await expect(page.getByText("No agent attempts have started.", { exact: false }).first()).toBeVisible();
    await page.getByRole("button", { name: "Refresh live snapshot" }).click();
    await expect(page.getByRole("heading", { name: "Review atoms" })).toBeVisible();
    expect(refreshRequests).toBe(1);
    expect(continueRequests).toBe(0);
  });

  test("announces the live pre-check while an evaluation worker is active", async ({ page }) => {
    const job = {
      id: "job-live-running", kind: "evaluation", status: "running",
      created_at: "2026-08-13T08:00:00+00:00", started_at: "2026-08-13T08:00:01+00:00",
      context: { dataset_id: "approved-live" },
      progress: { status: "prepared", runtime_phase: "checking_live_answers" },
    };
    await page.route("**/api/evaluations/catalog", (route) => route.fulfill({
      json: { datasets: [], profiles: [], agents: [], jobs: [job], permissions: { can_manage: true } },
    }));
    await page.route("**/api/evaluations/runs?*", (route) => route.fulfill({ json: { runs: [] } }));
    await page.route("**/api/evaluations/jobs/job-live-running", (route) => route.fulfill({ json: { job } }));

    await page.goto("/evaluations");

    await expect(page.locator("#runtime-label")).toHaveText("Checking live answers…");
    await expect(page.getByText("Agent attempts start only after the fresh pre-check completes.", { exact: false })).toBeVisible();
    await expect(page.locator("#metric-job")).toHaveText("Yes");
  });

  test("shows ordinary progress for static evaluation attempts", async ({ page }) => {
    const job = {
      id: "job-running-attempts", kind: "evaluation", status: "running",
      created_at: "2026-08-13T08:00:00+00:00", started_at: "2026-08-13T08:00:01+00:00",
      context: { dataset_id: "static-dataset" },
      progress: { status: "prepared", runtime_phase: "running_attempts" },
    };
    await page.route("**/api/evaluations/catalog", (route) => route.fulfill({
      json: { datasets: [], profiles: [], agents: [], jobs: [job], permissions: { can_manage: true } },
    }));
    await page.route("**/api/evaluations/runs?*", (route) => route.fulfill({ json: { runs: [] } }));
    await page.route("**/api/evaluations/jobs/job-running-attempts", (route) => route.fulfill({ json: { job } }));

    await page.goto("/evaluations");

    await expect(page.locator("#runtime-label")).toHaveText("evaluation · running attempts");
    await expect(page.getByText("Agent attempts start only after the fresh pre-check completes.", { exact: false })).toHaveCount(0);
    await expect(page.locator("#metric-job")).toHaveText("Yes");
  });

  test("labels live datasets and counts static-only omissions", async ({ page }) => {
    const dataset = {
      id: "live-parent", name: "Live parent", source_filename: "live.json", sha256: "a".repeat(64),
      schema_version: "qa-dataset-v2", item_count: 2, eligible_item_count: 2,
      dataset_role: "definition_parent",
      time_sensitive_item_count: 1, supplied_atom_item_count: 1, atom_count: 1,
      contains_live_answers: false, generation_scope: null, parent_dataset_id: null,
      categories: [], answer_sources: [], created_at: "2026-08-13T08:00:00+00:00",
    };
    const liveChild = {
      ...dataset, id: "live-child", name: "Approved live child", contains_live_answers: true,
      dataset_role: "approved_child", generation_scope: "complete", parent_dataset_id: dataset.id,
    };
    const completedJob = {
      id: "job-static-draft", kind: "generate_atoms", status: "completed",
      context: { dataset_id: dataset.id }, result: { draft_id: "draft-static", draft_status: "open" },
    };
    const completeDraft = {
      id: "draft-static", status: "open", dataset_name: dataset.name, schema_version: "qa-dataset-v2", dataset_role: "definition_parent", generation_scope: "complete",
      items: [{ item_id: "live-capacity", question: "Current capacity?", time_sensitive: true,
        status: "preparation_failed", atoms: [], error: "Oracle unavailable." }],
    };
    const staticDraft = {
      ...completeDraft, generation_scope: "static_only",
      items: [{ ...completeDraft.items[0], status: "skipped_live" }],
    };
    await page.route("**/api/evaluations/catalog", (route) => route.fulfill({
      json: { datasets: [dataset, liveChild], profiles: [], agents: [], jobs: [completedJob], permissions: { can_manage: true } },
    }));
    await page.route("**/api/evaluations/runs?*", (route) => route.fulfill({ json: { runs: [] } }));
    await page.route("**/api/evaluations/atom-drafts/draft-static", async (route) => {
      if (route.request().method() === "POST") await route.fulfill({ json: { draft: staticDraft } });
      else await route.fulfill({ json: { draft: completeDraft } });
    });
    await page.route("**/api/evaluations/atom-drafts/draft-static/static-only", (route) => route.fulfill({ json: { draft: staticDraft } }));

    await page.goto("/evaluations");
    await page.getByRole("button", { name: /Datasets/ }).click();
    await expect(page.getByText("includes live answers", { exact: false })).toBeVisible();
    await page.getByRole("button", { name: "Resume atom review" }).click();
    await page.getByRole("button", { name: "Switch draft to static-only" }).click();

    await expect(page.getByText("1 live question was omitted by static-only generation.", { exact: false })).toBeVisible();
  });

  test("shows retryable live-validation slots without inventing agent answers", async ({ page }) => {
    const runRow = {
      id: "live-failed-run", run_id: "live-failed-run", name: "Live failed run",
      status: "scored", created_at: "2026-08-13T08:00:00+00:00",
      dataset_name: "Approved live set", attempts: 1, valid: true,
      overall_attempt_pass_rate: null,
    };
    await page.route("**/api/evaluations/catalog", (route) => route.fulfill({
      json: { datasets: [], profiles: [], agents: [], jobs: [], permissions: { can_manage: false } },
    }));
    await page.route("**/api/evaluations/runs?*", (route) => route.fulfill({ json: { runs: [runRow] } }));
    await page.route("**/api/evaluations/runs/live-failed-run", (route) => route.fulfill({
      json: {
        run: {
          manifest: { schema_version: "qa-v2", run_id: "live-failed-run", status: "scored" },
          metadata: { name: "Live failed run", dataset_name: "Approved live set" },
          capabilities: { retry_failed: true }, report_available: false,
          prepared_items: [{
            item_id: "live-capacity", question: "Current capacity?", time_sensitive: true,
            gold_atoms: [{ id: "capacity", text: "Capacity is seven", required: true }],
          }],
          answers: [],
          evaluation_results: [{
            item_id: "live-capacity", attempt_id: "live-capacity-attempt-1", ordinal: 1,
            status: "live_validation_failed",
            live_validation: { phase: "pre_run", reason: "answer_changed", detail: "Approved baseline changed." },
          }],
          summary: {
            overall_attempt_pass_rate: null, passed_attempts: 0, quality_accounted_attempts: 0,
            attempt_lifecycle_counts: { scored: 0, execution_failed: 0, evaluation_failed: 0, live_validation_failed: 1 },
          },
        },
      },
    }));

    await page.goto("/evaluations");
    await page.getByText("Live failed run", { exact: true }).click();

    await expect(page.getByText("No agent answer was created for this live-validation failure.")).toBeVisible();
    await expect(page.getByRole("button", { name: "Retry failed attempts (1)" })).toBeVisible();
    await expect(page.getByText("Live validation failed", { exact: true })).toBeVisible();
    await expect(page.locator("#run-detail-content").getByText("Live answer", { exact: true })).toBeVisible();
  });

  test("materializes, approves, validates, refreshes, gates, continues, and records a real qa-v2 live run", async ({ page }) => {
    test.setTimeout(90_000);
    await page.request.post("/api/evaluation-test/live-value", { data: { value: 7 } });
    await page.goto("/evaluations");
    await page.getByRole("button", { name: /Datasets/ }).click();
    await page.getByLabel("Name").first().fill("Production live definition");
    await page.getByLabel("Dataset file").setInputFiles({
      name: "production-live.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify({
        schema_version: "qa-dataset-v2",
        items: [
          {
            id: "static-policy", question: "What is the stable policy?", answer: "The policy is stable.",
            time_sensitive: false,
            expected_atoms: [{ id: "policy", text: "The policy is stable.", required: true }],
          },
          {
            id: "live-capacity", question: "What is current primary capacity?", time_sensitive: true,
            oracle: { kind: "mcp", calls: [{
              id: "capacity", server: "fixture", tool: "current_capacity",
              arguments: { service: "primary" }, answer_fields: { available: "/available" },
              metadata_fields: { revision: "/revision" },
            }] },
          },
        ],
      })),
    });
    await page.getByRole("button", { name: "Validate and import" }).click();
    const parent = page.locator("#dataset-list [data-dataset-id]").filter({
      has: page.getByText("Production live definition", { exact: true }),
    });
    await expect(parent).toBeVisible();
    await parent.click();
    await page.getByRole("button", { name: "Generate Atoms" }).click();

    const materializeDialog = page.getByRole("dialog", { name: "Review atoms · Production live definition" });
    await expect(materializeDialog).toBeVisible({ timeout: 20_000 });
    const materializedLivePanel = materializeDialog.locator(".atom-item").filter({ hasText: "live-capacity" });
    await materializedLivePanel.locator(":scope > summary").click();
    await expect(materializedLivePanel.getByText("Live answer", { exact: true })).toBeVisible();
    await expect(materializedLivePanel.locator(".expected-answer")).toContainText('"available": 7');
    await page.getByLabel("New dataset name").fill("Approved live seven");
    await page.getByRole("button", { name: "Save as new dataset" }).click();
    const approvedSeven = page.locator("#dataset-list [data-dataset-id]").filter({
      has: page.getByText("Approved live seven", { exact: true }),
    });
    await expect(approvedSeven).toBeVisible();
    await expect(approvedSeven).toContainText("includes live answers");

    await page.getByRole("button", { name: "Run evaluation" }).click();
    await page.getByLabel("Evaluation name").fill("Live seven baseline");
    await page.getByRole("button", { name: "Start evaluation" }).click();
    await expect(page.getByRole("heading", { name: "Live seven baseline" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText("Includes live answers", { exact: true })).toBeVisible();
    await expect(page.locator("#run-detail-content").getByText("Live answer", { exact: true })).toBeVisible();

    await page.request.post("/api/evaluation-test/live-value", { data: { value: 8 } });
    await page.getByRole("button", { name: /Datasets/ }).click();
    await approvedSeven.click();
    await page.getByRole("button", { name: "Refresh live snapshot" }).click();
    const refreshDialog = page.getByRole("dialog", { name: "Review atoms · Production live definition" });
    await expect(refreshDialog).toBeVisible({ timeout: 20_000 });
    const refreshedLivePanel = refreshDialog.locator(".atom-item").filter({ hasText: "live-capacity" });
    await refreshedLivePanel.locator(":scope > summary").click();
    await expect(refreshedLivePanel.getByText("changed", { exact: false })).toBeVisible();
    await expect(refreshedLivePanel.getByText("Previous approved answer")).toBeVisible();
    await expect(refreshedLivePanel.locator(".expected-answer")).toContainText('"available": 8');
    await page.getByLabel("New dataset name").fill("Approved live eight");
    await page.getByRole("button", { name: "Save as new dataset" }).click();
    await expect(page.getByText("Approved live eight", { exact: true }).first()).toBeVisible();

    await approvedSeven.click();
    await page.getByRole("button", { name: "Run evaluation" }).click();
    await page.getByLabel("Evaluation name").fill("Changed live gate");
    await page.getByRole("button", { name: "Start evaluation" }).click();
    await expect(page.getByRole("heading", { name: "Live answers need attention" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText("No agent attempts have started.", { exact: false }).first()).toBeVisible();
    await expect(page.locator("#attention-items").getByText("live-capacity", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Continue with valid questions" }).click();
    await expect(page.getByRole("heading", { name: "Changed live gate" })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText("No agent answer was created for this live-validation failure.")).toBeVisible();
    await expect(page.getByText("Live validation failed", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Retry failed attempts (1)" }).click();
    await expect(page.getByRole("heading", { name: "Changed live gate · retry 1" })).toBeVisible({ timeout: 30_000 });
    await expect(page.locator(".lineage-callout")).toContainText("retried from Changed live gate");
    await expect(page.getByText("Live validation failed", { exact: true })).toBeVisible();
  });

  test("aligns the start evaluation action to the launch card edge", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto("/evaluations");
    await page.getByRole("button", { name: /New evaluation/ }).click();

    const launchCard = page.locator(".launch-card");
    const startButton = page.getByRole("button", { name: "Start evaluation" });
    const desktopCardBox = await launchCard.boundingBox();
    const desktopButtonBox = await startButton.boundingBox();
    expect(desktopCardBox).not.toBeNull();
    expect(desktopButtonBox).not.toBeNull();
    const desktopRightInset = Math.round(
      desktopCardBox!.x + desktopCardBox!.width - desktopButtonBox!.x - desktopButtonBox!.width,
    );
    expect(desktopRightInset).toBeGreaterThanOrEqual(21);
    expect(desktopRightInset).toBeLessThanOrEqual(24);
    expect(desktopButtonBox!.x).toBeGreaterThan(desktopCardBox!.x + desktopCardBox!.width / 2);

    await page.setViewportSize({ width: 390, height: 844 });
    const mobileCardBox = await launchCard.boundingBox();
    const mobileButtonBox = await startButton.boundingBox();
    expect(mobileCardBox).not.toBeNull();
    expect(mobileButtonBox).not.toBeNull();
    const mobileLeftInset = Math.round(mobileButtonBox!.x - mobileCardBox!.x);
    const mobileRightInset = Math.round(
      mobileCardBox!.x + mobileCardBox!.width - mobileButtonBox!.x - mobileButtonBox!.width,
    );
    expect(mobileLeftInset).toBeGreaterThanOrEqual(21);
    expect(mobileLeftInset).toBeLessThanOrEqual(24);
    expect(mobileRightInset).toBe(mobileLeftInset);
  });

  test("cancels the active evaluation and records it in history", async ({ page }) => {
    let canceled = false;
    const activeJob = {
      id: "job-active",
      kind: "evaluation",
      status: "running",
      context: { name: "Mistaken run", workspace_id: "workspace-active", attempts: 1 },
    };
    const canceledRun = {
      id: "history-canceled",
      run_id: "workspace-active",
      name: "Mistaken run",
      status: "canceled",
      created_at: "2026-08-12T10:00:00+00:00",
      dataset_name: "Reviewed set",
      agent_spec: "agent.md",
      attempts: 1,
      overall_attempt_pass_rate: null,
      valid: true,
    };
    await page.route("**/api/evaluations/catalog", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          datasets: [], profiles: [], agents: [],
          jobs: canceled ? [{ ...activeJob, status: "canceled" }] : [activeJob],
        }),
      });
    });
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ runs: canceled ? [canceledRun] : [] }),
      });
    });
    await page.route("**/api/evaluations/jobs/job-active", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ job: { ...activeJob, status: canceled ? "canceled" : "running" } }),
      });
    });
    await page.route("**/api/evaluations/jobs/job-active/cancel", async (route) => {
      canceled = true;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          job: { ...activeJob, status: "canceled", completed_at: "2026-08-12T10:01:00+00:00" },
          history_id: "history-canceled",
        }),
      });
    });

    await page.goto("/evaluations");
    const cancelButton = page.getByRole("button", { name: "Cancel evaluation" });
    await expect(cancelButton).toBeVisible();
    page.once("dialog", async (dialog) => {
      expect(dialog.message()).toContain("requests already accepted by a remote provider");
      await dialog.accept();
    });
    await cancelButton.click();

    await expect(page.locator("#job-banner")).toBeHidden();
    await expect(page.locator("#runs-body")).toContainText("Mistaken run");
    await expect(page.locator("#runs-body .status.canceled")).toHaveText("canceled");
    await expect(page.locator("#toast")).toContainText("Evaluation canceled");
  });

  test("reloads the run table and every trend from one bounded history window", async ({ page }) => {
    const requestedRanges: string[] = [];
    let releaseYear!: () => void;
    const yearGate = new Promise<void>((resolve) => { releaseYear = resolve; });
    const yearRun = {
      id: "year-run",
      run_id: "year-run",
      name: "Year run",
      status: "scored",
      created_at: "2026-01-15T10:00:00+00:00",
      dataset_key: "dataset-year",
      dataset_name: "Year dataset",
      attempts: 1,
      overall_attempt_pass_rate: 1,
      passed_attempts: 1,
      quality_accounted_attempts: 1,
      attempt_lifecycle_counts: {
        scored: 1,
        execution_failed: 0,
        evaluation_failed: 0,
      },
      technical_failure_rate: 0,
      latency: {
        total_attempts: 1,
        timed_attempts: 1,
        average_ms: 120,
        best_ms: 120,
        worst_ms: 120,
      },
      valid: true,
    };
    await page.route("**/api/evaluations/runs?*", async (route) => {
      const historyRange = new URL(route.request().url()).searchParams.get("range") || "";
      requestedRanges.push(historyRange);
      if (historyRange === "365d") await yearGate;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ runs: historyRange === "365d" ? [yearRun] : [] }),
      });
    });

    await page.goto("/evaluations");

    const historyWindow = page.getByLabel("History window");
    await expect(historyWindow).toHaveValue("90d");
    await expect(historyWindow.locator("option")).toHaveText([
      "Last 7 days",
      "Last 30 days",
      "Last 90 days",
      "Last 365 days",
    ]);
    await expect.poll(() => requestedRanges).toEqual(["90d"]);
    await expect(page.locator("#runs-body tr")).toHaveCount(0);

    await historyWindow.selectOption("365d");
    await expect(historyWindow).toBeDisabled();
    await expect(page.locator(".history-trends")).toHaveAttribute("aria-busy", "true");
    releaseYear();

    await expect(historyWindow).toBeEnabled();
    await expect(page.locator(".history-trends")).not.toHaveAttribute("aria-busy", "true");
    await expect.poll(() => requestedRanges).toEqual(["90d", "365d"]);
    await expect(page.locator("#runs-body tr")).toHaveCount(1);
    await expect(page.locator('#trend-latency-chart [data-series="average"]')).toHaveCount(1);
    await expect(page.locator('#trend-pass-chart [data-series="pass"]')).toHaveCount(1);
    await expect(page.locator('#trend-failure-chart [data-series="failure"]')).toHaveCount(1);
    await expect(historyWindow.locator('option[value="all"]')).toHaveCount(0);
  });

  test("uses the Archi page header and selected dark theme", async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem("archi_theme", "dark"));
    await page.goto("/evaluations");

    const header = page.locator(".evaluation-header");
    const backToChat = page.getByRole("link", { name: "Back to Chat" });
    await expect(header).toBeVisible();
    await expect(header.getByRole("heading", { name: "Evaluation Console" })).toBeVisible();
    await expect(backToChat).toBeVisible();
    await expect(backToChat).toHaveAttribute("href", "/chat");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

    const colors = await page.evaluate(() => ({
      page: getComputedStyle(document.body).backgroundColor,
      header: getComputedStyle(document.querySelector(".evaluation-header")!).backgroundColor,
      panel: getComputedStyle(document.querySelector(".panel")!).backgroundColor,
      text: getComputedStyle(document.body).color,
    }));
    expect(colors).toEqual({
      page: "rgb(15, 18, 23)",
      header: "rgb(20, 25, 35)",
      panel: "rgb(20, 25, 35)",
      text: "rgb(229, 231, 235)",
    });

    const backBox = await backToChat.boundingBox();
    const headerBox = await header.boundingBox();
    const sidebarBox = await page.locator(".sidebar").boundingBox();
    expect(backBox).not.toBeNull();
    expect(headerBox).not.toBeNull();
    expect(sidebarBox).not.toBeNull();
    expect(backBox!.x).toBeLessThan(40);
    expect(backBox!.y).toBeGreaterThanOrEqual(headerBox!.y);
    expect(backBox!.y + backBox!.height).toBeLessThanOrEqual(headerBox!.y + headerBox!.height);
    expect(sidebarBox!.y).toBe(headerBox!.y + headerBox!.height);
  });

  test("previews launch selections in equal-height cards and explains worker memory", async ({ page }) => {
    const dataset = {
      id: "dataset-launch",
      name: "Launch dataset",
      source_filename: "launch.json",
      sha256: "1".repeat(64),
      item_count: 4,
      eligible_item_count: 4,
      time_sensitive_item_count: 0,
      supplied_atom_item_count: 4,
      atom_count: 4,
      categories: ["operations"],
      answer_sources: ["handbook"],
      created_at: "2026-08-05T10:00:00+00:00",
      parent_dataset_id: null,
    };
    const profiles = [
      {
        id: "builtin",
        name: "Built-in QA profile",
        sha256: null,
        built_in: true,
        components: {
          atoms_extractor: { provider: "openai", model: "extractor-default" },
          evaluator: { provider: "openai", model: "judge-default" },
        },
      },
      {
        id: "profile-local",
        name: "Local deterministic judge",
        sha256: "2".repeat(64),
        built_in: false,
        components: {
          atoms_extractor: { provider: "ollama", model: "extractor-local", timeout: 30 },
          evaluator: { provider: "ollama", model: "judge-local", timeout: 45 },
        },
      },
    ];
    const agentContents: Record<string, { name: string; tools: string[]; content: string }> = {
      "alpha.md": {
        name: "Alpha operator",
        tools: ["search", "lookup"],
        content: "---\nname: Alpha operator\ntools:\n  - search\n  - lookup\n---\nAnswer only from retrieved evidence.",
      },
      "beta.md": {
        name: "Beta investigator",
        tools: ["search"],
        content: "---\nname: Beta investigator\ntools:\n  - search\n---\nCompare sources before answering.",
      },
    };
    let catalogAgents = [
      { id: "alpha.md", name: "alpha" },
      { id: "beta.md", name: "beta" },
      { id: "broken.md", name: "broken" },
    ];
    let releaseBeta!: () => void;
    const betaGate = new Promise<void>((resolve) => { releaseBeta = resolve; });
    const requestedAgents: string[] = [];
    await page.route("**/api/evaluations/catalog", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          datasets: [dataset],
          profiles,
          agents: catalogAgents,
          jobs: [],
        }),
      });
    });
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({ contentType: "application/json", body: JSON.stringify({ runs: [] }) });
    });
    await page.route("**/api/evaluations/agents/*", async (route) => {
      const id = decodeURIComponent(new URL(route.request().url()).pathname.split("/").pop()!);
      requestedAgents.push(id);
      const agent = agentContents[id];
      if (!agent) {
        await route.fulfill({
          status: 400,
          contentType: "application/json",
          body: JSON.stringify({ error: "selected agent spec does not exist" }),
        });
        return;
      }
      if (id === "beta.md") await betaGate;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          agent: { id, ...agent },
        }),
      });
    });

    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto("/evaluations");
    await page.getByRole("button", { name: /New evaluation/ }).click();

    const profileSnapshot = page.locator("#profile-snapshot");
    const agentSnapshot = page.locator("#agent-snapshot");
    await expect(profileSnapshot.getByRole("heading", { name: "Built-in QA profile" })).toBeVisible();
    await expect(profileSnapshot).toContainText("openai / extractor-default");
    await expect(profileSnapshot).toContainText("openai / judge-default");
    await expect(agentSnapshot.getByRole("heading", { name: "Alpha operator" })).toBeVisible();
    await expect(agentSnapshot.getByLabel("Selected specification contents")).toContainText(
      "Answer only from retrieved evidence.",
    );

    await page.getByLabel("Evaluator profile").selectOption("profile-local");
    await expect(profileSnapshot.getByRole("heading", { name: "Local deterministic judge" })).toBeVisible();
    await expect(profileSnapshot).toContainText("ollama / judge-local");
    await expect(profileSnapshot).toContainText("45s timeout");

    await page.getByLabel("Agent spec").selectOption("beta.md");
    await expect(agentSnapshot.getByRole("status")).toHaveText("Loading selected agent spec…");
    releaseBeta();
    await expect(agentSnapshot.getByRole("heading", { name: "Beta investigator" })).toBeVisible();
    await expect(agentSnapshot).toContainText("Compare sources before answering.");
    expect(requestedAgents).toEqual(["alpha.md", "beta.md"]);

    const cardHeights = await page.locator(".configuration-grid > .form-card").evaluateAll((cards) => (
      cards.map((card) => Math.round(card.getBoundingClientRect().height))
    ));
    expect(new Set(cardHeights).size).toBe(1);

    const runInfo = page.getByRole("button", { name: "Run phase concurrency and memory help" });
    const runTooltip = page.locator("#run-workers-tooltip");
    await runInfo.focus();
    await expect(runTooltip).toBeVisible();
    await expect(runTooltip).toContainText("isolated agent runtime");
    await expect(runTooltip).toContainText("memory generally grows roughly in proportion");
    await expect(runTooltip).toContainText("never overlap");
    const runTooltipBox = await runTooltip.boundingBox();
    const runInputBox = await page.getByLabel("Run workers").boundingBox();
    expect(runTooltipBox).not.toBeNull();
    expect(runInputBox).not.toBeNull();
    expect(runTooltipBox!.y + runTooltipBox!.height).toBeLessThanOrEqual(runInputBox!.y);

    const scoreInfo = page.getByRole("button", { name: "Evaluation phase concurrency and memory help" });
    await scoreInfo.focus();
    const scoreTooltip = page.locator("#score-workers-tooltip");
    await expect(scoreTooltip).toBeVisible();
    await expect(scoreTooltip).toContainText("isolated evaluator runtime");
    await expect(scoreTooltip).toContainText("not active during the run phase");

    await page.getByLabel("Agent spec").selectOption("broken.md");
    await expect(agentSnapshot.getByRole("status")).toContainText("Could not load this agent spec.");
    await expect(page.getByLabel("Agent spec")).toHaveValue("broken.md");

    await page.setViewportSize({ width: 390, height: 844 });
    await runInfo.focus();
    await expect(runTooltip).toBeVisible();
    const mobileTooltipBox = await runTooltip.boundingBox();
    expect(mobileTooltipBox).not.toBeNull();
    expect(mobileTooltipBox!.x).toBeGreaterThanOrEqual(0);
    expect(mobileTooltipBox!.x + mobileTooltipBox!.width).toBeLessThanOrEqual(390);
    const hasPageOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    );
    expect(hasPageOverflow).toBe(false);

    catalogAgents = [];
    await page.reload();
    await page.getByRole("button", { name: /New evaluation/ }).click();
    await expect(page.locator("#agent-snapshot").getByRole("status")).toHaveText(
      "No agent spec is available.",
    );
  });

  test("filters history trends by dataset and opens points with keyboard", async ({ page }) => {
    const runs = [
      {
        id: "trend-alpha",
        run_id: "trend-alpha",
        name: "Alpha baseline",
        status: "scored",
        created_at: "2026-08-01T10:00:00+00:00",
        dataset_id: "dataset-alpha",
        dataset_key: "dataset-alpha",
        dataset_name: "Alpha set",
        attempts: 4,
        retry_of_history_id: null,
        retry_number: null,
        overall_attempt_pass_rate: 2 / 3,
        passed_attempts: 2,
        quality_accounted_attempts: 3,
        attempt_lifecycle_counts: {
          scored: 2,
          execution_failed: 1,
          evaluation_failed: 1,
        },
        technical_failure_rate: 0.5,
        latency: {
          total_attempts: 4,
          timed_attempts: 3,
          average_ms: 200,
          best_ms: 100,
          worst_ms: 300,
        },
        valid: true,
      },
      {
        id: "trend-alpha-retry",
        run_id: "trend-alpha-retry",
        name: "Alpha retry",
        status: "scored",
        created_at: "2026-08-02T10:00:00+00:00",
        dataset_id: "dataset-alpha",
        dataset_key: "dataset-alpha",
        dataset_name: "Alpha set",
        attempts: 4,
        retry_of_history_id: "trend-alpha",
        retry_number: 1,
        overall_attempt_pass_rate: 2 / 3,
        passed_attempts: 2,
        quality_accounted_attempts: 3,
        attempt_lifecycle_counts: {
          scored: 2,
          execution_failed: 1,
          evaluation_failed: 1,
        },
        technical_failure_rate: 0.5,
        latency: {
          total_attempts: 4,
          timed_attempts: 3,
          average_ms: 600,
          best_ms: 250,
          worst_ms: 900,
        },
        valid: true,
      },
      {
        id: "trend-beta",
        run_id: "trend-beta",
        name: "Beta comparison",
        status: "scored",
        created_at: "2026-08-03T10:00:00+00:00",
        dataset_id: "dataset-beta",
        dataset_key: "dataset-beta",
        dataset_name: "Beta set",
        attempts: 2,
        retry_of_history_id: null,
        retry_number: null,
        overall_attempt_pass_rate: 0.5,
        passed_attempts: 1,
        quality_accounted_attempts: 2,
        attempt_lifecycle_counts: {
          scored: 2,
          execution_failed: 0,
          evaluation_failed: 0,
        },
        technical_failure_rate: 0,
        latency: {
          total_attempts: 2,
          timed_attempts: 2,
          average_ms: 450,
          best_ms: 400,
          worst_ms: 500,
        },
        valid: true,
      },
      {
        id: "trend-prepared",
        run_id: "trend-prepared",
        name: "Evaluation in progress",
        status: "prepared",
        created_at: "2026-08-04T10:00:00+00:00",
        dataset_id: null,
        dataset_key: "snapshot:prepared",
        dataset_name: "source.json",
        attempts: null,
        retry_of_history_id: null,
        retry_number: null,
        overall_attempt_pass_rate: null,
        passed_attempts: null,
        quality_accounted_attempts: null,
        attempt_lifecycle_counts: null,
        technical_failure_rate: null,
        latency: null,
        valid: true,
      },
      {
        id: "trend-run-completed",
        run_id: "trend-run-completed",
        name: "Execution complete, scoring pending",
        status: "run_completed",
        created_at: "2026-08-04T11:00:00+00:00",
        dataset_id: null,
        dataset_key: "dataset-alpha",
        dataset_name: "source.json",
        attempts: 1,
        retry_of_history_id: null,
        retry_number: null,
        overall_attempt_pass_rate: null,
        passed_attempts: null,
        quality_accounted_attempts: null,
        attempt_lifecycle_counts: null,
        technical_failure_rate: null,
        latency: {
          total_attempts: 1,
          timed_attempts: 1,
          average_ms: 700,
          best_ms: 700,
          worst_ms: 700,
        },
        valid: true,
      },
    ];
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ runs }) });
    });
    await page.route("**/api/evaluations/runs/trend-alpha", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          run: {
            id: "trend-alpha",
            metadata: { name: "Alpha baseline", dataset_name: "Alpha set" },
            manifest: { run_id: "trend-alpha", schema_version: "qa-v1", status: "scored" },
            summary: {
              overall_attempt_pass_rate: 2 / 3,
              passed_attempts: 2,
              quality_accounted_attempts: 3,
              attempt_lifecycle_counts: { scored: 2, execution_failed: 1, evaluation_failed: 1 },
            },
            prepared_items: [],
            answers: [],
            evaluation_results: [],
            capabilities: { retry_failed: true },
            report_available: false,
          },
        }),
      });
    });

    await page.goto("/evaluations");

    await expect(page.getByRole("heading", { name: "History trends" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Attempt latency" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Pass rate" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Technical failure rate" })).toBeVisible();
    await expect(page.getByText("Live-validation failures are excluded.", { exact: false })).toBeVisible();
    const alpha = page.getByRole("checkbox", { name: "Alpha set" });
    const beta = page.getByRole("checkbox", { name: "Beta set" });
    await expect(alpha).toBeChecked();
    await expect(beta).toBeChecked();
    await expect(page.getByRole("checkbox", { name: "source.json" })).toHaveCount(0);
    await expect(page.locator("#trend-filter-status")).toHaveText("Showing 2 of 2 datasets");
    await expect(page.locator('#trend-latency-chart [data-series="average"]')).toHaveCount(3);
    await expect(page.locator('#trend-pass-chart [data-series="pass"]')).toHaveCount(3);
    await expect(page.locator('#trend-failure-chart [data-series="failure"]')).toHaveCount(3);
    await expect(
      page.locator("#trend-latency-chart .trend-line.trend-average").first(),
    ).toHaveCSS("stroke-dasharray", "none");

    const alphaAverage = page.getByRole("link", {
      name: /Alpha baseline.*Alpha set.*Average latency 200 ms/,
    });
    await alphaAverage.hover();
    const tooltip = page.locator("#trend-latency-tooltip");
    await expect(tooltip).toBeVisible();
    await expect(tooltip).toContainText("Alpha set");
    await alphaAverage.focus();
    await expect(tooltip).toBeVisible();
    await expect(tooltip).toContainText("Alpha set");
    await expect(tooltip).toContainText("3 of 4 attempts timed");

    const retryPass = page.getByRole("link", {
      name: /Alpha retry.*Alpha set.*Pass rate 66.7%/,
    });
    await retryPass.focus();
    await expect(page.locator("#trend-pass-tooltip")).toContainText(
      "2 of 3 quality-accounted attempts passed",
    );
    const retryFailure = page.getByRole("link", {
      name: /Alpha retry.*Alpha set.*Technical failure rate 50.0%/,
    });
    await retryFailure.focus();
    const failureTooltip = page.locator("#trend-failure-tooltip");
    await expect(failureTooltip).toContainText("2 of 4 terminal attempts failed technically");
    await expect(failureTooltip).toContainText("1 execution · 1 evaluation failures");
    await expect(failureTooltip).toContainText("Retry of Alpha baseline");

    const alphaPass = page.getByRole("link", {
      name: /Alpha baseline.*Alpha set.*Pass rate 66.7%/,
    });
    await alphaPass.focus();
    await expect(page.locator("#trend-pass-tooltip")).toContainText(
      "2 of 3 quality-accounted attempts passed",
    );
    await alphaPass.press("Enter");
    await expect(page.getByRole("heading", { name: "Alpha baseline" })).toBeVisible();
    await page.getByRole("button", { name: "← All runs" }).click();

    await beta.uncheck();
    await expect(page.locator('#trend-latency-chart [data-series="average"]')).toHaveCount(2);
    await expect(page.locator('#trend-pass-chart [data-series="pass"]')).toHaveCount(2);
    await expect(page.locator('#trend-failure-chart [data-series="failure"]')).toHaveCount(2);
    await alpha.click();
    await expect(alpha).toBeChecked();
    await page.getByRole("button", { name: "Show all datasets" }).click();
    await expect(beta).toBeChecked();

    await page.setViewportSize({ width: 390, height: 844 });
    const hasPageOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    );
    expect(hasPageOverflow).toBe(false);

    await alphaAverage.focus();
    await alphaAverage.press("Space");
    await expect(page.getByRole("heading", { name: "Alpha baseline" })).toBeVisible();
    await expect(page.getByText("Alpha set · resolved agent")).toBeVisible();
  });

  test("keeps missing metric samples as visible gaps", async ({ page }) => {
    const runs = [
      { id: "gap-before", name: "Before gap", created_at: "2026-08-01T10:00:00+00:00", latency: { total_attempts: 1, timed_attempts: 1, average_ms: 100, best_ms: 100, worst_ms: 100 } },
      { id: "gap-missing", name: "Missing latency", created_at: "2026-08-02T10:00:00+00:00", latency: null },
      { id: "gap-after", name: "After gap", created_at: "2026-08-03T10:00:00+00:00", latency: { total_attempts: 1, timed_attempts: 1, average_ms: 300, best_ms: 300, worst_ms: 300 } },
    ].map((run) => ({
      ...run,
      run_id: run.id,
      status: "scored",
      dataset_key: "gap-dataset",
      dataset_name: "Gap dataset",
      overall_attempt_pass_rate: 1,
      passed_attempts: 1,
      quality_accounted_attempts: 1,
      attempt_lifecycle_counts: { scored: 1, execution_failed: 0, evaluation_failed: 0 },
      technical_failure_rate: 0,
      valid: true,
    }));
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ runs }),
      });
    });

    await page.goto("/evaluations");

    await expect(page.locator('#trend-latency-chart [data-series="average"]')).toHaveCount(2);
    await expect(page.locator("#trend-latency-chart path.trend-average")).toHaveCount(0);
    await expect(page.locator("#trend-latency-chart .trend-summary")).toHaveText(
      "2 runs plotted · 1 without this metric",
    );
    await expect(page.getByRole("group", { name: "2 runs plotted; 1 without this metric" })).toBeVisible();
    await expect(page.getByRole("link", { name: /Before gap.*Average latency 100 ms/ })).toBeVisible();
  });

  test("does not synthesize unavailable trends and honors reduced motion", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          runs: [
            {
              id: "legacy-dated",
              run_id: "legacy-dated",
              name: "Legacy dated run",
              status: "scored",
              created_at: "2026-07-01T10:00:00+00:00",
              dataset_key: "legacy-set",
              dataset_name: "Legacy set",
              overall_attempt_pass_rate: null,
              technical_failure_rate: null,
              attempt_lifecycle_counts: null,
              latency: null,
              valid: true,
            },
            {
              id: "missing-date",
              run_id: "missing-date",
              name: "Missing date",
              status: "scored",
              created_at: null,
              dataset_key: "legacy-set",
              dataset_name: "Legacy set",
              overall_attempt_pass_rate: 1,
              passed_attempts: 1,
              quality_accounted_attempts: 1,
              attempt_lifecycle_counts: {
                scored: 1,
                execution_failed: 0,
                evaluation_failed: 0,
              },
              technical_failure_rate: 0,
              latency: {
                total_attempts: 1,
                timed_attempts: 1,
                average_ms: 10,
                best_ms: 10,
                worst_ms: 10,
              },
              valid: true,
            },
          ],
        }),
      });
    });

    await page.goto("/evaluations");

    await expect(page.locator("#trend-latency-chart")).toContainText("Trend unavailable");
    await expect(page.locator("#trend-pass-chart")).toContainText("Trend unavailable");
    await expect(page.locator("#trend-failure-chart")).toContainText("Trend unavailable");
    for (const id of ["trend-latency-chart", "trend-pass-chart", "trend-failure-chart"]) {
      await expect(page.locator(`#${id}`)).toContainText("1 without this metric");
      await expect(page.locator(`#${id}`)).toContainText("1 without a timestamp");
    }
    await expect(page.locator(".trend-point")).toHaveCount(0);

    const filterMotion = await page.locator(".trend-dataset-option").evaluate((filter) => ({
      transitionDuration: getComputedStyle(filter).transitionDuration,
      transform: getComputedStyle(filter).transform,
    }));
    expect(filterMotion).toEqual({ transitionDuration: "0s", transform: "none" });
  });

  test("shows explicit graph errors when compact history cannot load", async ({ page }) => {
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ error: "History storage is unavailable." }),
      });
    });

    await page.goto("/evaluations");

    await expect(page.locator("#trend-filter-status")).toHaveText(
      "Trend history could not be loaded.",
    );
    for (const id of ["trend-latency-chart", "trend-pass-chart", "trend-failure-chart"]) {
      await expect(page.locator(`#${id}`)).toContainText("Could not load trends.");
      await expect(page.locator(`#${id}`)).toContainText("History storage is unavailable.");
    }
    await expect(page.locator(".trend-point")).toHaveCount(0);
  });

  test("imports, reviews atoms, saves a child dataset, launches, and opens history", async ({ page }) => {
    await page.goto("/evaluations");

    await page.getByRole("button", { name: /Datasets/ }).click();
    await page.getByLabel("Name").first().fill("Mock dataset");
    await page.getByLabel("Dataset file").setInputFiles({
      name: "mock-dataset.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify([
        {
          id: "mock-entry",
          question: "What value does the mock entry contain?",
          answer: "The mock entry contains 42.",
          time_sensitive: false,
          category: "test-fixture",
          answer_source: "mocked-entry",
        },
      ])),
    });
    await page.getByRole("button", { name: "Validate and import" }).click();
    await expect(page.getByText("Mock dataset", { exact: true }).first()).toBeVisible();
    const parentDataset = page.locator("#dataset-list [data-dataset-id]").filter({
      has: page.getByText("Mock dataset", { exact: true }),
    });
    const parentDatasetId = await parentDataset.getAttribute("data-dataset-id");
    expect(parentDatasetId).not.toBeNull();
    await parentDataset.click();

    await expect(page.getByLabel("Atom generation profile")).toBeVisible();
    await page.getByRole("button", { name: "Generate Atoms" }).click();
    await expect(page.getByRole("dialog")).toBeVisible({ timeout: 15_000 });
    await page.getByRole("button", { name: "Close" }).click();
    await page.reload();
    await page.getByRole("button", { name: /Datasets/ }).click();
    await page.locator(`[data-dataset-id="${parentDatasetId}"]`).click();
    await page.getByRole("button", { name: "Resume atom review" }).click();
    await expect(page.getByRole("dialog", { name: "Review atoms · Mock dataset" })).toBeVisible();
    const questionPanels = page.locator("#atom-editor details");
    await expect(questionPanels).toHaveCount(1);
    const firstPanel = questionPanels.first();
    await expect(firstPanel).toHaveAttribute("open", "");
    await expect(firstPanel.locator("summary")).toContainText("What value does the mock entry contain?");
    await expect(firstPanel.getByText("Expected answer", { exact: true })).toBeVisible();
    await expect(firstPanel.getByText("Atoms", { exact: true })).toBeVisible();
    await expect(firstPanel.getByText("1 atom · 1 required", { exact: true })).toBeVisible();
    const firstAtomText = firstPanel.getByLabel(/Atom text 1 for/);
    const requiredToggle = firstPanel.getByLabel(/Required for atom 1 in/);
    await expect(firstAtomText).toBeVisible();
    await expect(firstPanel.locator(".expected-answer")).toContainText("The mock entry contains 42.");
    await expect(firstAtomText).toHaveValue("The mock entry contains 42.");
    expect(await questionPanels.evaluateAll((panels) => panels.every((panel) => {
      const question = panel.querySelector(".atom-summary-copy strong")?.textContent?.trim();
      const answer = panel.querySelector(".expected-answer > div")?.textContent?.trim();
      const atoms = [...panel.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("[data-field=text]")];
      return Boolean(question && answer && atoms.length && atoms.every((atom) => atom.value.trim()));
    }))).toBe(true);
    expect(await firstPanel.evaluate((panel) => {
      const atom = panel.querySelector("[data-atom-row]");
      return atom !== null && atom.getBoundingClientRect().bottom <= panel.getBoundingClientRect().bottom;
    })).toBe(true);
    const reviewedDatasetName = page.getByLabel("New dataset name");
    await reviewedDatasetName.fill("   ");
    await page.getByRole("button", { name: "Save as new dataset" }).click();
    await expect(page.getByText("Enter a name for the reviewed dataset.", { exact: true })).toBeVisible();
    await expect(reviewedDatasetName).toBeFocused();
    await reviewedDatasetName.fill("Mock dataset · reviewed");
    await expect(page.getByText("Enter a name for the reviewed dataset.", { exact: true })).toHaveCount(0);
    const firstAtomId = firstPanel.getByLabel(/Atom ID 1 for/);
    await firstAtomId.fill("A1-reviewed");
    await firstAtomId.press("Enter");
    await expect(page.getByRole("dialog", { name: "Review atoms · Mock dataset" })).toBeVisible();
    await expect(firstAtomId).toHaveValue("A1-reviewed");
    await expect(requiredToggle).toBeChecked();
    await requiredToggle.uncheck();
    await expect(requiredToggle).not.toBeChecked();
    await expect(firstPanel.getByText("1 atom · 0 required", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Save as new dataset" }).click();
    await expect(firstPanel.getByRole("alert")).toHaveText("At least one atom must be required.");
    await expect(requiredToggle).toBeFocused();
    await requiredToggle.check();
    await expect(firstPanel.getByRole("alert")).toHaveCount(0);
    await expect(firstPanel.getByText("1 atom · 1 required", { exact: true })).toBeVisible();
    await firstPanel.getByRole("button", { name: /Add atom to/ }).click();
    const addedAtomText = firstPanel.getByLabel(/Atom text 2 for/);
    const addedAtomId = firstPanel.getByLabel(/Atom ID 2 for/);
    const addedRequired = firstPanel.getByLabel(/Required for atom 2 in/);
    await expect(addedAtomText).toBeFocused();
    await addedAtomId.fill("A1-reviewed");
    await requiredToggle.uncheck();
    await addedRequired.uncheck();
    await page.getByRole("button", { name: "Save as new dataset" }).click();
    await expect(firstPanel.getByRole("alert")).toContainText("Atom IDs must be unique within this question.");
    await expect(firstPanel.getByRole("alert")).toContainText("Every atom needs text.");
    await expect(firstPanel.getByRole("alert")).toContainText("At least one atom must be required.");
    await addedAtomText.fill("Optional supporting obligation");
    await expect(firstPanel.getByRole("alert")).not.toContainText("Every atom needs text.");
    await expect(firstPanel.getByRole("alert")).toContainText("Atom IDs must be unique within this question.");
    await expect(firstPanel.getByRole("alert")).toContainText("At least one atom must be required.");
    await firstPanel.getByRole("button", { name: /Remove atom 2 from/ }).click();
    await expect(firstPanel.getByRole("alert")).toHaveText("At least one atom must be required.");
    await requiredToggle.check();
    await expect(firstPanel.getByRole("alert")).toHaveCount(0);
    await firstPanel.getByRole("button", { name: /Add atom to/ }).click();
    await firstPanel.getByLabel(/Atom text 2 for/).fill("Optional supporting obligation");
    await firstAtomText.fill("Human-reviewed golden obligation");
    await page.getByRole("button", { name: "Save as new dataset" }).click();
    await expect(page.locator("#dataset-detail").getByRole("heading", {
      name: "Mock dataset · reviewed",
    })).toBeVisible();

    await page.locator(`[data-dataset-id="${parentDatasetId}"]`).click();
    await expect(page.getByRole("button", { name: "Resume atom review" })).toHaveCount(0);
    await page.locator("#dataset-list [data-dataset-id]").filter({
      has: page.getByText("Mock dataset · reviewed", { exact: true }),
    }).first().click();
    await expect(page.getByRole("button", { name: "Review Atoms" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Generate Atoms" })).toHaveCount(0);
    await expect(page.getByLabel("Atom generation profile")).toHaveCount(0);
    await page.getByRole("button", { name: "Evaluate" }).click();
    await page.getByLabel("Evaluation name").fill("Browser reviewed run");
    const runWorkers = page.getByLabel("Run workers");
    const scoreWorkers = page.getByLabel("Evaluation workers");
    for (const workers of [runWorkers, scoreWorkers]) {
      await expect(workers).toHaveValue("1");
      await expect(workers).toHaveAttribute("min", "1");
      await expect(workers).toHaveAttribute("max", "16");
      await expect(workers).toHaveAttribute("required", "");
    }
    await runWorkers.fill("17");
    expect(await runWorkers.evaluate((input: HTMLInputElement) => input.checkValidity())).toBe(false);
    await runWorkers.fill("2");
    await scoreWorkers.fill("3");
    const launchRequest = page.waitForRequest((request) => (
      request.url().endsWith("/api/evaluations/runs") && request.method() === "POST"
    ));
    await page.getByRole("button", { name: "Start evaluation" }).click();
    expect((await launchRequest).postDataJSON()).toMatchObject({
      run_workers: 2,
      score_workers: 3,
    });
    await expect(page.getByRole("heading", { name: "Browser reviewed run" })).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(".evidence-card").getByText("100.0%", { exact: true }).first()).toBeVisible();

    await page.getByRole("button", { name: "← All runs" }).click();
    await expect(page.locator("#runs-body").getByText("Browser reviewed run", { exact: true })).toBeVisible();
  });

  test("reviews a partially atomized dataset without generating missing atoms", async ({ page }) => {
    const requests: string[] = [];
    page.on("request", (request) => requests.push(request.url()));
    await page.goto("/evaluations");
    await page.getByRole("button", { name: /Datasets/ }).click();
    await page.getByLabel("Name").first().fill("Partially atomized");
    await page.getByLabel("Dataset file").setInputFiles({
      name: "partial.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify([
        {
          id: "with-atom",
          question: "Which row already has an atom?",
          answer: "This one.",
          time_sensitive: false,
          expected_atoms: [
            { id: "A1", text: "This row already has an atom.", required: true },
          ],
        },
        {
          id: "without-atom",
          question: "Which row needs atoms?",
          answer: "This other one.",
          answer_source: "curated knowledge",
          time_sensitive: false,
        },
      ])),
    });
    await page.getByRole("button", { name: "Validate and import" }).click();
    await expect(page.getByText("Partially atomized", { exact: true }).first()).toBeVisible();
    await page.locator("#dataset-list [data-dataset-id]").filter({
      has: page.getByText("Partially atomized", { exact: true }),
    }).click();

    await expect(page.getByRole("button", { name: "Review Atoms" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Generate Atoms" })).toHaveCount(0);
    await expect(page.getByLabel("Atom generation profile")).toHaveCount(0);
    const reviewButton = page.getByRole("button", { name: "Review Atoms" });
    await reviewButton.click();

    const dialog = page.getByRole("dialog", { name: "Review atoms · Partially atomized" });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("button", { name: "Close" })).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(dialog).not.toBeVisible();
    await expect(reviewButton).toBeFocused();
    await reviewButton.click();
    await expect(dialog).toBeVisible();
    const panels = dialog.locator("#atom-editor details");
    await expect(panels).toHaveCount(2);
    await expect(panels.first().getByLabel("Atom text 1 for with-atom")).toHaveValue(
      "This row already has an atom.",
    );
    await panels.nth(1).locator("summary").click();
    await expect(panels.nth(1).getByText("Source: curated knowledge")).toBeVisible();
    await expect(panels.nth(1).getByText("No atoms yet for this answer.")).toBeVisible();
    await expect(panels.nth(1).locator("[data-atom-row]")).toHaveCount(0);
    await panels.nth(1).getByRole("button", { name: "Add atom to without-atom" }).click();
    await expect(panels.nth(1).getByLabel("Atom text 1 for without-atom")).toHaveValue("");
    expect(requests.some((url) => url.endsWith("/generate-atoms"))).toBe(false);
    expect(requests.some((url) => url.endsWith("/review-atoms"))).toBe(true);
  });

  test("groups result evidence by question with readable scores, answers, and atom judgments", async ({ page }) => {
    const historyId = "human-readable-results";
    const question = "How does Archi preserve evidence across evaluation attempts?";
    const longToolResponse = `Complete response: ${"evidence ".repeat(300)}`;
    const atoms = [
      { id: "A1", text: "The run stores immutable input snapshots.", required: true },
      { id: "A2", text: "Each attempt retains its full model answer.", required: true },
      { id: "A3", text: "Evaluator judgments remain inspectable per atom.", required: false },
    ];
    const results = [
      {
        item_id: "q-human",
        attempt_id: "q-human-attempt-1",
        ordinal: 1,
        status: "scored",
        passed: true,
        atom_score: 1,
        judgments: atoms.map((atom) => ({
          atom_id: atom.id,
          outcome: "entailed",
          rationale: "The first response contains this expected content.",
        })),
      },
      {
        item_id: "q-human",
        attempt_id: "q-human-attempt-2",
        ordinal: 2,
        status: "scored",
        passed: false,
        atom_score: 0.25,
        judgments: [
          {
            atom_id: "A1",
            outcome: "entailed",
            rationale: "The immutable snapshot is described.",
          },
          {
            atom_id: "A2",
            outcome: "not_mentioned",
            rationale: "The answer does not mention preserving the full model response.",
          },
          {
            atom_id: "A3",
            outcome: "contradicted",
            rationale: "The answer incorrectly says atom judgments are discarded.",
          },
        ],
      },
    ];

    await page.route(`**/api/evaluations/runs/${historyId}`, async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          run: {
            manifest: {
              run_id: "readable-run",
              schema_version: "qa-v1",
              status: "scored",
            },
            metadata: {
              name: "Readable evidence run",
              dataset_name: "Human review set",
              agent_spec: "qa-agent.md",
            },
            summary: {
              overall_attempt_pass_rate: 0.5,
              passed_attempts: 1,
              quality_accounted_attempts: 2,
              macro_mean_scored_attempt_required_atom_recall: 0.75,
              attempt_lifecycle_counts: {
                scored: 2,
                execution_failed: 0,
                evaluation_failed: 0,
              },
            },
            prepared_items: [{
              item_id: "q-human",
              question,
              gold_atoms: atoms,
            }],
            answers: [
              {
                attempt_id: "q-human-attempt-1",
                item_id: "q-human",
                ordinal: 1,
                duration_ms: 1200,
                tool_calls: [
                  { ordinal: 1, name: "search", status: "success", duration_ms: 200 },
                  { ordinal: 2, name: "lookup", status: "success", duration_ms: 100 },
                ],
                answer: "Archi stores immutable snapshots, full answers, and per-atom judgments.",
              },
              {
                attempt_id: "q-human-attempt-2",
                item_id: "q-human",
                ordinal: 2,
                duration_ms: 2800,
                tool_calls: [
                  {
                    ordinal: 1,
                    name: "search",
                    status: "success",
                    query: '{"query":"immutable evaluation evidence","record_id":9007199254740993,"duplicate":"first","duplicate":"second"}',
                    response: JSON.stringify({ matches: ["snapshot", longToolResponse] }),
                    duration_ms: 1900,
                  },
                  {
                    ordinal: 2,
                    name: "lookup",
                    status: "error",
                    query: "missing-record",
                    error: '<img src=x onerror="window.traceInjected=true">Lookup failed',
                    duration_ms: 400,
                  },
                  {
                    ordinal: 3,
                    name: "unfinished_fetch",
                    status: "incomplete",
                    query: '{"cursor":"next"}',
                  },
                ],
                answer: "Archi stores snapshots but discards detailed atom judgments.",
              },
            ],
            evaluation_results: results,
            report_available: false,
          },
        }),
      });
    });
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          runs: [{
            id: historyId,
            valid: true,
            name: "Readable evidence run",
            status: "scored",
            attempts: 2,
            overall_attempt_pass_rate: 0.5,
          }],
        }),
      });
    });

    await page.goto("/evaluations");
    await page.locator(`[data-run-id="${historyId}"]`).click();
    await expect(page.getByRole("heading", { name: "Readable evidence run" })).toBeVisible();
    const latencyPanel = page.locator(".latency-panel");
    await expect(latencyPanel.getByRole("heading", { name: "Latency per question" })).toBeVisible();
    await expect(latencyPanel.getByRole("listitem")).toHaveCount(1);
    await expect(latencyPanel).toContainText(question);
    const latencyItem = latencyPanel.locator(".latency-item");
    const attemptSelector = latencyItem.getByLabel("Attempt for q-human");
    await expect(attemptSelector).toHaveValue("q-human-attempt-1");
    await expect(latencyItem.locator(".latency-total-value")).toHaveText("1.20 s total");
    await expect(latencyItem.locator(".latency-tool-value")).toHaveText("300 ms tools");
    await expect(latencyItem.locator(".latency-other-value")).toHaveText("900 ms other agent time");
    await expect(latencyItem).toContainText("2 tool calls");
    const latencyBar = latencyItem.locator(".latency-bar");
    await expect(latencyBar).toHaveAttribute("style", /height:\s*42\.86%/);
    await expect(latencyItem.locator(".latency-tool-segment")).toHaveAttribute(
      "style",
      /height:\s*25(\.00)?%/,
    );
    expect(await latencyBar.evaluate((bar) => getComputedStyle(bar).transitionProperty)).toContain(
      "height",
    );

    await attemptSelector.selectOption("q-human-attempt-2");
    await expect(latencyItem.locator(".latency-total-value")).toHaveText("2.80 s total");
    await expect(latencyItem.locator(".latency-tool-value")).toHaveText("2.30 s timed tools");
    await expect(latencyItem.locator(".latency-other-value")).toHaveText("Remaining time unattributed");
    await expect(latencyItem.locator(".latency-tool-count")).toHaveText("2 timed of 3 tool calls");
    await expect(latencyBar).toHaveAttribute("style", /height:\s*100(\.00)?%/);
    await expect(latencyItem.locator(".latency-tool-segment")).toHaveAttribute(
      "style",
      /height:\s*82\.14%/,
    );
    await expect(latencyItem.locator(".latency-unknown-segment")).toHaveAttribute(
      "style",
      /height:\s*17\.86%/,
    );
    await expect(page.locator("#run-detail-content > :first-child")).toHaveClass(/latency-panel/);
    await expect(page.getByText("Atoms recall", { exact: true })).toBeVisible();
    await expect(page.getByText("Scored attempts", { exact: true })).toHaveCount(0);
    await expect(page.getByText("66.7%", { exact: true })).toBeVisible();
    const atomRecallInfo = page.getByRole("button", { name: "About atoms recall" });
    await atomRecallInfo.hover();
    const atomRecallHelp = page.getByRole("tooltip", { name: /all atoms marked as entailed/ });
    await expect(atomRecallHelp).toBeVisible();
    await expect(atomRecallHelp).toContainText("including required and optional atoms");
    await expect(atomRecallHelp).toContainText("contradictions also reduce the separate atom score");

    await expect(page.getByText("Required atoms recall", { exact: true })).toBeVisible();
    const recallInfo = page.getByRole("button", { name: "About required atoms recall" });
    await recallInfo.hover();
    const recallHelp = page.getByRole("tooltip", { name: /required atoms marked as entailed/ });
    await expect(recallHelp).toBeVisible();
    await expect(recallHelp).toContainText("required atoms marked as entailed ÷ all required atoms");
    await expect(recallHelp).toContainText("Aim for 100%");

    const questionGroup = page.locator(".question-result");
    await expect(questionGroup).toHaveCount(1);
    const questionSummary = questionGroup.locator(":scope > summary");
    await expect(questionSummary).toContainText("q-human");
    await expect(questionSummary).toContainText(question);
    await expect(questionSummary).toContainText("Average score");
    await expect(questionSummary).toContainText("62.5%");
    await expect(questionSummary).toContainText("Best A1 100.0% · Worst A2 25.0%");
    await expect(page.getByLabel("Average atom score for q-human")).toHaveAttribute(
      "value",
      "0.625",
    );

    await questionSummary.click();
    const userQuestion = questionGroup.locator(".user-question");
    await expect(userQuestion).toHaveAttribute("open", "");
    await expect(userQuestion.locator(".evidence-copy")).toHaveText(question);

    const attempts = questionGroup.locator(".attempt-result");
    await expect(attempts).toHaveCount(2);
    await expect(attempts.nth(0).locator(":scope > summary")).toContainText("Attempt 1");
    await expect(attempts.nth(0).locator(":scope > summary")).toContainText("100.0%");
    await expect(attempts.nth(1).locator(":scope > summary")).toContainText("Attempt 2");
    await expect(attempts.nth(1).locator(":scope > summary")).toContainText("25.0%");

    await attempts.nth(0).locator(":scope > summary").click();
    const historicalTools = attempts.nth(0).locator(".tool-call-disclosure");
    await historicalTools.locator(":scope > summary").click();
    const historicalCall = historicalTools.locator(".tool-call-detail").first();
    await historicalCall.locator(":scope > summary").click();
    await expect(historicalCall.locator(".tool-call-duration")).toHaveText("200 ms");
    await expect(historicalCall).toContainText(
      "Query and response details were not captured for this historical call.",
    );

    await attempts.nth(1).locator(":scope > summary").click();
    const answer = attempts.nth(1).locator(".model-answer");
    await expect(answer).toHaveAttribute("open", "");
    await expect(answer.locator(".evidence-copy")).toContainText(
      "discards detailed atom judgments",
    );

    const toolDisclosure = attempts.nth(1).locator(".tool-call-disclosure");
    await expect(toolDisclosure.locator(":scope > summary")).toContainText("3 calls");
    await toolDisclosure.locator(":scope > summary").focus();
    expect(await toolDisclosure.locator(":scope > summary").evaluate(
      (summary) => getComputedStyle(summary).boxShadow,
    )).toContain("rgb(16, 163, 127)");
    await page.keyboard.press("Enter");
    await expect(toolDisclosure).toHaveAttribute("open", "");
    const toolCalls = toolDisclosure.locator(".tool-call-detail");
    await expect(toolCalls).toHaveCount(3);

    await expect(toolCalls.nth(0).locator("pre")).toHaveCount(0);
    await toolCalls.nth(0).locator(":scope > summary").focus();
    expect(await toolCalls.nth(0).locator(":scope > summary").evaluate(
      (summary) => getComputedStyle(summary).boxShadow,
    )).toContain("rgb(16, 163, 127)");
    await page.keyboard.press("Enter");
    await expect(toolCalls.nth(0)).toHaveAttribute("open", "");
    await expect(toolCalls.nth(0).locator(".tool-call-duration")).toHaveText("1900 ms");
    await expect(toolCalls.nth(0).getByText("Query", { exact: true })).toBeVisible();
    await expect(toolCalls.nth(0).locator("pre").first()).toContainText(
      '"query": "immutable evaluation evidence"',
    );
    await expect(toolCalls.nth(0).locator("pre").first()).toContainText(
      '"record_id": 9007199254740993',
    );
    await expect(toolCalls.nth(0).locator("pre").first()).toContainText(
      '"duplicate": "first"',
    );
    await expect(toolCalls.nth(0).locator("pre").first()).toContainText(
      '"duplicate": "second"',
    );
    const responseText = await toolCalls.nth(0).locator("pre").nth(1).textContent();
    expect(responseText).toContain(longToolResponse);
    const responsePre = toolCalls.nth(0).locator("pre").nth(1);
    await expect(responsePre).toHaveAttribute("tabindex", "0");
    const responseLabelId = await responsePre.getAttribute("aria-labelledby");
    expect(responseLabelId).toBeTruthy();
    await expect(page.locator(`#${responseLabelId}`)).toHaveText("Response");
    const responseLayout = await toolCalls.nth(0).locator("pre").nth(1).evaluate((pre) => ({
      clientHeight: pre.clientHeight,
      scrollHeight: pre.scrollHeight,
      overflowY: getComputedStyle(pre).overflowY,
    }));
    expect(responseLayout.overflowY).toBe("auto");
    expect(responseLayout.scrollHeight).toBeGreaterThan(responseLayout.clientHeight);
    await responsePre.focus();
    await page.keyboard.press("PageDown");
    await expect.poll(() => responsePre.evaluate((pre) => pre.scrollTop)).toBeGreaterThan(0);

    await toolCalls.nth(1).locator(":scope > summary").click();
    await expect(toolCalls.nth(1).getByText("Error", { exact: true })).toBeVisible();
    await expect(toolCalls.nth(1).locator("pre").nth(1)).toContainText(
      '<img src=x onerror="window.traceInjected=true">Lookup failed',
    );
    await expect(toolCalls.nth(1).locator("img")).toHaveCount(0);
    expect(await page.evaluate(() => (window as Window & { traceInjected?: boolean }).traceInjected)).toBeUndefined();

    await toolCalls.nth(2).locator(":scope > summary").click();
    await expect(toolCalls.nth(2).locator(".tool-call-duration")).toHaveCount(0);
    await expect(toolCalls.nth(2)).toContainText(
      "No tool response was captured for this incomplete call.",
    );

    const judgments = attempts.nth(1).locator(".atom-judgment");
    await expect(judgments).toHaveCount(3);
    await expect(judgments.nth(0).locator(".atom-outcome")).toHaveText("Passed");
    await expect(judgments.nth(0)).toHaveClass(/outcome-passed/);
    await expect(judgments.nth(1).locator(".atom-outcome")).toHaveText("Not mentioned");
    await expect(judgments.nth(1)).toHaveClass(/outcome-not-mentioned/);
    await expect(judgments.nth(2).locator(".atom-outcome")).toHaveText("Contradicted");
    await expect(judgments.nth(2)).toHaveClass(/outcome-contradicted/);
    await expect(judgments.nth(2).getByText("Expected content", { exact: true })).toBeVisible();
    await expect(judgments.nth(2).getByText("Evaluator judgment", { exact: true })).toBeVisible();

    await answer.locator(":scope > summary").click();
    await expect(answer).not.toHaveAttribute("open", "");
    await userQuestion.locator(":scope > summary").click();
    await expect(userQuestion).not.toHaveAttribute("open", "");

    await page.setViewportSize({ width: 390, height: 844 });
    await atomRecallInfo.focus();
    await expect(atomRecallHelp).toBeVisible();
    const tooltipBounds = await atomRecallHelp.boundingBox();
    expect(tooltipBounds).not.toBeNull();
    expect(tooltipBounds!.x).toBeGreaterThanOrEqual(0);
    expect(tooltipBounds!.x + tooltipBounds!.width).toBeLessThanOrEqual(390);
  });

  test("shows answer-owned tool evidence before a run is scored", async ({ page }) => {
    const historyId = "run-completed-tools";
    await page.route(`**/api/evaluations/runs/${historyId}`, async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          run: {
            manifest: {
              run_id: "run-completed",
              schema_version: "qa-v1",
              status: "run_completed",
            },
            metadata: {
              name: "Unscored tool run",
              dataset_name: "Pending score dataset",
              agent_spec: "qa-agent.md",
            },
            prepared_items: [{
              item_id: "pending-question",
              question: "What evidence was gathered?",
              gold_atoms: [{ id: "A1", text: "Evidence", required: true }],
            }],
            answers: [{
              item_id: "pending-question",
              attempt_id: "pending-question-attempt-1",
              ordinal: 1,
              status: "answer_ready",
              duration_ms: 20,
              answer: "The evidence was gathered.",
              tool_calls: [{
                ordinal: 1,
                name: "search",
                status: "success",
                query: '{"query":"evidence"}',
                response: '{"matches":["evidence"]}',
                duration_ms: 7,
              }],
            }, {
              item_id: "pending-question",
              attempt_id: "pending-question-attempt-2",
              ordinal: 2,
              status: "answer_ready",
              duration_ms: 30,
              answer: "The unfinished lookup produced no terminal result.",
              tool_calls: [{
                ordinal: 1,
                name: "unfinished_lookup",
                status: "incomplete",
                query: '{"cursor":"next"}',
              }],
            }],
            evaluation_results: [],
            report_available: false,
          },
        }),
      });
    });
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          runs: [{
            id: historyId,
            valid: true,
            name: "Unscored tool run",
            status: "run_completed",
            attempts: 2,
          }],
        }),
      });
    });

    await page.goto("/evaluations");
    await page.locator(`[data-run-id="${historyId}"]`).click();
    const latencyItem = page.locator(".latency-item");
    await latencyItem.getByLabel("Attempt for pending-question").selectOption(
      "pending-question-attempt-2",
    );
    await expect(latencyItem.locator(".latency-tool-value")).toHaveText(
      "Tool timing unavailable",
    );
    await expect(latencyItem.locator(".latency-other-value")).toHaveText(
      "Remaining time unattributed",
    );
    await expect(latencyItem.locator(".latency-tool-count")).toHaveText(
      "0 timed of 1 tool call",
    );
    await expect(latencyItem.locator(".latency-unknown-segment")).toHaveAttribute(
      "style",
      /height:\s*100(\.00)?%/,
    );
    const question = page.locator(".question-result");
    await expect(question.locator(":scope > summary")).toContainText("No scored attempts");
    await question.locator(":scope > summary").click();
    const attempt = question.locator(".attempt-result").first();
    await expect(attempt.locator(":scope > summary")).toContainText("answer_ready");
    await expect(attempt.locator(":scope > summary")).toContainText("Not scored");
    await attempt.locator(":scope > summary").click();
    const tools = attempt.locator(".tool-call-disclosure");
    await tools.locator(":scope > summary").click();
    const call = tools.locator(".tool-call-detail");
    await call.locator(":scope > summary").click();
    await expect(call.locator("pre").first()).toContainText('"query": "evidence"');
    await expect(call.locator("pre").nth(1)).toContainText('"matches": [');
  });

  test("marks tool latency unavailable for a historical total-only attempt", async ({ page }) => {
    const historyId = "legacy-total-only";
    await page.route(`**/api/evaluations/runs/${historyId}`, async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          run: {
            manifest: {
              run_id: "legacy-total-run",
              schema_version: "qa-v1",
              status: "scored",
            },
            metadata: {
              name: "Legacy total-only run",
              dataset_name: "Historical dataset",
              agent_spec: "qa-agent.md",
            },
            summary: {
              overall_attempt_pass_rate: 1,
              passed_attempts: 1,
              quality_accounted_attempts: 1,
              macro_mean_scored_attempt_required_atom_recall: 1,
              attempt_lifecycle_counts: {
                scored: 1,
                execution_failed: 0,
                evaluation_failed: 0,
              },
            },
            prepared_items: [{
              item_id: "legacy-question",
              question: "How long did this historical attempt take?",
              gold_atoms: [{ id: "A1", text: "It took 1.50 seconds.", required: true }],
            }],
            answers: [{
              item_id: "legacy-question",
              attempt_id: "legacy-question-attempt-1",
              ordinal: 1,
              status: "answer_ready",
              duration_ms: 1500,
              answer: "It took 1.50 seconds.",
            }],
            evaluation_results: [{
              item_id: "legacy-question",
              attempt_id: "legacy-question-attempt-1",
              ordinal: 1,
              status: "scored",
              passed: true,
              atom_score: 1,
              judgments: [{
                atom_id: "A1",
                outcome: "entailed",
                rationale: "The answer provides the expected duration.",
              }],
            }],
            report_available: false,
          },
        }),
      });
    });
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          runs: [{
            id: historyId,
            valid: true,
            name: "Legacy total-only run",
            status: "scored",
            attempts: 1,
            overall_attempt_pass_rate: 1,
          }],
        }),
      });
    });

    await page.goto("/evaluations");
    await page.locator(`[data-run-id="${historyId}"]`).click();

    const latencyItem = page.locator(".latency-item");
    await expect(latencyItem.locator(".latency-total-value")).toHaveText("1.50 s total");
    await expect(latencyItem.locator(".latency-tool-value")).toHaveText(
      "Tool timing unavailable",
    );
    await expect(latencyItem.locator(".latency-other-value")).toHaveText(
      "Remaining time unavailable",
    );
    await expect(latencyItem.locator(".latency-unknown-segment")).toHaveAttribute(
      "style",
      /height:\s*100(\.00)?%/,
    );
    const question = page.locator(".question-result");
    await question.locator(":scope > summary").click();
    const attempt = question.locator(".attempt-result");
    await attempt.locator(":scope > summary").click();
    const toolDisclosure = attempt.locator(".tool-call-disclosure");
    await toolDisclosure.locator(":scope > summary").click();
    await expect(toolDisclosure).toContainText(
      "Tool-call details were not captured for this historical attempt.",
    );
  });

  test("does not offer retries for read-only qa-v0 history", async ({ page }) => {
    const historyId = "legacy-read-only";
    await page.route(`**/api/evaluations/runs/${historyId}`, async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          run: {
            manifest: {
              run_id: "legacy-read-only-run",
              schema_version: "qa-v0",
              status: "scored",
            },
            metadata: { name: "Legacy read-only run" },
            capabilities: { retry_failed: false },
            summary: {
              overall_attempt_pass_rate: 0,
              attempt_lifecycle_counts: {
                scored: 0,
                execution_failed: 1,
                evaluation_failed: 0,
              },
            },
            prepared_items: [{
              item_id: "legacy-question",
              question: "Can this historical failure be retried?",
              gold_atoms: [{ id: "A1", text: "No.", required: true }],
            }],
            answers: [{
              item_id: "legacy-question",
              attempt_id: "legacy-question-attempt-1",
              ordinal: 1,
              status: "execution_failed",
              error: { type: "RuntimeError", message: "Historical failure" },
            }],
            evaluation_results: [],
            report_available: false,
          },
        }),
      });
    });
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          runs: [{
            id: historyId,
            valid: true,
            name: "Legacy read-only run",
            status: "scored",
            attempts: 1,
            overall_attempt_pass_rate: 0,
          }],
        }),
      });
    });

    await page.goto("/evaluations");
    await page.locator(`[data-run-id="${historyId}"]`).click();

    await expect(page.locator("#retry-failed-evaluation")).toBeHidden();
    await expect(page.getByRole("heading", { name: "Legacy read-only run" })).toBeVisible();
  });

  test("does not synthesize latency for a legacy run without attempt timings", async ({ page }) => {
    const historyId = "legacy-without-latency";
    await page.route(`**/api/evaluations/runs/${historyId}`, async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          run: {
            manifest: {
              run_id: "legacy-run",
              schema_version: "qa-v1",
              status: "scored",
            },
            metadata: {
              name: "Legacy run",
              dataset_name: "Historical dataset",
              agent_spec: "qa-agent.md",
            },
            summary: {
              overall_attempt_pass_rate: 1,
              passed_attempts: 1,
              quality_accounted_attempts: 1,
              macro_mean_scored_attempt_required_atom_recall: 1,
              attempt_lifecycle_counts: {
                scored: 1,
                execution_failed: 0,
                evaluation_failed: 0,
              },
            },
            prepared_items: [{
              item_id: "legacy-question",
              question: "How was this historical answer produced?",
              gold_atoms: [{ id: "A1", text: "It predates timing capture.", required: true }],
            }],
            answers: [{
              item_id: "legacy-question",
              attempt_id: "legacy-question-attempt-1",
              ordinal: 1,
              status: "answer_ready",
              tool_calls: [],
              answer: "This answer predates timing capture.",
            }],
            evaluation_results: [{
              item_id: "legacy-question",
              attempt_id: "legacy-question-attempt-1",
              ordinal: 1,
              status: "scored",
              passed: true,
              atom_score: 1,
              judgments: [{
                atom_id: "A1",
                outcome: "entailed",
                rationale: "The answer provides the expected fact.",
              }],
            }],
            report_available: false,
          },
        }),
      });
    });
    await page.route("**/api/evaluations/runs?*", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          runs: [{
            id: historyId,
            valid: true,
            name: "Legacy run",
            status: "scored",
            attempts: 1,
            overall_attempt_pass_rate: 1,
          }],
        }),
      });
    });

    await page.goto("/evaluations");
    await page.locator(`[data-run-id="${historyId}"]`).click();

    const latencyPanel = page.locator(".latency-panel");
    await expect(latencyPanel).toContainText("Latency unavailable for this run.");
    await expect(latencyPanel).toContainText(
      "This historical run has no authoritative per-attempt timings.",
    );
    await expect(latencyPanel.getByRole("listitem")).toHaveCount(0);
    const question = page.locator(".question-result");
    await question.locator(":scope > summary").click();
    const attempt = question.locator(".attempt-result");
    await attempt.locator(":scope > summary").click();
    const toolDisclosure = attempt.locator(".tool-call-disclosure");
    await toolDisclosure.locator(":scope > summary").click();
    await expect(toolDisclosure).toContainText(
      "This attempt performed no recorded tool calls.",
    );
  });

  test("retries only failed atoms and opens a complete evaluation successor", async ({ page }) => {
    await page.goto("/evaluations");
    await page.getByRole("button", { name: /Datasets/ }).click();
    await page.getByLabel("Name").first().fill("Failure retry dataset");
    await page.getByLabel("Dataset file").setInputFiles({
      name: "failure-retry.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify([
        {
          id: "retry-scored",
          question: "Keep this successful atom?",
          answer: "Keep this generated atom.",
          time_sensitive: false,
        },
        {
          id: "retry-execution",
          question: "Recover this atom?",
          answer: "Recover this generated atom.",
          time_sensitive: false,
        },
        {
          id: "retry-evaluation",
          question: "Score this answer again?",
          answer: "Reuse this terminal answer.",
          time_sensitive: false,
        },
      ])),
    });
    await page.getByRole("button", { name: "Validate and import" }).click();
    await page.locator("#dataset-list [data-dataset-id]").filter({
      has: page.getByText("Failure retry dataset", { exact: true }),
    }).click();
    await page.getByRole("button", { name: "Generate Atoms" }).click();

    const dialog = page.getByRole("dialog", {
      name: "Review atoms · Failure retry dataset",
    });
    await expect(dialog).toBeVisible({ timeout: 15_000 });
    await expect(dialog.getByText("Generation failed:", { exact: false })).toHaveCount(1);
    const retryAtoms = dialog.getByRole("button", {
      name: "Retry failed atoms (1)",
    });
    await expect(retryAtoms).toBeVisible();
    const preservedPanel = dialog.locator("details").filter({
      hasText: "Keep this successful atom?",
    });
    if ((await preservedPanel.getAttribute("open")) === null) {
      await preservedPanel.locator("summary").click();
    }
    const preservedAtom = preservedPanel.getByLabel(
      "Atom text 1 for retry-scored",
    );
    await preservedAtom.fill("Unsaved reviewer edit remains visible.");

    await retryAtoms.click();
    await preservedAtom.fill("Edit made while retry was running.");

    await expect(dialog.getByText("Generation failed:", { exact: false })).toHaveCount(0, {
      timeout: 15_000,
    });
    await expect(
      dialog.getByRole("button", { name: /Retry failed atoms/ }),
    ).toHaveCount(0);
    await expect(preservedAtom).toHaveValue("Edit made while retry was running.");
    await dialog.getByLabel("New dataset name").fill(
      "Failure retry dataset · reviewed",
    );
    await dialog.getByRole("button", { name: "Save as new dataset" }).click();

    await expect(page.locator("#dataset-detail").getByRole("heading", {
      name: "Failure retry dataset · reviewed",
    })).toBeVisible();
    await page.getByRole("button", { name: "Evaluate" }).click();
    await page.getByLabel("Evaluation name").fill("Failure-only evaluation");
    await page.getByRole("button", { name: "Start evaluation" }).click();

    await expect(page.getByRole("heading", {
      name: "Failure-only evaluation",
    })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole("button", {
      name: "Retry failed attempts (2)",
    })).toBeVisible();
    await expect(page.locator(".attempt-result .status")).toHaveText([
      "scored",
      "execution_failed",
      "evaluation_failed",
    ]);
    const executionFailure = page.locator('.question-result[data-question-id="retry-execution"]');
    await expect(executionFailure.locator(":scope > summary")).toContainText(
      "No scored attempts",
    );
    await expect(executionFailure.locator(":scope > summary")).toContainText(
      "1 attempt · awaiting scores",
    );
    await executionFailure.locator(":scope > summary").click();
    await expect(executionFailure.locator(".attempt-score")).toHaveText("Not scored");

    await page.getByRole("button", {
      name: "Retry failed attempts (2)",
    }).click();

    await expect(page.getByRole("heading", {
      name: "Failure-only evaluation · retry 1",
    })).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(".lineage-callout")).toContainText(
      "retried from Failure-only evaluation",
    );
    await expect(page.locator(".attempt-result .status")).toHaveText([
      "scored",
      "scored",
      "scored",
    ]);
    await expect(page.getByRole("button", {
      name: /Retry failed attempts/,
    })).toHaveCount(0);
    await expect(page.locator(".evidence-card").getByText(
      "100.0%",
      { exact: true },
    ).first()).toBeVisible();

    await page.getByRole("button", { name: "← All runs" }).click();
    await expect(page.locator("#runs-body").getByText(
      "Failure-only evaluation",
      { exact: true },
    )).toBeVisible();
    await expect(page.locator("#runs-body").getByText(
      "Failure-only evaluation · retry 1",
      { exact: true },
    )).toBeVisible();
  });
});
