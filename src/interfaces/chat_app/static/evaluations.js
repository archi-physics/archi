(() => {
  "use strict";

  const state = {
    datasets: [], profiles: [], agents: [], runs: [], jobs: [], permissions: { can_run: true, can_manage: true }, selectedDataset: null,
    draft: null, openRunId: null, pollingJobId: null, reviewValidationActive: false,
    openToolCalls: new Map(), nextToolCallKey: 1, trendDatasetKeys: [],
    selectedTrendDatasets: new Set(), trendFiltersInitialized: false,
    trendPointData: new Map(), agentSnapshotRequest: 0, historyRange: "90d"
  };
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
  const shortHash = (value) => value ? `${value.slice(0, 8)}…${value.slice(-5)}` : "built-in";
  const percent = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
  const formatDuration = (value) => {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "—";
    if (value < 1000) return `${Math.round(value)} ms`;
    return `${(value / 1000).toFixed(value < 10000 ? 2 : 1)} s`;
  };
  const when = (value) => value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "—";
  const scoreValue = (value) => typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null;
  const liveCheckLabel = (projection) => {
    const labels = { not_checked: "Not checked", matched_baseline: "Matched baseline", change_observed: "Change observed", last_check_unavailable: "Last check unavailable" };
    if (!projection) return "Not checked";
    return `${labels[projection.status] || "Not checked"}${projection.checked_at ? ` · ${when(projection.checked_at)}` : ""}`;
  };
  const outcomePresentation = {
    entailed: { label: "Passed", className: "passed" },
    not_mentioned: { label: "Not mentioned", className: "not-mentioned" },
    contradicted: { label: "Contradicted", className: "contradicted" }
  };

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("json") ? await response.json() : await response.text();
    if (!response.ok) throw new Error(payload.error || payload || `Request failed (${response.status})`);
    return payload;
  }

  function toast(title, message) {
    $("#toast-title").textContent = title;
    $("#toast-message").textContent = message;
    $("#toast").classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => $("#toast").classList.remove("show"), 3200);
  }

  function showView(name) {
    $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
    $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function activeJob() {
    return state.jobs.find((job) => ["queued", "running", "cancel_requested", "attention_required"].includes(job.status));
  }

  function renderRuntime() {
    const job = activeJob();
    const runtimePhase = job?.progress?.runtime_phase;
    const progress = job?.status === "cancel_requested"
      ? null
      : (runtimePhase ? runtimePhase.replaceAll("_", " ") : job?.progress?.status);
    const canRun = state.permissions.can_run !== false;
    const canSeeAttention = canRun || state.permissions.can_manage;
    const attention = job?.status === "attention_required" && canSeeAttention;
    const checkingLiveAnswers = job?.kind === "evaluation"
      && ["queued", "running"].includes(job.status)
      && runtimePhase === "checking_live_answers";
    $("#runtime-label").textContent = checkingLiveAnswers
      ? "Checking live answers…"
      : (job ? `${job.kind.replace("_", " ")} · ${progress || job.status}` : "Ready");
    $("#metric-job").textContent = job && !attention ? "Yes" : "No";
    $("#job-banner").hidden = !job;
    const cancelButton = $("#cancel-evaluation");
    const continueButton = $("#continue-evaluation");
    const refreshButton = $("#refresh-live-evaluation");
    const attentionPanel = $("#attention-panel");
    if (job) {
      $("#job-banner-message").textContent = attention
        ? "Live answers changed or could not be checked. No agent attempts have started."
        : (checkingLiveAnswers
          ? "Checking live answers… Agent attempts start only after the fresh pre-check completes."
          : `${job.kind.replace("_", " ")} is ${progress || job.status}. This page can be closed; state is persisted.`);
      cancelButton.hidden = job.kind !== "evaluation" || !canRun;
      cancelButton.disabled = job.status === "cancel_requested";
      cancelButton.textContent = job.status === "cancel_requested" ? "Canceling…" : "Cancel evaluation";
      continueButton.hidden = !attention || !canRun;
      refreshButton.hidden = !attention;
      attentionPanel.hidden = !attention;
      if (attention) renderAttention(job);
    } else {
      $("#job-banner-message").textContent = "";
      cancelButton.hidden = true;
      continueButton.hidden = true;
      refreshButton.hidden = true;
      attentionPanel.hidden = true;
    }
  }

  function renderAttention(job) {
    const attention = job.result?.attention_required || {};
    const affected = attention.affected_items || [];
    const changed = affected.some((item) => item.reason === "answer_changed");
    const managerAction = changed
      ? "The live answer changed. Refresh the live snapshot before running a complete evaluation."
      : "One or more live answers could not be checked. Refresh the live snapshot before running a complete evaluation.";
    $("#attention-remediation").textContent = state.permissions.can_manage
      ? managerAction
      : `${changed ? "The live answer changed." : "One or more live answers could not be checked."} Contact a dataset manager to refresh the approved live snapshot.`;
    $("#attention-summary").innerHTML = `
      <div><span>Live questions checked</span><strong>${attention.live_items ?? "—"}</strong></div>
      <div><span>Affected questions</span><strong>${affected.length}</strong></div>
      <div><span>Observation</span><strong>${esc(when(attention.checked_at || job.started_at))}</strong></div>`;
    $("#attention-items").innerHTML = affected.map((item) => item.question ? `<details class="attention-evidence">
      <summary><code>${esc(item.item_id)}</code><strong>${esc((item.reason || "oracle_failed").replaceAll("_", " "))}</strong><span>${esc(item.question)}</span></summary>
      <div><section><p class="field-label">Approved answer</p><pre>${esc(JSON.stringify(item.approved_answer, null, 2))}</pre></section><section><p class="field-label">Current observation</p><pre>${esc(JSON.stringify(item.current_answer, null, 2))}</pre></section><details><summary>Metadata, recipe, and calls</summary><pre>${esc(JSON.stringify({ metadata: item.metadata, oracle: item.oracle, calls: item.calls }, null, 2))}</pre></details></div>
    </details>` : `<article>
      <code>${esc(item.item_id)}</code><strong>${esc((item.reason || "oracle_failed").replaceAll("_", " "))}</strong><span>${esc((item.phase || "pre_run").replaceAll("_", " "))}</span>
    </article>`).join("");
    $("#continue-evaluation").disabled = attention.can_continue === false;
    $("#refresh-live-evaluation").hidden = !state.permissions.can_manage;
  }

  function renderLaunchOptions() {
    const launchable = state.datasets.filter((item) => item.time_sensitive_item_count === 0 || item.contains_live_answers);
    $("#launch-dataset").innerHTML = launchable.length
      ? launchable.map((item) => `<option value="${esc(item.id)}">${esc(item.name)} · ${item.item_count} items${item.contains_live_answers ? " · approved live snapshot" : (item.generation_scope === "static_only" ? " · static only" : "")}</option>`).join("")
      : `<option value="">Import a dataset first</option>`;
    $("#launch-profile").innerHTML = state.profiles.map((item) => `<option value="${esc(item.id)}">${esc(item.name)}</option>`).join("");
    $("#launch-agent").innerHTML = state.agents.length
      ? state.agents.map((item) => `<option value="${esc(item.id)}">${esc(item.name)} · ${esc(item.id)}</option>`).join("")
      : `<option value="">No agent specs found</option>`;
    renderProfileSnapshot();
    renderAgentSnapshot();
  }

  function descriptorSummary(label, descriptor) {
    const timeout = descriptor.timeout == null ? "default timeout" : `${descriptor.timeout}s timeout`;
    return `<div class="snapshot-row"><span>${esc(label)}</span><strong>${esc(descriptor.provider)} / ${esc(descriptor.model)}</strong><small>${esc(timeout)}</small></div>`;
  }

  function renderProfileSnapshot() {
    const profileId = $("#launch-profile").value;
    const profile = state.profiles.find((item) => item.id === profileId);
    if (!profile) {
      $("#profile-snapshot").innerHTML = `<div class="snapshot-status" role="status">No evaluator profile is available.</div>`;
      return;
    }
    $("#profile-snapshot").innerHTML = `
      <header class="snapshot-head"><div><span>Selected profile</span><h3 id="profile-snapshot-title">${esc(profile.name)}</h3></div><code>${esc(shortHash(profile.sha256))}</code></header>
      <div class="snapshot-content">
        ${descriptorSummary("Atom extractor", profile.components.atoms_extractor)}
        ${descriptorSummary("Answer evaluator", profile.components.evaluator)}
      </div>`;
  }

  async function renderAgentSnapshot() {
    const agentId = $("#launch-agent").value;
    const requestId = ++state.agentSnapshotRequest;
    if (!agentId) {
      $("#agent-snapshot").innerHTML = `<div class="snapshot-status" role="status">No agent spec is available.</div>`;
      return;
    }
    $("#agent-snapshot").setAttribute("aria-busy", "true");
    $("#agent-snapshot").innerHTML = `<div class="snapshot-status" role="status">Loading selected agent spec…</div>`;
    try {
      const { agent } = await api(`/api/evaluations/agents/${encodeURIComponent(agentId)}`);
      if (requestId !== state.agentSnapshotRequest || $("#launch-agent").value !== agentId) return;
      $("#agent-snapshot").innerHTML = `
        <header class="snapshot-head"><div><span>Selected agent</span><h3 id="agent-snapshot-title">${esc(agent.name)}</h3></div><code>${esc(agent.id)}</code></header>
        <div class="snapshot-tools"><span>Tools</span><strong>${agent.tools.length ? agent.tools.map(esc).join(" · ") : "None"}</strong></div>
        <pre class="agent-spec-content" tabindex="0" aria-label="Selected specification contents">${esc(agent.content)}</pre>`;
    } catch (error) {
      if (requestId !== state.agentSnapshotRequest || $("#launch-agent").value !== agentId) return;
      $("#agent-snapshot").innerHTML = `<div class="snapshot-status error" role="status"><strong>Could not load this agent spec.</strong><span>${esc(error.message)}</span></div>`;
    } finally {
      if (requestId === state.agentSnapshotRequest) $("#agent-snapshot").removeAttribute("aria-busy");
    }
  }

  function renderProfiles() {
    $("#profile-count").textContent = state.profiles.length;
    $("#profile-list").innerHTML = state.profiles.map((profile) => {
      const extractor = profile.components.atoms_extractor;
      const evaluator = profile.components.evaluator;
      return `<article class="profile-card ${profile.built_in ? "builtin" : ""}">
        <p class="eyebrow">${profile.built_in ? "Workspace default" : "Imported profile"}</p>
        <h2>${esc(profile.name)}</h2>
        <small class="muted">${esc(shortHash(profile.sha256))}</small>
        <div class="model-line"><span>Atom extractor</span><code>${esc(extractor.provider)} / ${esc(extractor.model)}</code></div>
        <div class="model-line"><span>Comparator</span><code>${esc(evaluator.provider)} / ${esc(evaluator.model)}</code></div>
      </article>`;
    }).join("");
  }

  function filteredDatasets() {
    const query = ($("#dataset-search").value || "").trim().toLowerCase();
    return state.datasets.filter((item) => !query || [item.name, item.source_filename, ...(item.categories || []), ...(item.answer_sources || [])].join(" ").toLowerCase().includes(query));
  }

  function renderDatasets() {
    const datasets = filteredDatasets();
    $("#dataset-count").textContent = state.datasets.length;
    $("#metric-datasets").textContent = state.datasets.length;
    $("#datasets-empty").hidden = datasets.length > 0;
    $("#dataset-list").innerHTML = datasets.map((item) => `<button class="catalog-item ${state.selectedDataset?.id === item.id ? "selected" : ""}" data-dataset-id="${esc(item.id)}">
      <span><strong>${esc(item.name)}</strong><small>${item.item_count} items · ${item.supplied_atom_item_count} with supplied atoms${item.parent_dataset_id ? " · reviewed child" : ""}${item.contains_live_answers ? " · includes live answers" : (item.generation_scope === "static_only" ? " · static only" : "")}</small></span>
      <code>${esc(shortHash(item.sha256))}</code>
    </button>`).join("");
    $$("[data-dataset-id]").forEach((button) => button.addEventListener("click", () => selectDataset(button.dataset.datasetId)));
  }

  function selectDataset(id) {
    state.selectedDataset = state.datasets.find((item) => item.id === id);
    renderDatasets();
    const item = state.selectedDataset;
    if (!item) return;
    const hasAtoms = item.atom_count > 0;
    const isDefinitionParent = item.dataset_role === "definition_parent";
    const unresolvedLive = isDefinitionParent && item.time_sensitive_item_count > 0;
    const approvedLive = Boolean(item.contains_live_answers);
    const staticOnly = item.generation_scope === "static_only";
    const resumableJob = state.jobs.find((job) =>
      job.kind === "generate_atoms" &&
      job.status === "completed" &&
      job.context?.dataset_id === item.id &&
      job.result?.draft_id &&
      job.result?.draft_status === "open"
    );
    $("#dataset-detail").innerHTML = `
      <p class="eyebrow">${item.parent_dataset_id ? "Reviewed child dataset" : "Imported dataset"}</p>
      <h2>${esc(item.name)}</h2>
      <p class="muted">${esc(item.source_filename)} · ${esc(shortHash(item.sha256))}</p>
      <div class="definition-list">
        <div><span>Total items</span><strong>${item.item_count}</strong></div>
        <div><span>Eligible / live</span><strong>${item.eligible_item_count} / ${item.time_sensitive_item_count}</strong></div>
        <div><span>Items with atoms</span><strong>${item.supplied_atom_item_count}</strong></div>
        <div><span>Imported</span><strong>${esc(when(item.created_at))}</strong></div>
        ${approvedLive ? `<div><span>Last live check</span><strong>${esc(liveCheckLabel(item.last_live_check))}</strong></div>` : ""}
      </div>
      <p class="eyebrow">Categories</p><div class="chips">${(item.categories || []).map((value) => `<span class="chip">${esc(value)}</span>`).join("") || `<span class="muted">None supplied</span>`}</div>
      <p class="eyebrow">Answer sources</p><div class="chips">${(item.answer_sources || []).map((value) => `<span class="chip">${esc(value)}</span>`).join("") || `<span class="muted">None supplied</span>`}</div>
      <div class="detail-actions">
        ${hasAtoms && !unresolvedLive ? "" : `<label>Atom generation profile<select id="atom-profile">${state.profiles.map((profile) => `<option value="${esc(profile.id)}">${esc(profile.name)}</option>`).join("")}</select></label>`}
        ${unresolvedLive ? `<label class="scope-choice"><input id="static-only-generation" type="checkbox"> <span>Create a static-only dataset — omit ${item.time_sensitive_item_count} live questions</span></label>` : ""}
        ${unresolvedLive ? `<button class="button primary" id="generate-atoms">Generate Atoms</button>` : (approvedLive || staticOnly ? "" : `<button class="button primary" id="${hasAtoms ? "review-atoms" : "generate-atoms"}">${hasAtoms ? "Review Atoms" : "Generate Atoms"}</button>`)}
        ${approvedLive && state.permissions.can_manage ? `<button class="button primary" id="refresh-live-snapshot">Refresh live snapshot</button>` : ""}
        ${staticOnly && state.permissions.can_manage ? `<button class="button primary" id="refresh-live-snapshot">Add live questions from parent</button>` : ""}
        ${(approvedLive || staticOnly) && !state.permissions.can_manage ? `<p class="callout supporting-callout">Contact a dataset manager to refresh this live snapshot.</p>` : ""}
        ${resumableJob ? `<button class="button quiet" id="resume-atoms" data-draft-id="${esc(resumableJob.result.draft_id)}">Resume atom review</button>` : ""}
        ${item.parent_dataset_id ? `<button class="button quiet" id="view-parent-definition">View parent definition</button>` : ""}
        ${unresolvedLive ? "" : `<button class="button quiet" id="evaluate-dataset">${item.dataset_role === "legacy" || item.generation_scope === "review" ? "Evaluate" : (staticOnly ? "Run static evaluation" : "Run evaluation")}</button>`}
      </div>`;
    $("#generate-atoms")?.addEventListener("click", generateAtoms);
    $("#review-atoms")?.addEventListener("click", reviewAtoms);
    $("#resume-atoms")?.addEventListener("click", (event) => resumeAtomDraft(event.currentTarget.dataset.draftId));
    $("#refresh-live-snapshot")?.addEventListener("click", () => refreshLiveSnapshot(item.id));
    $("#view-parent-definition")?.addEventListener("click", () => selectDataset(item.parent_dataset_id));
    $("#evaluate-dataset")?.addEventListener("click", () => {
      showView("new");
      $("#launch-dataset").value = item.id;
    });
  }

  function renderRuns() {
    const valid = state.runs.filter((run) => run.valid);
    $("#metric-runs").textContent = valid.length;
    $("#metric-pass").textContent = valid.length ? percent(valid[0].overall_attempt_pass_rate) : "—";
    $("#runs-empty").hidden = state.runs.length > 0;
    $("#runs-body").innerHTML = state.runs.map((run) => `<tr data-run-id="${esc(run.id)}">
      <td><strong>${esc(run.name || run.run_id || "Unnamed run")}</strong><small>${esc(when(run.created_at))}</small></td>
      <td>${esc(run.dataset_name || "CLI snapshot")}</td>
      <td><code>${esc(run.agent_spec || "resolved artifact")}</code></td>
      <td>${run.attempts ?? "—"}</td>
      <td>${percent(run.overall_attempt_pass_rate)}</td>
      <td><span class="status ${run.valid ? (run.status === "scored" ? "good" : (run.status === "canceled" ? "canceled" : "run")) : "bad"}">${esc(run.status)}</span></td>
    </tr>`).join("");
    $$("[data-run-id]").forEach((row) => row.addEventListener("click", () => openRun(row.dataset.runId)));
    renderHistoryTrends();
  }

  function scoredTrendRuns() {
    return state.runs.filter((run) => run.valid && run.status === "scored");
  }

  function trendDatasets() {
    const byKey = new Map();
    scoredTrendRuns().filter((run) => typeof run.dataset_key === "string" && run.dataset_key).forEach((run) => {
      if (!byKey.has(run.dataset_key)) byKey.set(run.dataset_key, run.dataset_name || "CLI snapshot");
    });
    const nameCounts = new Map();
    [...byKey.values()].forEach((name) => nameCounts.set(name, (nameCounts.get(name) || 0) + 1));
    return [...byKey.entries()].map(([key, name]) => ({
      key,
      name,
      label: nameCounts.get(name) > 1 ? `${name} · ${key.length > 18 ? shortHash(key) : key}` : name
    })).sort((left, right) => left.label.localeCompare(right.label));
  }

  function syncTrendDatasets(datasets) {
    const nextKeys = datasets.map((dataset) => dataset.key);
    const previouslyAll = !state.trendFiltersInitialized
      || state.trendDatasetKeys.every((key) => state.selectedTrendDatasets.has(key));
    const nextSelection = previouslyAll
      ? new Set(nextKeys)
      : new Set(nextKeys.filter((key) => state.selectedTrendDatasets.has(key)));
    if (!nextSelection.size && nextKeys.length) nextSelection.add(nextKeys[0]);
    state.trendDatasetKeys = nextKeys;
    state.selectedTrendDatasets = nextSelection;
    state.trendFiltersInitialized = true;
  }

  function updateTrendFilterStatus(message = "") {
    const selected = state.selectedTrendDatasets.size;
    const total = state.trendDatasetKeys.length;
    $("#trend-filter-status").textContent = message || (total
      ? `Showing ${selected} of ${total} datasets`
      : "No datasets in history");
    $("#trend-select-all").disabled = !total || selected === total;
  }

  function renderTrendFilters(datasets) {
    const filters = $("#trend-dataset-filters");
    filters.innerHTML = datasets.map((dataset) => `<label class="trend-dataset-option">
      <input type="checkbox" value="${esc(dataset.key)}" ${state.selectedTrendDatasets.has(dataset.key) ? "checked" : ""}>
      <span>${esc(dataset.label)}</span>
    </label>`).join("");
    filters.querySelectorAll('input[type="checkbox"]').forEach((input) => input.addEventListener("change", () => {
      if (input.checked) {
        state.selectedTrendDatasets.add(input.value);
      } else if (state.selectedTrendDatasets.size === 1) {
        input.checked = true;
        updateTrendFilterStatus("At least one dataset must remain selected.");
        return;
      } else {
        state.selectedTrendDatasets.delete(input.value);
      }
      updateTrendFilterStatus();
      renderTrendCharts();
    }));
    updateTrendFilterStatus();
  }

  function datedTrendRuns() {
    return scoredTrendRuns().map((run) => ({ run, timestamp: Date.parse(run.created_at) }))
      .filter(({ run, timestamp }) => state.selectedTrendDatasets.has(run.dataset_key)
        && Number.isFinite(timestamp))
      .sort((left, right) => left.timestamp - right.timestamp);
  }

  function selectedTrendRuns() {
    return scoredTrendRuns().filter((run) => state.selectedTrendDatasets.has(run.dataset_key));
  }

  function svgPath(points) {
    return points.map((point, index) => `${index ? "L" : "M"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
  }

  function trendDateLabel(timestamp, includeTime = false) {
    const options = includeTime
      ? { hour: "2-digit", minute: "2-digit", second: "2-digit" }
      : { month: "short", day: "numeric" };
    return new Intl.DateTimeFormat(undefined, options).format(new Date(timestamp));
  }

  function trendPointDetails(kind, series, run, value) {
    const parent = run.retry_of_history_id
      ? state.runs.find((candidate) => candidate.id === run.retry_of_history_id)
      : null;
    const details = {
      title: run.name || run.run_id || "Unnamed run",
      dataset: run.dataset_name || "CLI snapshot",
      date: when(run.created_at),
      lineage: run.retry_of_history_id ? `Retry of ${parent?.name || run.retry_of_history_id}` : "",
      lines: []
    };
    if (kind === "latency") {
      const latency = run.latency;
      const labels = { average: "Average latency", best: "Best latency", worst: "Worst latency" };
      details.valueLabel = `${labels[series]} ${formatDuration(value)}`;
      details.lines = [
        `Average ${formatDuration(latency.average_ms)} · Best ${formatDuration(latency.best_ms)} · Worst ${formatDuration(latency.worst_ms)}`,
        `${latency.timed_attempts} of ${latency.total_attempts} attempts timed`
      ];
    } else if (kind === "pass") {
      details.valueLabel = `Pass rate ${percent(value)}`;
      details.lines = Number.isInteger(run.passed_attempts) && Number.isInteger(run.quality_accounted_attempts)
        ? [`${run.passed_attempts} of ${run.quality_accounted_attempts} quality-accounted attempts passed`]
        : ["Pass numerator and denominator unavailable in this historical artifact"];
    } else {
      const counts = run.attempt_lifecycle_counts;
      const total = counts.scored + counts.execution_failed + counts.evaluation_failed;
      details.valueLabel = `Technical failure rate ${percent(value)}`;
      details.lines = [
        `${counts.execution_failed + counts.evaluation_failed} of ${total} terminal attempts failed technically`,
        `${counts.execution_failed} execution · ${counts.evaluation_failed} evaluation failures`
      ];
    }
    return details;
  }

  function renderTrendChart({ id, kind, series, rate = false }) {
    const container = $(`#${id}`);
    const dated = datedTrendRuns();
    const selected = selectedTrendRuns();
    const undatedCount = selected.length - dated.length;
    const seriesPoints = series.map((definition) => ({
      ...definition,
      points: dated.map(({ run, timestamp }, index) => ({ run, timestamp, index, value: definition.value(run) }))
        .filter((point) => typeof point.value === "number" && Number.isFinite(point.value) && point.value >= 0)
    }));
    const availableRunIds = new Set(seriesPoints.flatMap((definition) => definition.points.map((point) => point.run.id)));
    if (!dated.length || !availableRunIds.size) {
      const reason = dated.length
        ? "This selection has no authoritative values for this metric."
        : "This selection has no runs with an authoritative timestamp.";
      const coverage = [];
      if (dated.length) coverage.push(`${dated.length} without this metric`);
      if (undatedCount) coverage.push(`${undatedCount} without a timestamp`);
      container.innerHTML = `<div class="trend-empty"><strong>Trend unavailable.</strong><span>${esc(reason)}</span>${coverage.length ? `<small>${esc(coverage.join(" · "))}</small>` : ""}</div>`;
      return;
    }

    const width = 920, height = 280;
    const plot = { left: 62, right: 24, top: 24, bottom: 48 };
    const plotWidth = width - plot.left - plot.right;
    const plotHeight = height - plot.top - plot.bottom;
    const values = seriesPoints.flatMap((definition) => definition.points.map((point) => point.value));
    const maximumValue = rate ? 1 : Math.max(...values, 1);
    const indexByRun = new Map(dated.map((entry, index) => [entry.run, index]));
    const xFor = (index) => dated.length === 1
      ? plot.left + plotWidth / 2
      : plot.left + (index / (dated.length - 1)) * plotWidth;
    const yFor = (value) => plot.top + plotHeight - (Math.min(value, maximumValue) / maximumValue) * plotHeight;
    const yTicks = [0, .25, .5, .75, 1].map((fraction) => ({
      y: plot.top + plotHeight - fraction * plotHeight,
      label: rate ? `${Math.round(fraction * 100)}%` : formatDuration(maximumValue * fraction)
    }));
    const xIndexes = [...new Set([0, Math.floor((dated.length - 1) / 2), dated.length - 1])];
    const xDateLabels = xIndexes.map((index) => trendDateLabel(dated[index].timestamp));
    const showTimes = new Set(xDateLabels).size !== xDateLabels.length;
    const tooltipId = id.replace("-chart", "-tooltip");

    const paths = seriesPoints.map((definition) => {
      const points = definition.points.map((point) => ({
        ...point,
        x: xFor(indexByRun.get(point.run)),
        y: yFor(point.value)
      }));
      const segments = points.reduce((groups, point) => {
        const current = groups[groups.length - 1];
        if (!current || point.index !== current[current.length - 1].index + 1) groups.push([point]);
        else current.push(point);
        return groups;
      }, []);
      const path = segments.filter((segment) => segment.length > 1)
        .map((segment) => `<path class="trend-line trend-${definition.key}" d="${svgPath(segment)}"></path>`)
        .join("");
      const dots = points.map((point) => {
        const pointKey = `${kind}:${definition.key}:${point.run.id}`;
        const details = trendPointDetails(kind, definition.key, point.run, point.value);
        state.trendPointData.set(pointKey, details);
        const accessibleName = `${details.title}, ${details.dataset}, ${details.valueLabel}, ${details.date}`;
        return `<a href="#evaluation-${esc(point.run.id)}" class="trend-point-link" data-point-key="${esc(pointKey)}" data-run-id="${esc(point.run.id)}" aria-label="${esc(accessibleName)}" aria-describedby="${tooltipId}">
          <circle class="trend-point trend-${definition.key}" data-series="${esc(definition.key)}" cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="5"></circle>
        </a>`;
      }).join("");
      return `${path}${dots}`;
    }).join("");
    const partial = dated.length - availableRunIds.size;
    const summaryParts = [`${availableRunIds.size} run${availableRunIds.size === 1 ? "" : "s"} plotted`];
    if (partial) summaryParts.push(`${partial} without this metric`);
    if (undatedCount) summaryParts.push(`${undatedCount} without a timestamp`);
    container.innerHTML = `<div class="trend-chart-scroll">
      <svg class="trend-chart" viewBox="0 0 ${width} ${height}" role="group" aria-label="${esc(summaryParts.join("; "))}">
        <g class="trend-grid-lines">${yTicks.map((tick) => `<line x1="${plot.left}" y1="${tick.y}" x2="${width - plot.right}" y2="${tick.y}"></line><text x="${plot.left - 10}" y="${tick.y + 4}" text-anchor="end">${esc(tick.label)}</text>`).join("")}</g>
        <line class="trend-axis" x1="${plot.left}" y1="${plot.top + plotHeight}" x2="${width - plot.right}" y2="${plot.top + plotHeight}"></line>
        <g class="trend-x-labels">${xIndexes.map((index) => `<text x="${xFor(index)}" y="${height - 16}" text-anchor="middle">${esc(trendDateLabel(dated[index].timestamp, showTimes))}</text>`).join("")}</g>
        <g class="trend-series">${paths}</g>
      </svg>
    </div><p class="trend-summary">${esc(summaryParts.join(" · "))}</p>`;
  }

  function hideTrendTooltip(tooltip) {
    if (tooltip) tooltip.hidden = true;
  }

  function showTrendTooltip(link) {
    const details = state.trendPointData.get(link.dataset.pointKey);
    const tooltip = $(`#${link.getAttribute("aria-describedby")}`);
    if (!details || !tooltip) return;
    tooltip.replaceChildren();
    const title = document.createElement("strong");
    title.textContent = details.title;
    const dataset = document.createElement("span");
    dataset.textContent = `${details.dataset} · ${details.date}`;
    const value = document.createElement("b");
    value.textContent = details.valueLabel;
    tooltip.append(title, dataset, value);
    details.lines.forEach((line) => {
      const row = document.createElement("small");
      row.textContent = line;
      tooltip.append(row);
    });
    if (details.lineage) {
      const lineage = document.createElement("small");
      lineage.textContent = details.lineage;
      tooltip.append(lineage);
    }
    tooltip.hidden = false;
    const panel = link.closest(".trend-panel");
    const panelBox = panel.getBoundingClientRect();
    const pointBox = link.getBoundingClientRect();
    const left = Math.min(
      Math.max(12, pointBox.left + pointBox.width / 2 - panelBox.left - tooltip.offsetWidth / 2),
      panelBox.width - tooltip.offsetWidth - 12
    );
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${pointBox.bottom - panelBox.top + 9}px`;
  }

  function bindTrendPoints() {
    $$(".trend-point-link").forEach((link) => {
      const tooltip = $(`#${link.getAttribute("aria-describedby")}`);
      link.addEventListener("pointerenter", () => showTrendTooltip(link));
      link.addEventListener("pointerleave", () => hideTrendTooltip(tooltip));
      link.addEventListener("focus", () => showTrendTooltip(link));
      link.addEventListener("blur", () => hideTrendTooltip(tooltip));
      link.addEventListener("click", (event) => {
        event.preventDefault();
        openRun(link.dataset.runId);
      });
      link.addEventListener("keydown", (event) => {
        if (event.key === " ") {
          event.preventDefault();
          openRun(link.dataset.runId);
        }
      });
    });
  }

  function renderTrendCharts() {
    const focusedPoint = document.activeElement?.dataset?.pointKey;
    state.trendPointData = new Map();
    renderTrendChart({
      id: "trend-latency-chart", kind: "latency",
      series: [
        { key: "average", value: (run) => run.latency?.average_ms },
        { key: "best", value: (run) => run.latency?.best_ms },
        { key: "worst", value: (run) => run.latency?.worst_ms }
      ]
    });
    renderTrendChart({
      id: "trend-pass-chart", kind: "pass", rate: true,
      series: [{ key: "pass", value: (run) => run.overall_attempt_pass_rate }]
    });
    renderTrendChart({
      id: "trend-failure-chart", kind: "failure", rate: true,
      series: [{ key: "failure", value: (run) => run.technical_failure_rate }]
    });
    bindTrendPoints();
    if (focusedPoint) {
      const replacement = document.querySelector(`[data-point-key="${CSS.escape(focusedPoint)}"]`);
      if (replacement) replacement.focus();
      else $("#trend-dataset-filters input:checked")?.focus();
    }
  }

  function renderHistoryTrends() {
    const datasets = trendDatasets();
    syncTrendDatasets(datasets);
    renderTrendFilters(datasets);
    renderTrendCharts();
  }

  function renderTrendError(error) {
    $("#trend-filter-status").textContent = "Trend history could not be loaded.";
    ["trend-latency-chart", "trend-pass-chart", "trend-failure-chart"].forEach((id) => {
      $(`#${id}`).innerHTML = `<div class="trend-empty error"><strong>Could not load trends.</strong><span>${esc(error.message)}</span></div>`;
    });
  }

  async function loadCatalog() {
    const selectedDatasetId = state.selectedDataset?.id;
    const payload = await api("/api/evaluations/catalog");
    Object.assign(state, payload);
    renderRuntime();
    renderProfiles();
    renderLaunchOptions();
    const nextDatasetId = state.datasets.some((item) => item.id === selectedDatasetId)
      ? selectedDatasetId
      : state.datasets[0]?.id;
    if (nextDatasetId) selectDataset(nextDatasetId); else {
      state.selectedDataset = null;
      renderDatasets();
    }
    const job = activeJob();
    if (job && state.pollingJobId !== job.id) monitorExistingJob(job.id);
  }

  async function loadRuns() {
    const historyTrends = $(".history-trends");
    const historyWindow = $("#history-window");
    historyTrends.setAttribute("aria-busy", "true");
    historyWindow.disabled = true;
    try {
      const payload = await api(`/api/evaluations/runs?range=${encodeURIComponent(state.historyRange)}`);
      state.runs = payload.runs;
      renderRuns();
    } catch (error) {
      renderTrendError(error);
      throw error;
    } finally {
      historyTrends.removeAttribute("aria-busy");
      historyWindow.disabled = false;
    }
  }

  async function pollJob(jobId) {
    for (;;) {
      const { job } = await api(`/api/evaluations/jobs/${encodeURIComponent(jobId)}`);
      const existing = state.jobs.findIndex((item) => item.id === job.id);
      if (existing >= 0) state.jobs[existing] = job; else state.jobs.unshift(job);
      renderRuntime();
      if (["completed", "failed", "interrupted", "canceled", "attention_required"].includes(job.status)) {
        if (job.status === "attention_required") return { attention_required: true, job };
        if (job.status === "canceled") return { canceled: true };
        if (job.status !== "completed") throw new Error(job.error || `Job ${job.status}`);
        return job.result || {};
      }
      await new Promise((resolve) => setTimeout(resolve, 900));
    }
  }

  async function monitorExistingJob(jobId) {
    state.pollingJobId = jobId;
    try {
      const result = await pollJob(jobId);
      if (result.canceled) {
        toast("Evaluation canceled", "The stopped run is recorded in history.");
        await Promise.all([loadCatalog(), loadRuns()]);
        return;
      }
      if (result.attention_required) {
        toast("Live answers need attention", "No agent attempts have started.");
        await loadCatalog();
        const index = state.jobs.findIndex((item) => item.id === result.job.id);
        if (index >= 0) state.jobs[index] = result.job; else state.jobs.unshift(result.job);
        renderRuntime();
        return;
      }
      toast("Background job completed", result.draft_id ? "The atom draft is ready to resume." : "The evaluation is available in history.");
      await Promise.all([loadCatalog(), loadRuns()]);
    } catch (error) {
      toast("Background job stopped", error.message);
    } finally {
      state.pollingJobId = null;
    }
  }

  async function resumeAtomDraft(draftId) {
    try {
      const payload = await api(`/api/evaluations/atom-drafts/${encodeURIComponent(draftId)}`);
      if (payload.draft.status !== "open") throw new Error("This atom draft was already saved.");
      state.draft = payload.draft;
      openAtomEditor();
    } catch (error) { toast("Could not resume atom review", error.message); }
  }

  async function generateAtoms() {
    if (!state.selectedDataset) return;
    try {
      const payload = await api(`/api/evaluations/datasets/${encodeURIComponent(state.selectedDataset.id)}/generate-atoms`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile_id: $("#atom-profile").value,
          static_only: Boolean($("#static-only-generation")?.checked)
        })
      });
      state.jobs.unshift(payload.job);
      renderRuntime();
      toast("Atom generation started", "The provider job is running in the background.");
      const result = await pollJob(payload.job.id);
      const draftPayload = await api(`/api/evaluations/atom-drafts/${encodeURIComponent(result.draft_id)}`);
      state.draft = draftPayload.draft;
      openAtomEditor();
    } catch (error) { toast("Could not generate atoms", error.message); }
  }

  async function openLiveRefreshJob(job) {
    state.jobs.unshift(job); renderRuntime();
    toast("Live refresh started", "A new immutable review draft is being prepared.");
    const result = await pollJob(job.id);
    const draftPayload = await api(`/api/evaluations/atom-drafts/${encodeURIComponent(result.draft_id)}`);
    state.draft = draftPayload.draft;
    openAtomEditor();
  }

  async function refreshLiveSnapshot(datasetId) {
    try {
      const payload = await api(`/api/evaluations/datasets/${encodeURIComponent(datasetId)}/refresh-live`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_id: "builtin" })
      });
      await openLiveRefreshJob(payload.job);
    } catch (error) { toast("Could not refresh live snapshot", error.message); }
  }

  async function reviewAtoms() {
    if (!state.selectedDataset) return;
    try {
      const payload = await api(`/api/evaluations/datasets/${encodeURIComponent(state.selectedDataset.id)}/review-atoms`, {
        method: "POST"
      });
      state.draft = payload.draft;
      openAtomEditor();
    } catch (error) { toast("Could not review atoms", error.message); }
  }

  function openAtomEditor() {
    const eligible = eligibleDraftItems();
    const failed = eligible.filter((item) => item.status === "preparation_failed");
    const isNewDraft = $("#atom-editor").dataset.draftId !== state.draft.id;
    $("#atom-dialog-title").textContent = `Review atoms · ${state.draft.dataset_name}`;
    $("#retry-failed-atoms").hidden = failed.length === 0;
    $("#retry-failed-atoms").disabled = false;
    $("#retry-failed-atoms").textContent = `Retry failed atoms (${failed.length})`;
    const existingScopeAction = $("#switch-draft-static-only");
    existingScopeAction?.remove();
    if (state.draft.dataset_role === "definition_parent" &&
        state.draft.generation_scope === "complete" &&
        failed.some((item) => item.time_sensitive)) {
      const action = document.createElement("button");
      action.id = "switch-draft-static-only";
      action.type = "button";
      action.className = "button secondary";
      action.textContent = "Switch draft to static-only";
      $("#retry-failed-atoms").after(action);
      action.addEventListener("click", switchDraftStaticOnly);
    }
    if (isNewDraft) {
      state.reviewValidationActive = false;
      $("#reviewed-dataset-name").value = `${state.draft.dataset_name} · reviewed`;
      eligible.forEach((item, itemIndex) => { item.review_open = itemIndex === 0; });
    }
    $("#atom-editor").dataset.draftId = state.draft.id;
    $("#atom-editor").innerHTML = eligible.map((item, itemIndex) => {
      const atoms = item.atoms || [];
      const requiredCount = atoms.filter((atom) => atom.required).length;
      return `<details class="atom-item" data-item-index="${itemIndex}" ${item.review_open ? "open" : ""}>
      <summary>
        <span class="atom-summary-copy"><code>${esc(item.item_id)} · ${esc(item.status.replaceAll("_", " "))}</code><strong>${esc(item.question)}</strong></span>
        <span class="atom-summary-count" data-atom-summary-count>${atoms.length} atom${atoms.length === 1 ? "" : "s"} · ${requiredCount} required</span>
      </summary>
      <div class="atom-item-body">
        <section class="expected-answer" aria-labelledby="expected-answer-${itemIndex}">
          <p class="field-label" id="expected-answer-${itemIndex}">${item.time_sensitive ? "Resolved answer" : "Expected answer"}</p>
          <div>${esc(typeof item.answer === "string" ? item.answer : JSON.stringify(item.answer, null, 2))}</div>
          ${item.answer_source ? `<small class="expected-answer-source">Source: ${esc(item.answer_source)}</small>` : ""}
        </section>
        ${item.time_sensitive ? `<section class="live-oracle-evidence">
          <p class="live-answer-label">Live answer</p>
          <details><summary>Oracle recipe and evidence</summary><pre>${esc(JSON.stringify({ oracle: item.oracle, metadata: item.oracle_metadata, calls: item.oracle_calls }, null, 2))}</pre></details>
        </section>` : ""}
        ${item.live_state && item.live_state !== "resolved" ? `<div class="live-state"><strong>Live state:</strong> ${esc(item.live_state.replaceAll("_", " "))}</div>` : ""}
        ${item.previous_answer ? `<details class="previous-live-answer"><summary>Previous approved answer</summary><pre>${esc(JSON.stringify(item.previous_answer, null, 2))}</pre></details>` : ""}
        ${item.error ? `<div class="generation-error"><strong>Generation failed:</strong> ${esc(item.error)}</div>` : ""}
        ${item.status === "preparation_failed" ? `<div class="generation-resolution">${item.time_sensitive ? "Live answer could not be resolved. Retry failed questions or create a static-only child." : "Atom generation failed. Retry failed questions before approval."}</div>` : `<section class="atoms" aria-labelledby="atoms-title-${itemIndex}">
          <div class="atoms-heading"><div><p class="field-label" id="atoms-title-${itemIndex}">Atoms</p><span>Edit each obligation and choose whether it is required.</span></div></div>
          ${atoms.map((atom, atomIndex) => atomRow(itemIndex, atomIndex, atom, item.item_id)).join("")}
        ${atoms.length ? "" : `<div class="atoms-empty">No atoms yet for this answer.</div>`}
        <button class="add-atom" type="button" data-add-atom="${itemIndex}" aria-label="Add atom to ${esc(item.item_id)}">+ Add atom</button>
        </section>`}
      </div>
    </details>`;
    }).join("");
    bindAtomEditor();
    if (state.reviewValidationActive) validateReviewedItems(false);
    if (!$("#atom-dialog").open) $("#atom-dialog").showModal();
  }

  async function retryFailedAtoms() {
    if (!state.draft) return;
    syncAtomsFromDom();
    const retryIds = new Set(
      state.draft.items
        .filter((item) => item.status === "preparation_failed")
        .map((item) => item.item_id)
    );
    if (!retryIds.size) return;
    const localById = new Map(state.draft.items.map((item) => [
      item.item_id,
      {
        atoms: (item.atoms || []).map((atom) => ({ ...atom })),
        review_open: item.review_open
      }
    ]));
    const button = $("#retry-failed-atoms");
    button.disabled = true;
    button.textContent = `Retrying ${retryIds.size} failed atom${retryIds.size === 1 ? "" : "s"}…`;
    try {
      const payload = await api(`/api/evaluations/atom-drafts/${encodeURIComponent(state.draft.id)}/retry-failed`, {
        method: "POST"
      });
      state.jobs.unshift(payload.job);
      renderRuntime();
      const result = await pollJob(payload.job.id);
      syncAtomsFromDom();
      state.draft.items.forEach((item) => {
        localById.set(item.item_id, {
          atoms: (item.atoms || []).map((atom) => ({ ...atom })),
          review_open: item.review_open
        });
      });
      const draftPayload = await api(`/api/evaluations/atom-drafts/${encodeURIComponent(result.draft_id)}`);
      draftPayload.draft.items = draftPayload.draft.items.map((item) => {
        const local = localById.get(item.item_id);
        if (!local) return item;
        if (retryIds.has(item.item_id)) return { ...item, review_open: local.review_open };
        return { ...item, atoms: local.atoms, review_open: local.review_open };
      });
      state.draft = draftPayload.draft;
      openAtomEditor();
      const remaining = state.draft.items.filter((item) => item.status === "preparation_failed").length;
      toast(
        remaining ? "Atom retry completed with failures" : "Failed atoms recovered",
        remaining ? `${remaining} item${remaining === 1 ? "" : "s"} still need attention.` : "Generated candidates are ready for review."
      );
    } catch (error) {
      button.disabled = false;
      button.textContent = `Retry failed atoms (${retryIds.size})`;
      toast("Could not retry failed atoms", error.message);
    }
  }

  async function switchDraftStaticOnly() {
    if (!state.draft) return;
    try {
      const omittedCount = state.draft.items.filter((item) => item.time_sensitive && item.status !== "skipped_live").length;
      const payload = await api(`/api/evaluations/atom-drafts/${encodeURIComponent(state.draft.id)}/static-only`, {
        method: "POST"
      });
      state.draft = payload.draft;
      openAtomEditor();
      toast("Draft is now static-only", `${omittedCount} live question${omittedCount === 1 ? " was" : "s were"} omitted by static-only generation. The definition parent was not changed.`);
    } catch (error) {
      toast("Could not change draft scope", error.message);
    }
  }

  function atomRow(itemIndex, atomIndex, atom, itemId) {
    const ordinal = atomIndex + 1;
    return `<div class="atom-row" data-atom-row="${itemIndex}:${atomIndex}">
      <label class="atom-field atom-id"><span>Atom ID</span><input name="atom-${itemIndex}-${atomIndex}-id" data-field="id" value="${esc(atom.id)}" aria-label="Atom ID ${ordinal} for ${esc(itemId)}"></label>
      <label class="atom-field atom-text"><span>Atom text</span><textarea name="atom-${itemIndex}-${atomIndex}-text" data-field="text" rows="3" aria-label="Atom text ${ordinal} for ${esc(itemId)}">${esc(atom.text)}</textarea></label>
      <label class="required-toggle"><input name="atom-${itemIndex}-${atomIndex}-required" type="checkbox" data-field="required" ${atom.required ? "checked" : ""} aria-label="Required for atom ${ordinal} in ${esc(itemId)}"><span>Required</span></label>
      <button type="button" data-remove-atom="${itemIndex}:${atomIndex}" aria-label="Remove atom ${ordinal} from ${esc(itemId)}">Remove</button>
    </div>`;
  }

  function eligibleDraftItems() {
    return state.draft.items.filter((item) =>
      item.status === "prepared" ||
      item.status === "preparation_failed" ||
      (state.draft.dataset_role !== "definition_parent" && !item.time_sensitive)
    );
  }

  function syncAtomsFromDom() {
    eligibleDraftItems().forEach((item, itemIndex) => {
      const panel = $(`[data-item-index="${itemIndex}"]`);
      if (panel) item.review_open = panel.open;
      item.atoms = $$(`[data-atom-row^="${itemIndex}:"]`).map((row) => ({
        id: row.querySelector('[data-field="id"]').value,
        text: row.querySelector('[data-field="text"]').value,
        required: row.querySelector('[data-field="required"]').checked
      }));
    });
  }

  function clearReviewError(panel) {
    panel.querySelector(".review-validation-error")?.remove();
    panel.querySelectorAll("[aria-invalid]").forEach((field) => {
      field.removeAttribute("aria-invalid");
      field.removeAttribute("aria-describedby");
    });
  }

  function validateReviewedItems(focusInvalid = true) {
    let firstInvalid = null;
    eligibleDraftItems().forEach((item, itemIndex) => {
      const panel = $(`[data-item-index="${itemIndex}"]`);
      clearReviewError(panel);
      const rows = [...panel.querySelectorAll("[data-atom-row]")];
      const errors = [];
      const errorId = `atom-review-error-${itemIndex}`;
      const invalidFields = [];
      const markInvalid = (field) => {
        field.setAttribute("aria-invalid", "true");
        field.setAttribute("aria-describedby", errorId);
        invalidFields.push(field);
      };

      if (item.status === "preparation_failed") {
        errors.push(item.time_sensitive
          ? "Live answer could not be resolved. Retry failed questions or create a static-only child."
          : "Atom generation failed. Retry failed questions before approval.");
      }

      if (item.status !== "preparation_failed" && !rows.length) errors.push("Add at least one atom.");
      const ids = new Map();
      rows.forEach((row) => {
        const idField = row.querySelector('[data-field="id"]');
        const textField = row.querySelector('[data-field="text"]');
        const id = idField.value.trim();
        if (!id) {
          if (!errors.includes("Every atom needs an ID.")) errors.push("Every atom needs an ID.");
          markInvalid(idField);
        } else if (ids.has(id)) {
          if (!errors.includes("Atom IDs must be unique within this question.")) errors.push("Atom IDs must be unique within this question.");
          markInvalid(ids.get(id));
          markInvalid(idField);
        } else {
          ids.set(id, idField);
        }
        if (!textField.value.trim()) {
          if (!errors.includes("Every atom needs text.")) errors.push("Every atom needs text.");
          markInvalid(textField);
        }
      });
      if (rows.length && !rows.some((row) => row.querySelector('[data-field="required"]').checked)) {
        errors.push("At least one atom must be required.");
        rows.forEach((row) => markInvalid(row.querySelector('[data-field="required"]')));
      }
      if (!errors.length) return;

      item.review_open = true;
      panel.open = true;
      const error = document.createElement("div");
      error.id = errorId;
      error.className = "review-validation-error";
      error.setAttribute("role", "alert");
      error.textContent = errors.join(" ");
      panel.querySelector(".atom-item-body").prepend(error);
      firstInvalid ||= invalidFields[0] || panel.querySelector("[data-add-atom]") || $("#retry-failed-atoms");
    });
    state.reviewValidationActive = firstInvalid !== null;
    if (firstInvalid && focusInvalid) {
      firstInvalid.closest("[data-item-index]").scrollIntoView({ block: "nearest" });
      firstInvalid.focus();
    }
    return firstInvalid === null;
  }

  function validateReviewedDatasetName(focusInvalid = true) {
    const field = $("#reviewed-dataset-name");
    const errorId = "reviewed-dataset-name-error";
    $(`#${errorId}`)?.remove();
    field.removeAttribute("aria-invalid");
    field.removeAttribute("aria-describedby");
    if (field.value.trim()) return true;
    const error = document.createElement("span");
    error.id = errorId;
    error.className = "field-validation-error";
    error.setAttribute("role", "alert");
    error.textContent = "Enter a name for the reviewed dataset.";
    field.setAttribute("aria-invalid", "true");
    field.setAttribute("aria-describedby", errorId);
    field.after(error);
    if (focusInvalid) field.focus();
    return false;
  }

  function bindAtomEditor() {
    $$("[data-item-index]").forEach((panel) => panel.addEventListener("toggle", () => {
      eligibleDraftItems()[Number(panel.dataset.itemIndex)].review_open = panel.open;
    }));
    $$('[data-field="required"]').forEach((checkbox) => checkbox.addEventListener("change", () => {
      const panel = checkbox.closest("[data-item-index]");
      const shouldRevalidate = Boolean(panel.querySelector(".review-validation-error"));
      const atoms = panel.querySelectorAll("[data-atom-row]").length;
      const required = panel.querySelectorAll('[data-field="required"]:checked').length;
      panel.querySelector("[data-atom-summary-count]").textContent = `${atoms} atom${atoms === 1 ? "" : "s"} · ${required} required`;
      if (shouldRevalidate) validateReviewedItems(false);
    }));
    $$('[data-field="id"], [data-field="text"]').forEach((field) => field.addEventListener("input", () => {
      const panel = field.closest("[data-item-index]");
      if (panel.querySelector(".review-validation-error")) validateReviewedItems(false);
    }));
    $$("[data-add-atom]").forEach((button) => button.addEventListener("click", () => {
      syncAtomsFromDom();
      const item = eligibleDraftItems()[Number(button.dataset.addAtom)];
      const ids = new Set((item.atoms || []).map((atom) => atom.id));
      let ordinal = 1; while (ids.has(`A${ordinal}`)) ordinal += 1;
      item.atoms = [...(item.atoms || []), { id: `A${ordinal}`, text: "", required: true }];
      item.review_open = true;
      openAtomEditor();
      $(`[data-atom-row="${button.dataset.addAtom}:${item.atoms.length - 1}"] [data-field="text"]`).focus();
    }));
    $$("[data-remove-atom]").forEach((button) => button.addEventListener("click", () => {
      syncAtomsFromDom();
      const [itemIndex, atomIndex] = button.dataset.removeAtom.split(":").map(Number);
      const item = eligibleDraftItems()[itemIndex];
      item.atoms.splice(atomIndex, 1);
      item.review_open = true;
      openAtomEditor();
      const nextAtomIndex = Math.min(atomIndex, item.atoms.length - 1);
      const focusTarget = nextAtomIndex >= 0
        ? $(`[data-atom-row="${itemIndex}:${nextAtomIndex}"] [data-field="text"]`)
        : $(`[data-add-atom="${itemIndex}"]`);
      focusTarget.focus();
    }));
  }

  async function saveReviewedDataset() {
    try {
      syncAtomsFromDom();
      const nameValid = validateReviewedDatasetName(false);
      const atomsValid = validateReviewedItems();
      if (!nameValid || !atomsValid) {
        if (atomsValid) $("#reviewed-dataset-name").focus();
        return;
      }
      $("#reviewed-dataset-name").value = $("#reviewed-dataset-name").value.trim();
      const reviewed_items = eligibleDraftItems().map((item) => ({ item_id: item.item_id, atoms: item.atoms || [] }));
      const payload = await api(`/api/evaluations/atom-drafts/${encodeURIComponent(state.draft.id)}/save`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: $("#reviewed-dataset-name").value, reviewed_items })
      });
      $("#atom-dialog").close();
      toast("Reviewed dataset saved", `${payload.dataset.name} is ready to evaluate.`);
      await loadCatalog();
      selectDataset(payload.dataset.id);
    } catch (error) { toast("Could not save reviewed dataset", error.message); }
  }

  async function submitImport(form, endpoint, noun) {
    try {
      const payload = await api(endpoint, { method: "POST", body: new FormData(form) });
      toast(`${noun} ${payload.created ? "imported" : "already exists"}`, payload[noun.toLowerCase()].name);
      form.reset();
      await loadCatalog();
    } catch (error) { toast(`Could not import ${noun.toLowerCase()}`, error.message); }
  }

  async function launchEvaluation(event) {
    event.preventDefault();
    const body = Object.fromEntries(new FormData(event.currentTarget));
    body.attempts = Number(body.attempts);
    body.run_workers = Number(body.run_workers);
    body.score_workers = Number(body.score_workers);
    try {
      const payload = await api("/api/evaluations/runs", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
      });
      state.jobs.unshift(payload.job); renderRuntime();
      toast("Evaluation queued", "The run continues even if you leave this page.");
      const result = await pollJob(payload.job.id);
      if (result.attention_required) {
        toast("Live answers need attention", "No agent attempts have started.");
        renderRuntime();
        return;
      }
      await loadRuns();
      if (result.canceled) showView("runs"); else await openRun(result.history_id);
    } catch (error) { toast("Could not run evaluation", error.message); }
  }

  function groupQuestionResults(preparedItems, answers, evaluationResults) {
    const groups = [];
    const byId = new Map();
    const resultsByAttempt = new Map((evaluationResults || []).map((result) => [result.attempt_id, result]));
    const pairedResultIds = new Set();
    (preparedItems || []).forEach((prepared) => {
      const group = { itemId: prepared.item_id, prepared, attempts: [] };
      groups.push(group);
      byId.set(prepared.item_id, group);
    });
    (answers || []).forEach((answer) => {
      let group = byId.get(answer.item_id);
      if (!group) {
        group = { itemId: answer.item_id, prepared: {}, attempts: [] };
        groups.push(group);
        byId.set(answer.item_id, group);
      }
      const result = resultsByAttempt.get(answer.attempt_id) || null;
      if (result) pairedResultIds.add(result.attempt_id);
      group.attempts.push({ answer, result });
    });
    (evaluationResults || []).forEach((result) => {
      if (pairedResultIds.has(result.attempt_id)) return;
      let group = byId.get(result.item_id);
      if (!group) {
        group = { itemId: result.item_id, prepared: {}, attempts: [] };
        groups.push(group);
        byId.set(result.item_id, group);
      }
      group.attempts.push({ answer: null, result });
    });
    groups.forEach((group) => group.attempts.sort((left, right) =>
      ((left.answer || left.result)?.ordinal || 0) - ((right.answer || right.result)?.ordinal || 0)
    ));
    return groups;
  }

  function questionScoreSummary(attempts) {
    const scored = attempts
      .map((attempt) => ({ attempt, score: scoreValue(attempt.result?.atom_score) }))
      .filter((entry) => entry.score !== null);
    if (!scored.length) return { average: null, best: null, worst: null };
    return {
      average: scored.reduce((total, entry) => total + entry.score, 0) / scored.length,
      best: scored.reduce((best, entry) => entry.score > best.score ? entry : best),
      worst: scored.reduce((worst, entry) => entry.score < worst.score ? entry : worst)
    };
  }

  function macroMeanScoredAttemptAtomRecall(evaluationResults) {
    const recalls = (evaluationResults || [])
      .filter((result) => result.status === "scored" && Array.isArray(result.judgments) && result.judgments.length)
      .map((result) => result.judgments.filter((judgment) => judgment.outcome === "entailed").length / result.judgments.length);
    if (!recalls.length) return null;
    return recalls.reduce((total, recall) => total + recall, 0) / recalls.length;
  }

  function questionLatencyAttempts(preparedItems, answers) {
    const groups = new Map((preparedItems || []).map((item) => [
      item.item_id,
      { itemId: item.item_id, question: item.question || "Question text unavailable", attempts: [] }
    ]));
    (answers || []).forEach((answer) => {
      if (typeof answer.duration_ms !== "number" || !Number.isFinite(answer.duration_ms) || answer.duration_ms < 0) return;
      if (!groups.has(answer.item_id)) {
        groups.set(answer.item_id, {
          itemId: answer.item_id,
          question: "Question text unavailable",
          attempts: []
        });
      }
      const recordedToolCalls = Array.isArray(answer.tool_calls) ? answer.tool_calls : null;
      const timedToolCalls = recordedToolCalls
        ? answer.tool_calls.filter((call) => typeof call.duration_ms === "number" && Number.isFinite(call.duration_ms) && call.duration_ms >= 0)
        : null;
      groups.get(answer.item_id).attempts.push({
        attemptId: answer.attempt_id,
        ordinal: answer.ordinal,
        total: answer.duration_ms,
        toolTotal: timedToolCalls === null ? null : timedToolCalls.reduce((total, call) => total + call.duration_ms, 0),
        toolCount: recordedToolCalls === null ? null : recordedToolCalls.length,
        timedToolCount: timedToolCalls === null ? null : timedToolCalls.length,
        toolTimingComplete: recordedToolCalls !== null && timedToolCalls.length === recordedToolCalls.length
      });
    });
    return [...groups.values()]
      .filter((group) => group.attempts.length)
      .map((group) => {
        group.attempts.sort((left, right) => (left.ordinal || 0) - (right.ordinal || 0));
        return group;
      });
  }

  function renderLatencyChart(preparedItems, answers) {
    const rows = questionLatencyAttempts(preparedItems, answers);
    const maximumDuration = rows.length
      ? Math.max(...rows.flatMap((row) => row.attempts.map((attempt) => attempt.total)))
      : 0;
    return `<section class="panel latency-panel" aria-labelledby="latency-chart-title" data-maximum-duration="${maximumDuration}">
      <div class="latency-head">
        <div><p class="eyebrow">Execution timing</p><h2 id="latency-chart-title">Latency per question</h2></div>
        <div class="latency-legend" aria-label="Latency legend"><span><i class="tool"></i>Tool calls</span><span><i class="other"></i>Other agent time</span></div>
      </div>
      ${rows.length ? `<div class="latency-histogram" role="list">
        ${rows.map((row) => {
          const first = row.attempts[0];
          return `<article class="latency-item" role="listitem" data-question-id="${esc(row.itemId)}">
            <div class="latency-question"><code>${esc(row.itemId)}</code><span title="${esc(row.question)}">${esc(row.question)}</span></div>
            <label class="latency-attempt"><span>Attempt</span><select name="latency-attempt-${esc(row.itemId)}" aria-label="Attempt for ${esc(row.itemId)}">
              ${row.attempts.map((attempt) => `<option value="${esc(attempt.attemptId)}" data-total="${attempt.total}" data-tool="${attempt.toolTotal === null ? "" : attempt.toolTotal}" data-tool-count="${attempt.toolCount === null ? "" : attempt.toolCount}" data-timed-tool-count="${attempt.timedToolCount === null ? "" : attempt.timedToolCount}" data-tool-timing-complete="${attempt.toolTimingComplete}">Attempt ${esc(attempt.ordinal ?? "—")}</option>`).join("")}
            </select></label>
            <div class="latency-plot">
              <div class="latency-bar" role="img" aria-label="Latency for ${esc(row.itemId)}, attempt ${esc(first.ordinal ?? "—")}">
                <span class="latency-other-segment"></span>
                <span class="latency-tool-segment"></span>
                <span class="latency-unknown-segment"></span>
              </div>
            </div>
            <div class="latency-values">
              <strong class="latency-total-value"></strong>
              <span class="latency-tool-value"></span>
              <span class="latency-other-value"></span>
              <small class="latency-tool-count"></small>
            </div>
          </article>`;
        }).join("")}
      </div>` : `<div class="latency-unavailable"><strong>Latency unavailable for this run.</strong><span>This historical run has no authoritative per-attempt timings.</span></div>`}
    </section>`;
  }

  function bindLatencyChart() {
    const panel = $(".latency-panel");
    if (!panel) return;
    const maximumDuration = Number(panel.dataset.maximumDuration);
    $$(".latency-item").forEach((item) => {
      const select = item.querySelector("select");
      const bar = item.querySelector(".latency-bar");
      const toolSegment = item.querySelector(".latency-tool-segment");
      const otherSegment = item.querySelector(".latency-other-segment");
      const unknownSegment = item.querySelector(".latency-unknown-segment");
      const update = () => {
        const option = select.selectedOptions[0];
        const total = Number(option.dataset.total);
        const toolAvailable = option.dataset.tool !== "";
        const toolTimingComplete = option.dataset.toolTimingComplete === "true";
        const toolTotal = toolAvailable ? Number(option.dataset.tool) : null;
        const representedTool = toolAvailable ? Math.min(toolTotal, total) : 0;
        const other = toolTimingComplete ? Math.max(0, total - representedTool) : null;
        const barHeight = maximumDuration > 0 ? (total / maximumDuration) * 100 : 0;
        const toolHeight = total > 0 ? (representedTool / total) * 100 : 0;
        const otherHeight = toolTimingComplete ? 100 - toolHeight : 0;
        const unknownHeight = toolAvailable && !toolTimingComplete ? 100 - toolHeight : (toolAvailable ? 0 : 100);

        bar.style.height = `${barHeight.toFixed(2)}%`;
        toolSegment.style.height = `${toolHeight.toFixed(2)}%`;
        otherSegment.style.height = `${otherHeight.toFixed(2)}%`;
        unknownSegment.style.height = `${unknownHeight.toFixed(2)}%`;
        item.querySelector(".latency-total-value").textContent = `${formatDuration(total)} total`;
        const timedToolCount = option.dataset.timedToolCount;
        const toolCount = option.dataset.toolCount;
        item.querySelector(".latency-tool-value").textContent = toolAvailable && timedToolCount !== "0"
          ? `${formatDuration(toolTotal)} ${toolTimingComplete ? "tools" : "timed tools"}`
          : "Tool timing unavailable";
        item.querySelector(".latency-other-value").textContent = toolTimingComplete
          ? `${formatDuration(other)} other agent time`
          : (toolAvailable ? "Remaining time unattributed" : "Remaining time unavailable");
        item.querySelector(".latency-tool-count").textContent = toolAvailable
          ? (timedToolCount === toolCount
            ? `${toolCount} tool call${toolCount === "1" ? "" : "s"}`
            : `${timedToolCount} timed of ${toolCount} tool call${toolCount === "1" ? "" : "s"}`)
          : "";
        bar.setAttribute(
          "aria-label",
          `${item.dataset.questionId}, ${option.textContent}: ${formatDuration(total)} total; ${toolTimingComplete ? `${formatDuration(toolTotal)} in tools and ${formatDuration(other)} in other agent time` : (toolAvailable && timedToolCount !== "0" ? `${formatDuration(toolTotal)} in timed tools; remaining time unattributed` : "tool timing unavailable; remaining time unattributed")}`
        );
      };
      select.addEventListener("change", update);
      requestAnimationFrame(update);
    });
  }

  function readableError(error) {
    if (!error) return "";
    if (typeof error === "string") return error;
    if (typeof error.message === "string") return error.message;
    return JSON.stringify(error);
  }

  function renderAtomJudgment(atom, judgment, atomIndex) {
    const outcome = outcomePresentation[judgment?.outcome] || {
      label: judgment?.outcome ? judgment.outcome.replaceAll("_", " ") : "Not judged",
      className: "unscored"
    };
    return `<details class="atom-judgment outcome-${esc(outcome.className)}" open>
      <summary>
        <span class="atom-judgment-title"><code>${esc(atom.id || `Atom ${atomIndex + 1}`)}</code>${atom.required ? `<span class="required-chip">Required</span>` : ""}</span>
        <span class="atom-outcome ${esc(outcome.className)}" aria-label="Evaluator outcome: ${esc(judgment?.outcome || "not judged")}">${esc(outcome.label)}</span>
      </summary>
      <div class="atom-judgment-body">
        <section>
          <p class="field-label">Expected content</p>
          <div class="evidence-copy">${esc(atom.text || "Expected atom content is unavailable.")}</div>
        </section>
        <section class="judgment-copy">
          <p class="field-label">Evaluator judgment</p>
          <p>${esc(judgment?.rationale || "No evaluator judgment is available for this atom.")}</p>
        </section>
      </div>
    </details>`;
  }

  function formatToolTraceText(value) {
    if (typeof value !== "string") return JSON.stringify(value, null, 2);
    const trimmed = value.trim();
    if (!trimmed) return value;
    try {
      const structured = JSON.parse(trimmed);
      if (structured === null || typeof structured !== "object") return value;
    } catch (_error) {
      return value;
    }
    let output = "";
    let depth = 0;
    let inString = false;
    let escaped = false;
    const indentation = () => "  ".repeat(depth);
    for (const char of trimmed) {
      if (inString) {
        output += char;
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === '"') inString = false;
        continue;
      }
      if (char === '"') {
        inString = true;
        output += char;
      } else if (char === "{" || char === "[") {
        depth += 1;
        output += `${char}\n${indentation()}`;
      } else if (char === "}" || char === "]") {
        depth -= 1;
        output = `${output.trimEnd()}\n${indentation()}${char}`;
      } else if (char === ",") {
        output += `,\n${indentation()}`;
      } else if (char === ":") {
        output += ": ";
      } else if (!/\s/.test(char)) {
        output += char;
      }
    }
    return output;
  }

  function hasOwn(object, field) {
    return Object.prototype.hasOwnProperty.call(object, field);
  }

  function renderToolCall(call) {
    const key = `tool-call-${state.nextToolCallKey++}`;
    state.openToolCalls.set(key, call);
    const durationAvailable = typeof call.duration_ms === "number"
      && Number.isFinite(call.duration_ms)
      && call.duration_ms >= 0;

    return `<details class="tool-call-detail status-${esc(call.status)}" data-tool-key="${key}" data-tool-ordinal="${esc(call.ordinal)}">
      <summary>
        <span class="tool-call-title"><code>#${esc(call.ordinal)}</code><strong>${esc(call.name)}</strong></span>
        <span class="tool-call-meta">
          <span class="tool-call-status">${esc(call.status.replaceAll("_", " "))}</span>
          ${durationAvailable ? `<span class="tool-call-duration">${esc(call.duration_ms)} ms</span>` : ""}
        </span>
      </summary>
      <div class="tool-call-body" data-tool-call-body></div>
    </details>`;
  }

  function appendToolTraceSection(body, label, value, labelId, className = "") {
    const section = document.createElement("section");
    if (className) section.className = className;
    const heading = document.createElement("p");
    heading.className = "field-label";
    heading.id = labelId;
    heading.textContent = label;
    const content = document.createElement("pre");
    content.tabIndex = 0;
    content.setAttribute("role", "region");
    content.setAttribute("aria-labelledby", labelId);
    content.textContent = formatToolTraceText(value);
    section.append(heading, content);
    body.append(section);
  }

  function hydrateToolCall(detail) {
    const body = detail.querySelector("[data-tool-call-body]");
    if (!body || body.dataset.hydrated === "true") return;
    const call = state.openToolCalls.get(detail.dataset.toolKey);
    if (!call) return;
    const queryAvailable = hasOwn(call, "query");
    const responseField = hasOwn(call, "error") ? "error" : (hasOwn(call, "response") ? "response" : null);
    if (!queryAvailable && responseField === null) {
      const unavailable = document.createElement("p");
      unavailable.className = "tool-call-unavailable";
      unavailable.textContent = "Query and response details were not captured for this historical call.";
      body.append(unavailable);
    } else {
      if (queryAvailable) appendToolTraceSection(
        body,
        "Query",
        call.query,
        `${detail.dataset.toolKey}-query-label`,
      );
      if (responseField) {
        appendToolTraceSection(
          body,
          responseField === "error" ? "Error" : "Response",
          call[responseField],
          `${detail.dataset.toolKey}-${responseField}-label`,
          `tool-call-${responseField}`,
        );
      } else {
        const unavailable = document.createElement("p");
        unavailable.className = "tool-call-unavailable";
        unavailable.textContent = "No tool response was captured for this incomplete call.";
        body.append(unavailable);
      }
    }
    body.dataset.hydrated = "true";
    state.openToolCalls.delete(detail.dataset.toolKey);
  }

  function bindToolCallDetails() {
    $$(".tool-call-detail").forEach((detail) => detail.addEventListener("toggle", () => {
      if (detail.open) hydrateToolCall(detail);
    }));
  }

  function renderToolCalls(answer) {
    if (!Array.isArray(answer.tool_calls)) {
      return `<details class="readable-disclosure tool-call-disclosure">
        <summary><span>Tool calls</span><small>Details unavailable</small></summary>
        <div class="tool-call-empty"><strong>Tool-call details were not captured for this historical attempt.</strong></div>
      </details>`;
    }
    const calls = [...answer.tool_calls].sort((left, right) => (left.ordinal || 0) - (right.ordinal || 0));
    const countLabel = `${calls.length} call${calls.length === 1 ? "" : "s"}`;
    return `<details class="readable-disclosure tool-call-disclosure">
      <summary><span>Tool calls</span><small>${countLabel}</small></summary>
      ${calls.length
        ? `<div class="tool-call-list">${calls.map(renderToolCall).join("")}</div>`
        : `<div class="tool-call-empty"><strong>This attempt performed no recorded tool calls.</strong></div>`}
    </details>`;
  }

  function renderAttempt(attempt, prepared) {
    const answer = attempt.answer || {};
    const result = attempt.result || {};
    const score = scoreValue(result.atom_score);
    const identity = attempt.answer || result;
    const attemptStatus = result.status || answer.status || "unavailable";
    const statusClass = result.status === "scored" ? (result.passed ? "good" : "bad") : (attemptStatus === "execution_failed" ? "bad" : "run");
    const atoms = prepared.gold_atoms || prepared.expected_atoms || [];
    const judgmentsByAtom = new Map((result.judgments || []).map((judgment) => [judgment.atom_id, judgment]));
    const displayedAtoms = atoms.length
      ? atoms
      : (result.judgments || []).map((judgment) => ({ id: judgment.atom_id, text: "", required: false }));
    const modelAnswer = answer.answer || result.answer;
    const exposeLiveFailure = Boolean(result.live_validation);
    return `<details class="attempt-result" data-attempt-id="${esc(identity.attempt_id)}"${exposeLiveFailure ? " open" : ""}>
      <summary>
        <span class="attempt-title">
          <strong>Attempt ${esc(identity.ordinal)}</strong>
          <span class="status ${statusClass}">${esc(attemptStatus)}</span>
        </span>
        <span class="attempt-score ${score === null ? "unscored" : (result.passed ? "passed" : "failed")}">${score === null ? "Not scored" : percent(score)}</span>
      </summary>
      <div class="attempt-body">
        ${modelAnswer ? `<details class="readable-disclosure model-answer" open>
          <summary><span>Model answer</span><small>Full response</small></summary>
          <div class="evidence-copy">${esc(modelAnswer)}</div>
        </details>` : ""}
        ${attempt.answer ? renderToolCalls(answer) : `<div class="empty compact"><strong>No agent answer was created for this live-validation failure.</strong></div>`}
        ${result.live_validation ? `<section class="attempt-error"><p class="field-label">Live validation</p><p>${esc(result.live_validation.reason.replaceAll("_", " "))} · ${esc(result.live_validation.phase.replaceAll("_", " "))}</p><p>${esc(result.live_validation.detail)}</p></section>` : ""}
        ${(result.error || answer.error) ? `<section class="attempt-error"><p class="field-label">Attempt error</p><p>${esc(readableError(result.error || answer.error))}</p></section>` : ""}
        ${displayedAtoms.length ? `<section class="atom-evidence" aria-label="Atom judgments">
          <div class="atom-evidence-head"><div><p class="eyebrow">Atom evidence</p><h3>Expected content and evaluator judgment</h3></div><span>${displayedAtoms.length} atom${displayedAtoms.length === 1 ? "" : "s"}</span></div>
          <div class="atom-judgment-list">${displayedAtoms.map((atom, index) => renderAtomJudgment(atom, judgmentsByAtom.get(atom.id), index)).join("")}</div>
        </section>` : `<div class="empty compact"><strong>No atom judgments are available for this attempt.</strong></div>`}
      </div>
    </details>`;
  }

  function renderQuestionGroup(group) {
    const question = group.prepared.question || "Question text unavailable";
    const score = questionScoreSummary(group.attempts);
    const scoreLabel = score.average === null ? "No scored attempts" : percent(score.average);
    const bestWorst = score.average === null
      ? `${group.attempts.length} attempt${group.attempts.length === 1 ? "" : "s"} · awaiting scores`
      : `Best A${(score.best.attempt.answer || score.best.attempt.result).ordinal} ${percent(score.best.score)} · Worst A${(score.worst.attempt.answer || score.worst.attempt.result).ordinal} ${percent(score.worst.score)}`;
    const exposeLiveFailure = group.attempts.some((attempt) => Boolean(attempt.result?.live_validation));
    return `<details class="question-result" data-question-id="${esc(group.itemId)}"${exposeLiveFailure ? " open" : ""}>
      <summary>
        <span class="question-summary-copy"><span class="question-summary-tags"><code>${esc(group.itemId)}</code>${group.prepared.time_sensitive ? `<span class="live-answer-label">Live answer</span>` : ""}</span><strong>${esc(question)}</strong></span>
        <span class="question-summary-score">
          <span><small>Average score</small><strong>${scoreLabel}</strong></span>
          <meter min="0" max="1" value="${score.average ?? 0}" aria-label="Average atom score for ${esc(group.itemId)}" aria-valuetext="${esc(scoreLabel)}">${scoreLabel}</meter>
          <small>${esc(bestWorst)}</small>
        </span>
      </summary>
      <div class="question-result-body">
        <details class="readable-disclosure user-question" open>
          <summary><span>User question</span><small>Full prompt</small></summary>
          <div class="evidence-copy">${esc(question)}</div>
        </details>
        <div class="attempt-list">
          ${group.attempts.map((attempt) => renderAttempt(attempt, group.prepared)).join("") || `<div class="empty compact"><strong>No attempts are available for this question.</strong></div>`}
        </div>
      </div>
    </details>`;
  }

  async function openRun(id) {
    try {
      const { run } = await api(`/api/evaluations/runs/${encodeURIComponent(id)}`);
      const meta = run.metadata || {}, manifest = run.manifest || {}, summary = run.summary || {};
      state.openRunId = id;
      $("#run-title").textContent = meta.name || manifest.run_id || "Evaluation run";
      $("#run-subtitle").textContent = `${meta.dataset_name || "CLI dataset snapshot"} · ${meta.agent_spec || manifest.agent?.agent_class || "resolved agent"}`;
      $("#report-link").href = `/api/evaluations/runs/${encodeURIComponent(id)}/report`;
      $("#report-link").hidden = !run.report_available;
      if (manifest.status === "canceled") {
        $("#retry-failed-evaluation").hidden = true;
        $("#run-detail-content").innerHTML = `<section class="panel empty"><strong>This evaluation was canceled.</strong><span>No score or report was produced. Canceled ${esc(when(run.cancellation?.canceled_at))}.</span></section>`;
        showView("run-detail");
        return;
      }
      const counts = summary.attempt_lifecycle_counts || {};
      const liveValidationFailed = counts.live_validation_failed || 0;
      const retryableCount = (counts.execution_failed || 0) + (counts.evaluation_failed || 0) + liveValidationFailed;
      const retryButton = $("#retry-failed-evaluation");
      retryButton.hidden = retryableCount === 0 || run.capabilities?.retry_failed !== true;
      retryButton.disabled = false;
      retryButton.textContent = `Retry failed attempts (${retryableCount})`;
      const parent = state.runs.find((item) => item.id === meta.retry_of_history_id);
      state.openToolCalls = new Map();
      state.nextToolCallKey = 1;
      const questionGroups = groupQuestionResults(run.prepared_items || [], run.answers || [], run.evaluation_results || []);
      const atomRecall = macroMeanScoredAttemptAtomRecall(run.evaluation_results);
      $("#run-detail-content").innerHTML = `
        ${meta.retry_of_history_id ? `<div class="lineage-callout"><strong>Successor run ${esc(meta.retry_number || "")}</strong> · retried from ${esc(parent?.name || meta.retry_of_history_id)}. Scored attempts were carried forward unchanged.</div>` : ""}
        ${renderLatencyChart(run.prepared_items || [], run.answers || [])}
        <div class="evidence-grid">
          <article class="evidence-card"><span>Overall pass rate</span><strong>${percent(summary.overall_attempt_pass_rate)}</strong><small>${summary.passed_attempts ?? 0} / ${summary.quality_accounted_attempts ?? 0} accounted</small></article>
          <article class="evidence-card">
            <div class="metric-label">
              <span>Atoms recall</span>
              <button class="metric-info" type="button" aria-label="About atoms recall" aria-describedby="atoms-recall-help">i</button>
              <span class="metric-tooltip" id="atoms-recall-help" role="tooltip">
                For each scored attempt: all atoms marked as entailed ÷ all atoms, including required and optional atoms. This card shows the unweighted average across scored attempts; technical failures are excluded. 100% means every expected fact was covered, and 90% or more is strong overall. Unlike required atoms recall, missing an optional atom lowers this metric but does not by itself fail the attempt. Not-mentioned and contradicted atoms both count as not recalled; contradictions also reduce the separate atom score.
              </span>
            </div>
            <strong>${percent(atomRecall)}</strong>
            <small>macro mean, scored only</small>
          </article>
          <article class="evidence-card">
            <div class="metric-label">
              <span>Required atoms recall</span>
              <button class="metric-info" type="button" aria-label="About required atoms recall" aria-describedby="required-atoms-recall-help">i</button>
              <span class="metric-tooltip" id="required-atoms-recall-help" role="tooltip">
                For each scored attempt: required atoms marked as entailed ÷ all required atoms. This card shows the unweighted average across scored attempts; optional atoms and technical failures are excluded. Aim for 100% because every required atom is a pass condition. 90% or more is strong overall, while anything below 100% means at least one required fact was missed—inspect the attempt evidence to see whether misses are isolated or recurring.
              </span>
            </div>
            <strong>${percent(summary.macro_mean_scored_attempt_required_atom_recall)}</strong>
            <small>macro mean, scored only</small>
          </article>
          <article class="evidence-card"><span>Artifact schema</span><strong>${esc(manifest.schema_version || "—")}</strong><small>${esc(manifest.status || "unknown")}</small></article>
          ${manifest.schema_version === "qa-v2" ? `<article class="evidence-card"><span>Truth source</span><strong>${manifest.contains_live_answers ? "Includes live answers" : "Static only"}</strong><small>source type, not a freshness guarantee</small></article>` : ""}
          <article class="evidence-card"><span>Live validation failed</span><strong>${liveValidationFailed}</strong><small>excluded from quality and technical-failure rates</small></article>
        </div>
        <section class="panel"><div class="panel-head"><div><p class="eyebrow">Attempt evidence</p><h2>Answers and judgments</h2></div></div>
          <div class="result-list">${questionGroups.map((group) => renderQuestionGroup(group)).join("") || `<div class="empty"><strong>No question results are available.</strong></div>`}</div>
        </section>`;
      bindLatencyChart();
      bindToolCallDetails();
      showView("run-detail");
    } catch (error) { toast("Could not open run", error.message); }
  }

  async function retryFailedEvaluation() {
    if (!state.openRunId) return;
    const button = $("#retry-failed-evaluation");
    const idleLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Retrying failed attempts…";
    try {
      const payload = await api(`/api/evaluations/runs/${encodeURIComponent(state.openRunId)}/retry-failed`, {
        method: "POST"
      });
      state.jobs.unshift(payload.job);
      renderRuntime();
      toast("Evaluation retry queued", "Only technically failed attempts will invoke providers.");
      const result = await pollJob(payload.job.id);
      await loadRuns();
      if (result.canceled) showView("runs"); else await openRun(result.history_id);
    } catch (error) {
      button.disabled = false;
      button.textContent = idleLabel;
      toast("Could not retry evaluation", error.message);
    }
  }

  async function continueAttentionEvaluation() {
    const job = activeJob();
    if (!job || job.status !== "attention_required") return;
    const button = $("#continue-evaluation");
    button.disabled = true;
    try {
      const payload = await api(`/api/evaluations/jobs/${encodeURIComponent(job.id)}/continue`, { method: "POST" });
      const index = state.jobs.findIndex((item) => item.id === job.id);
      if (index >= 0) state.jobs[index] = payload.job;
      renderRuntime();
      toast("Live answers are being checked again", "Matching questions will start only after the fresh pre-check completes.");
      const result = await pollJob(job.id);
      if (result.attention_required) {
        toast("Live answers still need attention", "No agent attempts have started.");
        return;
      }
      await Promise.all([loadCatalog(), loadRuns()]);
      await openRun(result.history_id);
    } catch (error) {
      button.disabled = false;
      toast("Could not continue evaluation", error.message);
    }
  }

  async function refreshAttentionEvaluation() {
    const job = activeJob();
    if (!job || job.status !== "attention_required") return;
    const button = $("#refresh-live-evaluation");
    button.disabled = true;
    try {
      const payload = await api(`/api/evaluations/jobs/${encodeURIComponent(job.id)}/refresh-live`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_id: "builtin" })
      });
      const index = state.jobs.findIndex((item) => item.id === job.id);
      if (index >= 0) state.jobs[index] = payload.closed_evaluation.job;
      renderRuntime();
      await openLiveRefreshJob(payload.job);
    } catch (error) {
      button.disabled = false;
      toast("Could not refresh live snapshot", error.message);
    }
  }

  $$(".nav-item").forEach((item) => item.addEventListener("click", () => showView(item.dataset.view)));
  $$("[data-go]").forEach((item) => item.addEventListener("click", () => showView(item.dataset.go)));
  $("#dataset-search").addEventListener("input", renderDatasets);
  $("#launch-profile").addEventListener("change", renderProfileSnapshot);
  $("#launch-agent").addEventListener("change", renderAgentSnapshot);
  $("#refresh-runs").addEventListener("click", () => loadRuns().catch((error) => toast("Refresh failed", error.message)));
  $("#history-window").addEventListener("change", async (event) => {
    state.historyRange = event.currentTarget.value;
    try {
      await loadRuns();
    } catch (error) {
      toast("History could not load", error.message);
    }
  });
  $("#trend-select-all").addEventListener("click", () => {
    state.selectedTrendDatasets = new Set(state.trendDatasetKeys);
    $("#trend-dataset-filters").querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = true; });
    updateTrendFilterStatus();
    renderTrendCharts();
  });
  $("#dataset-import-form").addEventListener("submit", (event) => { event.preventDefault(); submitImport(event.currentTarget, "/api/evaluations/datasets", "Dataset"); });
  $("#profile-import-form").addEventListener("submit", (event) => { event.preventDefault(); submitImport(event.currentTarget, "/api/evaluations/profiles", "Profile"); });
  $("#evaluation-form").addEventListener("submit", launchEvaluation);
  $("#retry-failed-atoms").addEventListener("click", retryFailedAtoms);
  $("#retry-failed-evaluation").addEventListener("click", retryFailedEvaluation);
  $("#continue-evaluation").addEventListener("click", continueAttentionEvaluation);
  $("#refresh-live-evaluation").addEventListener("click", refreshAttentionEvaluation);
  $("#cancel-evaluation").addEventListener("click", async () => {
    const job = activeJob();
    if (!job || job.kind !== "evaluation") return;
    if (!window.confirm("Cancel this evaluation? Local evaluation work will stop, but requests already accepted by a remote provider cannot be recalled.")) return;
    job.status = "cancel_requested";
    renderRuntime();
    try {
      const payload = await api(`/api/evaluations/jobs/${encodeURIComponent(job.id)}/cancel`, { method: "POST" });
      const index = state.jobs.findIndex((item) => item.id === payload.job.id);
      if (index >= 0) state.jobs[index] = payload.job;
      renderRuntime();
      toast("Evaluation canceled", "The stopped run is recorded in history.");
      await Promise.all([loadCatalog(), loadRuns()]);
      showView("runs");
    } catch (error) {
      toast("Could not cancel evaluation", error.message);
      await loadCatalog();
    }
  });
  $("#save-reviewed-dataset").addEventListener("click", saveReviewedDataset);
  $("#reviewed-dataset-name").addEventListener("input", () => {
    if ($("#reviewed-dataset-name-error")) validateReviewedDatasetName(false);
  });
  $$("[data-close-atom-dialog]").forEach((button) => button.addEventListener("click", () => $("#atom-dialog").close("cancel")));

  Promise.all([loadCatalog(), loadRuns()]).catch((error) => toast("Console could not load", error.message));
})();
