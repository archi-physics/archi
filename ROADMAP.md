# Archi — 3-Month Development Roadmap

**Period:** March 2026 – May 2026
**Primary users:** CMS Computing Operations (comp ops), with expansion to other CMS/CERN teams
**Current state:** ReAct agent with MONIT/OpenSearch, JIRA, Redmine, Git, SSO scraping, MCP support, hybrid search, chat UI with data viewer/uploader, A/B comparison, agent tracing, Grafana monitoring, BYOK providers, benchmarking. Deployed on submit76.

---

## Current Feature Inventory

| Area | What Exists |
|------|------------|
| **Agent pipeline** | `CMSCompOpsAgent` (ReAct), tool registry (file search, metadata search, vector hybrid search, document fetch, MONIT OpenSearch search/aggregation, MCP), agent specs in markdown, configurable prompts |
| **LLM providers** | OpenAI, Anthropic, OpenRouter, Gemini, Local/Ollama/vLLM — BYOK per-session |
| **Data ingestion** | Links (HTTP + SSO), Git repos, JIRA, Redmine, local file upload |
| **Vectorstore** | Postgres + pgvector, hybrid retriever (BM25 + semantic) |
| **Chat UI** | Conversations, streaming, agent activity panel, data viewer, document upload, A/B comparison, reaction feedback, database viewer |
| **Other interfaces** | Grader, Piazza, Mattermost, Redmine mailer |
| **Infra** | CLI-driven Podman deployments, Grafana, PostgreSQL, data-manager service |
| **Testing** | Unit tests, Playwright UI tests, smoke tests, benchmarking harness |

## Key User Scenarios for CMS Comp Ops

1. **Operational troubleshooting** — Shifter or operator asks "why did transfers to T2_US_MIT fail last night?" → agent queries MONIT OpenSearch Rucio events, cross-references transfer documentation, returns actionable answer with evidence.
2. **Documentation lookup** — "What's the procedure for site downtime?" → agent retrieves from ingested CMS TWiki/Git docs, Redmine tickets, JIRA issues.
3. **Ticket triage/drafting** — Operator reads a failing-transfer alert, asks the agent to draft a JIRA/Redmine ticket with relevant context pulled from MONIT and docs.
4. **Shift handoff** — Outgoing shifter summarizes overnight issues; agent can generate a summary from recent conversation history and MONIT queries.
5. **Onboarding** — New comp ops member asks basic questions; agent points to the right docs and explains procedures.
6. **Cross-team adoption** — Other CMS teams (�PPD, trigger, DPG) or CERN experiments deploy their own Archi instance pointed at their own docs/JIRA projects.

---

## Month 1: March 2026 — Foundation & Reliability

**Theme:** Fix critical bugs, stabilize the platform, clean up technical debt so that CMS comp ops has a reliable daily-driver.

### Week 1–2: Critical Fixes

| # | Item | Notes | Related proposal |
|---|------|-------|-----------------|
| 1.1 | **Fix hybrid search (BM25 falls back to semantic-only)** | `HybridRetriever` calls `get_all_documents()` which doesn't exist on `PostgresVectorStore`. Switch to Postgres-native `hybrid_search()`. This is the single highest-impact retrieval quality issue. | `fix-hybrid-search-architecture` |
| 1.2 | **Secure data-manager API token** | Replace hardcoded `api_token: "chicken"` with `read_secret("DM_API_TOKEN")`. Mandatory before any external team uses the system. | `secure-dm-api-token` |
| 1.3 | **Unify model provider layers** | Deprecate legacy `src/archi/models/` (~1500 lines), consolidate onto `src/archi/providers/`. Removes confusion and dead code. | `unify-model-providers` |

### Week 3–4: Code Cleanup & CI

| # | Item | Notes | Related proposal |
|---|------|-------|-----------------|
| 1.4 | **Remove legacy code** | Delete ChromaDB remnants, `langchain_classic` imports, v1/v2 SQL duplication, model aliases (`OpenAIGPT4`, etc.). | `remove-legacy-code` |
| 1.5 | **Complete CI migration to GitHub runners** | Finish Podman→Docker switch, parallel jobs (lint, unit, smoke, Playwright), smaller Ollama model for CI. | `migrate-ci-to-github-runners` |
| 1.6 | **Local dev scripts** | `run_chat_local.sh`, `run_data_manager_local.sh`, `local-dev.yaml.example` so developers can iterate without deploying containers. | `add-local-dev-scripts` |

### Month 1 Deliverable
A clean, tested, correctly-searching Archi that developers can run locally and CI validates automatically. Hybrid search actually works.

---

## Month 2: April 2026 — CMS Comp Ops Feature Depth

**Theme:** Make the agent genuinely useful for daily comp ops work — better tools, better UX, better data management.

### Week 5–6: Agent & Tool Improvements

| # | Item | Notes | Related proposal |
|---|------|-------|-----------------|
| 2.1 | **MONIT OpenSearch skill expansion** | Add skills for `cms_��transfers`, `���site_status`, `����������computing_�accounting` indices beyond Rucio events. Each skill is a markdown file — low cost, high value. | (new) |
| 2.2 | **JIRA tool** | Let the agent search/read JIRA issues at query time (beyond ingested snapshots). Read-only first: search by project, keyword, status. CMSPROD is the primary project. | (new) |
| 2.3 | **Agent activity UX improvements** | Timeline view, elapsed time, thinking indicators between tool calls. Operators need to know what the agent is doing during long MONIT queries. | `improve-agent-activity-ux` |
| 2.4 | **Agent editor refinement** | Fix "New agent" crash, structured form (name, tool toggles, prompt textarea), active agent indicator. Needed so comp ops admins can customize agent behavior. | `refine-agent-editor-ux` |

### Week 7–8: Data Management & Observability

| # | Item | Notes | Related proposal |
|---|------|-------|-----------------|
| 2.5 | **Upload ingestion visibility** | Per-document status tracking (`pending`/`embedding`/`embedded`/`failed`), status badges in data viewer. Operators upload CMS docs and need to know when they're searchable. | `improve-upload-ingestion-visibility` |
| 2.6 | **Per-source dynamic scheduling** | Allow cron schedule changes via UI without restarting data-manager. CMS docs update on different cadences (TWiki daily, JIRA hourly, Git on-push). | `per-source-dynamic-scheduling` |
| 2.7 | **Message feedback UI** | Thumbs up/down on individual messages (backend exists, UI is stubbed). Needed to collect quality signal from comp ops users. | `add-message-feedback-ui` |
| 2.8 | **Grafana dashboards for comp ops** | Pre-built dashboards: query latency, tool usage breakdown, feedback ratios, data freshness per source. | (new) |

### Month 2 Deliverable
Comp ops operators can query MONIT across multiple indices, search live JIRA issues, upload and track docs, and give feedback — all with clear visibility into what the agent is doing.

---

## Month 3: May 2026 — Multi-Team Readiness & Scale

**Theme:** Make Archi deployable and useful for other CMS/CERN teams. Harden for production.

### Week 9–10: Multi-Tenancy & Deployment

| # | Item | Notes |
|---|------|-------|
| 3.1 | **Multi-agent support per deployment** | Allow multiple agent specs in one instance (e.g., comp ops agent + site support agent). Users select agent at conversation start. The agent spec system already supports multiple files in `agents_dir`. |
| 3.2 | **CERN SSO (OIDC) authentication** | Integrate with CERN's Keycloak-based SSO. The OAuth scaffold exists (`authlib` is imported). This is required before any broader CMS rollout. |
| 3.3 | **Deployment templates for CMS teams** | Pre-packaged config templates: `cms-comp-ops`, `cms-offline-computing`, `cms-trigger`, `generic-cern`. Each bundles relevant JIRA projects, TWiki links, MONIT indices, and agent prompts. |
| 3.4 | **Helm chart / Kubernetes deployment option** | CERN runs OpenShift/K8s. Provide a Helm chart alongside the existing Podman-compose for teams that want managed deployment. |

### Week 11–12: Quality, Benchmarking & Docs

| # | Item | Notes | Related proposal |
|---|------|-------|-----------------|
| 3.5 | **CMS comp ops benchmark suite** | Curated set of 50+ real comp ops questions with expected answers/sources. Run benchmarking harness weekly. Use to regress-test agent prompt/tool changes. | (new, builds on existing benchmarking) |
| 3.6 | **Conversation export & audit logging** | Export conversations as JSON/PDF for shift reports. Audit log of all agent actions for compliance. | (new) |
| 3.7 | **Comprehensive docs update** | Reconcile all pending doc proposals: update postgres docs, complete the comprehensive audit, add deployment guides for new teams. | `docs-comprehensive-audit`, `update-postgres-docs` |
| 3.8 | **Rate limiting & usage quotas** | Per-user and per-team rate limits on LLM calls. Prevents runaway API costs when multiple teams share an instance or use BYOK. | (new) |

### Month 3 Deliverable
Other CMS teams can deploy their own Archi instances with CERN SSO, team-specific config templates, and Kubernetes support. Quality is measured via benchmarks, usage is tracked and rate-limited.

---

## Summary Timeline

```
March 2026 (Month 1) — Foundation & Reliability
├── Fix hybrid search (critical retrieval bug)
├── Secure DM API token
├── Unify model providers (~1500 lines removed)
├── Remove legacy code (ChromaDB, v1/v2 SQL)
├── Complete CI on GitHub runners
└── Local dev scripts

April 2026 (Month 2) — CMS Comp Ops Feature Depth
├── MONIT skill expansion (transfers, site status, accounting)
├── Live JIRA query tool
├── Agent activity UX (timeline, thinking indicators)
├── Agent editor fix & refinement
├── Upload ingestion status tracking
├── Dynamic source scheduling
├── Message feedback UI
└── Grafana dashboards for ops

May 2026 (Month 3) — Multi-Team Readiness
├── Multi-agent per deployment
├── CERN SSO (OIDC) integration
├── Team deployment templates
├── Helm chart for Kubernetes/OpenShift
├── CMS comp ops benchmark suite (50+ queries)
├── Conversation export & audit logging
├── Docs update (all pending proposals)
└── Rate limiting & usage quotas
```

## Priority-Ordered Backlog (Parking Lot)

Items that didn't make the 3-month cut but are valuable:

| Item | Rationale |
|------|-----------|
| **Slack/Mattermost bi-directional integration** | Comp ops already uses Mattermost; a bot that answers in-channel would meet users where they are |
| **Automated shift summary generation** | Scheduled agent run that summarizes MONIT anomalies from the past 8 hours |
| **ServiceNow / SNOW connector** | CERN is migrating some ticketing to SNOW |
| **Fine-tuned embedding model** | Train on CMS-specific corpus for better retrieval relevance |
| **Multi-language support** | CERN has non-English speakers; UI localization + multilingual retrieval |
| **Data viewer Phase 2** | File tree navigation, smart content rendering (markdown/code), domain grouping — per `data-viewer-uploader-consolidation` Phase 2 |
| **Prompt management refactor** | Config-specified > deployment prompts > bundled defaults resolution — per `refactor-prompt-management` |

## Success Metrics

| Metric | Target by end of Month 3 |
|--------|-------------------------|
| Daily active comp ops users | 5+ |
| Questions answered per week | 100+ |
| Feedback ratio (thumbs up / total feedback) | >70% |
| Benchmark suite pass rate | >80% on curated CMS queries |
| Mean time to first response | <10s for non-MONIT queries |
| Teams with active deployments | 2+ (comp ops + 1 other) |
| CI pipeline pass rate | >95% on main branch |
