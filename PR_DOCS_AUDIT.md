# Comprehensive Documentation Audit & Restructure

## Summary

Complete audit, restructure, and accuracy verification of all project documentation. The monolithic `user_guide.md` (1,190 lines) has been decomposed into 15 focused pages, every factual claim has been cross-referenced against the source code, and 17 accuracy issues have been corrected.

**16 files changed — 2,340 insertions, 1,689 deletions across 2 commits.**

## Motivation

The existing docs had several problems:
- `user_guide.md` was a single 1,190-line monolith covering everything from CLI usage to provider config to agent specs
- `api_reference.md` contained fabricated endpoints not present in the codebase
- Config examples used legacy keys (`provider`/`model`) that `config_manager.py` explicitly rejects with a `ValueError`
- Outdated org references (`mit-submit` instead of `archi-physics`)
- Missing pages for common topics (troubleshooting, benchmarking, CLI reference, data sources)

## What Changed

### Structure: 7 → 15 pages

| Before | After |
|--------|-------|
| index.md | index.md (streamlined) |
| install.md | install.md |
| quickstart.md | quickstart.md (expanded) |
| user_guide.md (monolith) | user_guide.md (hub page, 185 lines) |
| | → data_sources.md (new) |
| | → services.md (new) |
| | → models_providers.md (new) |
| | → agents_tools.md (new) |
| | → configuration.md (new) |
| | → cli_reference.md (new) |
| api_reference.md | api_reference.md (corrected) |
| | → benchmarking.md (new) |
| advanced_setup_deploy.md | advanced_setup_deploy.md |
| developer_guide.md | developer_guide.md (corrected) |
| | → troubleshooting.md (new) |

### Accuracy Fixes (17 issues found and corrected)

Every claim was verified against the actual source code. Issues found and fixed:

| # | Issue | Source of Truth |
|---|-------|-----------------|
| 1 | `provider`/`model` → `default_provider`/`default_model` | `config_manager.py` raises `ValueError` on legacy keys |
| 2 | `recursion_limit: 100` → `50` | `base_react.py` L232 hardcodes 50 |
| 3 | Removed `chain_update_time` config key | Does not exist anywhere in codebase |
| 4 | Auth routes `/api/auth/login` → `/login`, `/logout`, `/auth/user` | `app.py` L2017-2019 |
| 5 | Agent API routes corrected | `app.py` L1971-1976 |
| 6 | BYOK key hierarchy reversed | Session keys override env keys, not vice versa (`get_provider_with_api_key` bypasses env lookup) |
| 7 | BYOK provider restriction corrected | `app.py` accepts all `ProviderType` values for session keys |
| 8 | OpenRouter default model | `anthropic/claude-sonnet-4` per `openrouter_provider.py` L22 |
| 9 | Uploader is not a separate service | Not in `service_registry.py`; upload is integrated into chatbot at `/upload` |
| 10 | `use_hybrid_search` is dynamic runtime config | Stored in DB, not a YAML config key |
| 11 | Document formats updated | Full set from `loader_utils.py` L27 |
| 12 | `register_provider` is a function, not a decorator | `providers/__init__.py` |
| 13 | `AgentSpec` path and field names | Located at `src/archi/pipelines/agents/agent_spec.py`, field is `source_path` not `file_path` |
| 14 | CLI `--env-file`/`--services` are optional | `cli_main.py` — not marked `required=True` |
| 15 | `num_documents_to_retrieve` config path | Correct path: `retrievers.hybrid_retriever.num_documents_to_retrieve` |
| 16 | GitHub org URL | `mit-submit` → `archi-physics` |
| 17 | Removed hardcoded `version: 1.2.4` from mkdocs config | Stale value |

### Other Improvements

- **Navigation**: Logical grouping from getting-started → configuration → reference → advanced
- **Cross-linking**: Every page links to related pages; user_guide.md serves as a hub
- **Consistent formatting**: All config examples use the same YAML style, all CLI examples use the same flag format
- **`mkdocs build --strict`** passes clean with zero warnings

## New Pages Overview

- **data_sources.md** — Git repos, local files, URLs, Piazza, Mattermost, Redmine; document upload via chat UI; data viewer
- **services.md** — All 7 services from `service_registry.py` with descriptions, ports, required secrets
- **models_providers.md** — All 5 providers (OpenAI, Anthropic, OpenRouter, Gemini, Local/vLLM), embedding models, BYOK
- **agents_tools.md** — Agent specs, tools, prompt customization, vectorstore/retriever config
- **configuration.md** — Complete YAML reference with every config key, organized by section
- **cli_reference.md** — All CLI flags, deployment lifecycle commands, directory structure
- **benchmarking.md** — Benchmark setup, query format, prompt templates, running benchmarks
- **troubleshooting.md** — Common errors, Docker issues, provider problems, database troubleshooting

## Testing

- [x] `mkdocs build --strict` — clean, zero warnings
- [x] All 15 pages render correctly
- [x] All internal cross-links resolve
- [x] Every config example uses correct key names verified against source
- [x] Every API endpoint verified against route registrations in `app.py`

## Commits

1. `fb4289fb` — comprehensive docs audit: restructure into 15 pages, fix accuracy issues
2. `f99f429b` — fix docs accuracy: verify all claims against source code
