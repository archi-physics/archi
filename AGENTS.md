<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds core (`src/archi`), CLI (`src/cli`), ingestion (`src/data_manager`), interfaces (`src/interfaces`), and utilities (`src/utils`).
- `tests/` includes `smoke/` and `pr_preview_config/`.
- `docs/` contains the mkdocs site; `requirements/` and `src/cli/templates/dockerfiles/` store base image requirements; `examples/` has sample configs.

## Codebase Map
- CLI entrypoint is `src/cli/cli_main.py`, with registries in `src/cli/service_registry.py` and `src/cli/source_registry.py`, and managers in `src/cli/managers/`.
- Service entrypoints live in `src/bin/` and wire Flask apps from `src/interfaces/`.
- Runtime config is loaded from `/root/archi/configs/` by `src/utils/config_loader.py`; CLI deployments render under `~/.archi/archi-<name>` (override with `Archi_DIR`).
- Core orchestration lives in `src/archi/archi.py` with pipelines in `src/archi/pipelines/`; ingestion is in `src/data_manager/`.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode for local development.
- `archi --help` verifies the CLI entrypoint defined in `pyproject.toml`.
- `cd docs && mkdocs serve` previews documentation locally.

## Coding Style & Naming Conventions
- Python 3.7+; follow PEP 8 with 4-space indentation.
- Use `snake_case` for modules/functions and `PascalCase` for classes; keep filenames descriptive (e.g., `test_interfaces.py`).
- Import ordering is generally maintained with `isort` when formatting is applied.
- Shell scripts under `scripts/` and `tests/smoke/` use `bash` with `set -euo pipefail`.

## Testing Guidelines
- **Unit tests:** Run `pytest tests/unit/ -v --tb=short` (requires project dependencies: `pip install ".[all]"`).
- **UI tests:** Run `npx playwright test` against a running deployment (set `BASE_URL` env var). Install with `npm ci && npx playwright install --with-deps chromium`.
- **Smoke tests:** Run via `scripts/dev/run_smoke_preview.sh <name>`. Requires Ollama with a model pulled, Docker, and the archi CLI.
- **Lint:** Run `black --check .` and `isort --check .` for formatting checks.
- **CI:** All PR checks run on `ubuntu-latest` GitHub runners. PR CI includes lint, unit tests, smoke deployment, and Playwright UI tests.

## Commit & Pull Request Guidelines
- Recent history uses short, lowercase summaries (e.g., `fix bug`, `split data manager...`); keep commits concise and descriptive.
- PRs should include: a brief summary, test results, and documentation impact; link related issues and include screenshots/logs when UI or API changes are involved.

## Agent Workflow
- When changing user-facing behavior, CLI flags, configuration, or public APIs, update the relevant docs in `docs/` and/or `README.md` in the same change.
- If no docs change is needed, note the reason briefly in the PR description or commit message.

## Deployment & Validation Policy
- **Match the real runtime path before debugging:** Verify which code path the running service imports (workspace source vs installed `site-packages`) and patch/reload the active path.
- **Deployment assumptions must be explicit:** State which container/service is being validated (for example `chatbot-debug` and its dependent `postgres-debug` / `data-manager-debug`).
- **Always validate behavior after changes:** Do not stop at code edits. Run at least one end-to-end check against the running deployment and confirm expected outputs in logs/trace/events.
- **Use source-of-truth checks for trace bugs:** Validate both streamed events and persisted DB trace rows (for example `agent_traces.events`) when debugging tool-call rendering mismatches.
- **Iterate until intent is confirmed:** If validation fails or is inconclusive, continue debugging and re-test after each fix until the observed behavior matches the requested goal.
