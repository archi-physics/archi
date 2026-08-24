from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from src.evaluation.qa import QAWorkflow


def _run(action, success_message: str) -> None:
    try:
        manifest = action()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if (
        manifest["status"] == "prepared"
        and manifest["phases"]["prepare"]["prepared_items"] == 0
    ):
        raise click.ClickException(
            "QA preparation produced no usable items; diagnostic artifacts were written"
        )
    click.echo(f"{success_message}: {manifest['run_id']} ({manifest['status']})")


@click.group(name="eval")
def eval_cli() -> None:
    """Run Archi evaluation suites."""


@eval_cli.group(name="qa", invoke_without_command=True)
@click.pass_context
@click.option(
    "--dataset", type=click.Path(path_type=Path), help="JSON or JSONL QA dataset."
)
@click.option(
    "--agent-config",
    type=click.Path(path_type=Path),
    help="Archi agent YAML configuration.",
)
@click.option(
    "--agent-spec", type=click.Path(path_type=Path), help="Archi agent Markdown spec."
)
@click.option(
    "--evaluator-profile",
    "evaluator_profile_path",
    type=click.Path(path_type=Path),
    help="QA evaluator YAML profile.",
)
@click.option(
    "--mcp-config",
    "mcp_config_path",
    type=click.Path(path_type=Path),
    help="Evaluator-only MCP connection registry.",
)
@click.option(
    "--skip-live",
    is_flag=True,
    help="Omit V2 live rows during preparation without calling MCP.",
)
@click.option("--output-dir", type=click.Path(path_type=Path), help="QA run workspace.")
@click.option(
    "--attempts",
    "attempts",
    "-n",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
)
@click.option(
    "--run-workers",
    type=click.IntRange(min=1, max=QAWorkflow.MAX_PHASE_WORKERS),
    default=1,
    show_default=True,
    help="Concurrent tested-agent attempts.",
)
@click.option(
    "--score-workers",
    type=click.IntRange(min=1, max=QAWorkflow.MAX_PHASE_WORKERS),
    default=1,
    show_default=True,
    help="Concurrent evaluator comparisons.",
)
@click.option("--overwrite", is_flag=True, help="Replace evaluator-owned artifacts.")
def qa_cli(
    ctx: click.Context,
    dataset: Optional[Path],
    agent_config: Optional[Path],
    agent_spec: Optional[Path],
    evaluator_profile_path: Optional[Path],
    mcp_config_path: Optional[Path],
    skip_live: bool,
    output_dir: Optional[Path],
    attempts: int,
    run_workers: int,
    score_workers: int,
    overwrite: bool,
) -> None:
    """Evaluate agent answers against hidden canonical answers."""
    if ctx.invoked_subcommand is not None:
        return
    missing = [
        flag
        for flag, value in (
            ("--dataset", dataset),
            ("--agent-config", agent_config),
            ("--agent-spec", agent_spec),
            ("--output-dir", output_dir),
        )
        if value is None
    ]
    if missing:
        raise click.UsageError(
            "the composite QA workflow requires " + ", ".join(missing), ctx=ctx
        )
    workflow = QAWorkflow()
    live_options = {"skip_live": skip_live} if skip_live else {}
    if mcp_config_path is not None:
        live_options["mcp_config_path"] = mcp_config_path
    _run(
        lambda: workflow.composite(
            dataset=dataset,
            agent_config=agent_config,
            agent_spec=agent_spec,
            evaluator_profile_path=evaluator_profile_path,
            output_dir=output_dir,
            attempts=attempts,
            run_workers=run_workers,
            score_workers=score_workers,
            overwrite=overwrite,
            **live_options,
        ),
        "QA evaluation completed",
    )


@qa_cli.command(name="prepare")
@click.argument("dataset", type=click.Path(path_type=Path))
@click.option(
    "--evaluator-profile",
    "evaluator_profile_path",
    type=click.Path(path_type=Path),
    help="QA evaluator YAML profile.",
)
@click.option(
    "--mcp-config",
    "mcp_config_path",
    type=click.Path(path_type=Path),
    help="Evaluator-only MCP connection registry.",
)
@click.option("--skip-live", is_flag=True, help="Omit V2 live rows.")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--overwrite", is_flag=True, help="Replace preparation and downstream artifacts."
)
def prepare_cli(
    dataset: Path,
    evaluator_profile_path: Optional[Path],
    mcp_config_path: Optional[Path],
    skip_live: bool,
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Validate DATASET and prepare fixed gold atoms."""
    workflow = QAWorkflow()
    live_options = {"skip_live": skip_live} if skip_live else {}
    if mcp_config_path is not None:
        live_options["mcp_config_path"] = mcp_config_path
    _run(
        lambda: workflow.prepare(
            dataset,
            output_dir,
            evaluator_profile_path,
            overwrite,
            **live_options,
        ),
        "QA preparation completed",
    )


@qa_cli.command(name="run")
@click.argument("run_dir", type=click.Path(path_type=Path))
@click.option("--agent-config", type=click.Path(path_type=Path), required=True)
@click.option("--agent-spec", type=click.Path(path_type=Path), required=True)
@click.option(
    "--mcp-config",
    "mcp_config_path",
    type=click.Path(path_type=Path),
    help="Evaluator-only MCP connection registry.",
)
@click.option(
    "--attempts",
    "attempts",
    "-n",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
)
@click.option(
    "--run-workers",
    type=click.IntRange(min=1, max=QAWorkflow.MAX_PHASE_WORKERS),
    default=1,
    show_default=True,
    help="Concurrent tested-agent attempts.",
)
@click.option("--overwrite", is_flag=True, help="Replace run and downstream artifacts.")
def run_cli(
    run_dir: Path,
    agent_config: Path,
    agent_spec: Path,
    mcp_config_path: Optional[Path],
    attempts: int,
    run_workers: int,
    overwrite: bool,
) -> None:
    """Run isolated Archi attempts in prepared RUN_DIR."""
    workflow = QAWorkflow()
    live_options = (
        {"mcp_config_path": mcp_config_path} if mcp_config_path is not None else {}
    )
    _run(
        lambda: workflow.run(
            run_dir,
            agent_config,
            agent_spec,
            attempts,
            overwrite,
            run_workers=run_workers,
            **live_options,
        ),
        "QA agent run completed",
    )


@qa_cli.command(name="score")
@click.argument("run_dir", type=click.Path(path_type=Path))
@click.option(
    "--evaluator-profile",
    "evaluator_profile_path",
    type=click.Path(path_type=Path),
    help="Matching QA evaluator profile.",
)
@click.option("--overwrite", is_flag=True, help="Replace score and report artifacts.")
@click.option(
    "--score-workers",
    type=click.IntRange(min=1, max=QAWorkflow.MAX_PHASE_WORKERS),
    default=1,
    show_default=True,
    help="Concurrent evaluator comparisons.",
)
def score_cli(
    run_dir: Path,
    evaluator_profile_path: Optional[Path],
    overwrite: bool,
    score_workers: int,
) -> None:
    """Compare and score terminal answers in RUN_DIR."""
    workflow = QAWorkflow()
    _run(
        lambda: workflow.score(
            run_dir,
            evaluator_profile_path,
            overwrite,
            score_workers=score_workers,
        ),
        "QA scoring completed",
    )
