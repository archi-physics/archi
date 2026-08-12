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
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--overwrite", is_flag=True, help="Replace preparation and downstream artifacts."
)
def prepare_cli(
    dataset: Path,
    evaluator_profile_path: Optional[Path],
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Validate DATASET and prepare fixed gold atoms."""
    workflow = QAWorkflow()
    _run(
        lambda: workflow.prepare(
            dataset, output_dir, evaluator_profile_path, overwrite
        ),
        "QA preparation completed",
    )


@qa_cli.command(name="run")
@click.argument("run_dir", type=click.Path(path_type=Path))
@click.option("--agent-config", type=click.Path(path_type=Path), required=True)
@click.option("--agent-spec", type=click.Path(path_type=Path), required=True)
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
    attempts: int,
    run_workers: int,
    overwrite: bool,
) -> None:
    """Run isolated Archi attempts in prepared RUN_DIR."""
    workflow = QAWorkflow()
    _run(
        lambda: workflow.run(
            run_dir,
            agent_config,
            agent_spec,
            attempts,
            overwrite,
            run_workers=run_workers,
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
