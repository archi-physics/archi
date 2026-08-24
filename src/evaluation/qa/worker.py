from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from .artifacts import read_json, sha256_file, write_json
from .history import EvaluationHistory
from .workflow import QAWorkflow


class WorkerOperation(str, Enum):
    COMPOSITE = "composite"
    CONTINUE = "continue"
    RETRY = "retry"


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    return Path(value)


def execute(request: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("worker request must be an object")
    try:
        operation = WorkerOperation(request["operation"])
    except (KeyError, ValueError) as exc:
        raise ValueError("unsupported worker operation") from exc
    if operation is WorkerOperation.COMPOSITE:
        required = {
            "operation",
            "output_dir",
            "dataset",
            "agent_config",
            "agent_spec",
            "evaluator_profile_path",
            "attempts",
            "run_workers",
            "score_workers",
        }
        optional = {"mcp_config_path", "trusted_dataset", "pause_on_live_mismatch"}
        if not required.issubset(request) or set(request) - required - optional:
            raise ValueError("composite worker request has invalid fields")
    elif operation is WorkerOperation.CONTINUE:
        required = {
            "operation",
            "output_dir",
            "agent_config",
            "agent_spec",
            "attempts",
            "run_workers",
            "score_workers",
            "authorize_staged_invalid",
        }
        optional = {"mcp_config_path"}
        if not required.issubset(request) or set(request) - required - optional:
            raise ValueError("continue worker request has invalid fields")
    else:
        expected = {
            "operation",
            "output_dir",
            "parent_path",
        }
        if not expected.issubset(request) or set(request) - expected - {
            "mcp_config_path"
        }:
            raise ValueError("retry worker request has invalid fields")

    output_dir = _path(request["output_dir"], "output_dir")
    workflow = QAWorkflow()

    if operation is WorkerOperation.COMPOSITE:
        evaluator_profile_path = request["evaluator_profile_path"]
        manifest = workflow.composite(
            dataset=_path(request["dataset"], "dataset"),
            agent_config=_path(request["agent_config"], "agent_config"),
            agent_spec=_path(request["agent_spec"], "agent_spec"),
            output_dir=output_dir,
            evaluator_profile_path=(
                _path(evaluator_profile_path, "evaluator_profile_path")
                if evaluator_profile_path is not None
                else None
            ),
            attempts=request["attempts"],
            run_workers=request["run_workers"],
            score_workers=request["score_workers"],
            overwrite=False,
            mcp_config_path=(
                _path(request["mcp_config_path"], "mcp_config_path")
                if request.get("mcp_config_path") is not None
                else None
            ),
            trusted_dataset=bool(request.get("trusted_dataset", False)),
            pause_on_live_mismatch=bool(request.get("pause_on_live_mismatch", False)),
        )
        result = {
            "run_id": manifest["run_id"],
            "status": manifest.get("status", "scored"),
            "attention_required": manifest.get("attention_required"),
        }
    elif operation is WorkerOperation.CONTINUE:
        if request["authorize_staged_invalid"] is not True:
            raise ValueError("continued run must authorize its staged invalid items")
        manifest = workflow.run(
            output_dir,
            _path(request["agent_config"], "agent_config"),
            _path(request["agent_spec"], "agent_spec"),
            attempts=request["attempts"],
            overwrite=True,
            run_workers=request["run_workers"],
            mcp_config_path=(
                _path(request["mcp_config_path"], "mcp_config_path")
                if request.get("mcp_config_path") is not None
                else None
            ),
            pause_on_live_mismatch=True,
            authorize_staged_invalid=True,
        )
        if manifest["status"] != "attention_required":
            manifest = workflow.score(
                output_dir,
                score_workers=request["score_workers"],
            )
        result = {
            "run_id": manifest["run_id"],
            "status": manifest["status"],
            "attention_required": manifest.get("attention_required"),
        }
    else:
        retry_options = (
            {"mcp_config_path": _path(request["mcp_config_path"], "mcp_config_path")}
            if request.get("mcp_config_path") is not None
            else {}
        )
        manifest = workflow.retry(
            _path(request["parent_path"], "parent_path"),
            output_dir,
            **retry_options,
        )
        result = {"run_id": manifest["run_id"], "status": manifest["status"]}

    metadata_path = output_dir / "console_metadata.json"
    manifest["artifacts"][metadata_path.name] = sha256_file(metadata_path)
    write_json(output_dir / "manifest.json", manifest)
    result["history_id"] = EvaluationHistory(output_dir.parent).id_for_path(output_dir)
    return result


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: worker REQUEST_JSON RESULT_PATH")
    result_path = Path(sys.argv[2])
    try:
        request = json.loads(sys.argv[1])
        result = execute(request)
    except Exception as exc:
        write_json(result_path, {"error": str(exc)})
        return 1
    write_json(result_path, {"result": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
