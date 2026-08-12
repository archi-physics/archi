import subprocess
import sys
import time

from src.evaluation.qa.artifacts import artifact_hashes, write_json


class RecordingWorkflow:
    def composite(self, **kwargs):
        output_dir = kwargs["output_dir"]
        write_json(
            output_dir / "worker_arguments.json",
            {
                "evaluator_profile_path": (
                    str(kwargs["evaluator_profile_path"])
                    if kwargs["evaluator_profile_path"] is not None
                    else None
                ),
                "run_workers": kwargs["run_workers"],
                "score_workers": kwargs["score_workers"],
            },
        )
        return {
            "run_id": "run-1",
            "status": "scored",
            "artifacts": {},
        }


class RetryWorkflow:
    def retry(self, parent_path, output_dir):
        for name in ("input.snapshot.json", "preparation.jsonl"):
            (output_dir / name).write_bytes((parent_path / name).read_bytes())
        write_json(output_dir / "summary.json", {"overall_attempt_pass_rate": 1.0})
        (output_dir / "report.md").write_text("# Retry report\n")
        return {
            "schema_version": "qa-v1",
            "run_id": "retry-run",
            "status": "scored",
            "input": {"snapshot": "input.snapshot.json"},
            "attempts": 1,
            "phases": {
                "prepare": {"status": "completed", "input_items": 1},
                "run": {"status": "completed"},
                "score": {"status": "completed"},
            },
            "artifacts": artifact_hashes(
                output_dir,
                {
                    "input.snapshot.json",
                    "preparation.jsonl",
                    "summary.json",
                    "report.md",
                },
            ),
        }


class SlowWorkflow:
    def composite(self, **_kwargs):
        time.sleep(30)
        return {"run_id": "too-late", "artifacts": {}}


class DescendantWorkflow:
    def composite(self, **kwargs):
        child_pid_path = kwargs["output_dir"] / "child.pid"
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, signal, sys, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(child_pid_path),
            ]
        )
        while not child_pid_path.is_file():
            time.sleep(0.01)
        time.sleep(30)
        return {"run_id": "too-late", "artifacts": {}}
