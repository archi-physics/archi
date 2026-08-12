"""Test-only subprocess entrypoint for the evaluation console browser suite."""

import src.evaluation.qa.worker as worker

from .evaluation_test_server import FakeWorkflow

if __name__ == "__main__":
    worker.QAWorkflow = FakeWorkflow
    raise SystemExit(worker.main())
