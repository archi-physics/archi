# isort: skip_file
"""Test-only subprocess entrypoint for selecting deterministic workflows."""

import os

import src.evaluation.qa.worker as worker

from .worker_support import DescendantWorkflow
from .worker_support import RecordingWorkflow
from .worker_support import RetryWorkflow
from .worker_support import SlowWorkflow

WORKFLOWS = {
    "descendant": DescendantWorkflow,
    "recording": RecordingWorkflow,
    "retry": RetryWorkflow,
    "slow": SlowWorkflow,
}


if __name__ == "__main__":
    worker.QAWorkflow = WORKFLOWS[os.environ["ARCHI_TEST_WORKFLOW"]]
    raise SystemExit(worker.main())
