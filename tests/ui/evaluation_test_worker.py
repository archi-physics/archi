"""Test-only subprocess entrypoint for the evaluation console browser suite."""

import src.evaluation.qa.worker as worker
import src.evaluation.qa.workflow as workflow

from .evaluation_test_server import (  # isort: skip
    FakeAgentRuntime,
    FakeEvaluator,
    FakeWorkflow,
)

if __name__ == "__main__":
    workflow.ArchiAgentRuntime = FakeAgentRuntime
    workflow.LangChainEvaluatorRuntime = lambda _profile: FakeEvaluator()
    worker.QAWorkflow = FakeWorkflow
    raise SystemExit(worker.main())
