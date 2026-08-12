from src.archi.pipelines.agents.base_react import BaseReActAgent
from src.archi.utils.output_dataclass import PipelineOutput


def test_invoke_forwards_callbacks_to_agent_graph():
    observed = {}
    callback = object()

    class Graph:
        def invoke(self, inputs, config):
            observed["inputs"] = inputs
            observed["config"] = config
            return object()

    pipeline = BaseReActAgent.__new__(BaseReActAgent)
    pipeline.agent = Graph()
    pipeline._prepare_agent_inputs = lambda **kwargs: {"messages": ["question"]}
    pipeline._recursion_limit = lambda: 17
    pipeline._extract_messages = lambda output: []
    pipeline._metadata_from_agent_output = lambda output: {}
    pipeline._build_output_from_messages = (
        lambda messages, metadata: PipelineOutput(answer="answer")
    )

    output = pipeline.invoke(history=[("User", "question")], callbacks=[callback])

    assert output.answer == "answer"
    assert observed == {
        "inputs": {"messages": ["question"]},
        "config": {"recursion_limit": 17, "callbacks": [callback]},
    }


def test_start_run_memory_replaces_attempt_state():
    pipeline = BaseReActAgent.__new__(BaseReActAgent)
    pipeline._active_memory = None

    first = pipeline.start_run_memory()
    first.note("first attempt")
    second = pipeline.start_run_memory()

    assert first is not second
    assert pipeline.active_memory is second
    assert first.notes == ("first attempt",)
    assert second.notes == ()


def test_refresh_agent_discovers_mcp_tools_once():
    pipeline = BaseReActAgent.__new__(BaseReActAgent)
    pipeline.selected_tool_names = ["mcp"]
    pipeline._static_tools = []
    pipeline._static_middleware = []
    pipeline._mcp_tools = None
    pipeline._active_tools = []
    pipeline._active_middleware = []
    pipeline.agent = None
    builds = []
    graph = object()
    mcp_tool = object()

    def build_mcp_tools():
        builds.append(True)
        return [mcp_tool]

    pipeline._build_mcp_tools = build_mcp_tools
    pipeline._create_agent = lambda tools, middleware: graph

    assert pipeline.refresh_agent() is graph
    assert pipeline.refresh_agent() is graph
    assert builds == [True]
    assert pipeline._mcp_tools == [mcp_tool]
