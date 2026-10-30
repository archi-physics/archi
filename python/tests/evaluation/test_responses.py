"""req.r3-responses-api.agent-calls: the agent arms on OpenAI's Responses API. The HTTP layer is a
fake (`requests.post` answers from a script with recorded-shape Responses and chat/completions
bodies), so nothing here reaches a network or costs money."""
import asyncio
import copy
import json

import pytest
import yaml

from archi.bundles.cli import main
from archi.evaluation import agent as agent_mod
from archi.evaluation.agent import (
    ModelError,
    OpenAIChat,
    OpenAIResponses,
    make_model,
    model_api,
    openai_tools,
)
from archi.evaluation.raw_arm import (
    ANSWERED,
    ERROR,
    OUTPUT_LIMIT,
    STOPPED,
    Caps,
    FatalModelError,
    Ledger,
    Metered,
    Price,
    run_try,
)
from evaluation.raw_support import SEARCH, FakeMcp, search_rows, write_corpus
from evaluation.test_raw_arm import QUESTIONS, Endpoint, agent_args, rows

OPENAI = "https://api.openai.com/v1"
LOCAL = "http://127.0.0.1:11434/v1"
PRICE = Price(0.10, 0.50, 0.01)

#: Output items in the shapes the Responses API documents: a reasoning item (opaque encrypted content,
#: since nothing is stored at OpenAI), a function call, a message.
REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA-opaque"}
USAGE = {"input_tokens": 1200, "input_tokens_details": {"cached_tokens": 1000}, "output_tokens": 300,
         "output_tokens_details": {"reasoning_tokens": 250}, "total_tokens": 1500}


def function_call(name, arguments, call_id="call_1"):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name,
            "arguments": json.dumps(arguments), "status": "completed"}


def message(text):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def response(*output, status="completed", usage=USAGE, incomplete=None):
    body = {"id": "resp_1", "object": "response", "status": status, "model": "gpt-x", "output": list(output),
            "usage": usage}
    if incomplete:
        body["incomplete_details"] = {"reason": incomplete}
    return body


class Answer:
    def __init__(self, status, payload, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeOpenAI:
    """`requests.post`: records each request and answers `script(url, body, n)`, a (status, payload)
    pair or a ready Answer."""

    def __init__(self, script):
        self.script, self.requests = script, []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.requests.append((url, copy.deepcopy(json), dict(headers or {})))
        answer = self.script(url, json, len(self.requests))
        return answer if isinstance(answer, Answer) else Answer(*answer)


@pytest.fixture
def server(monkeypatch):
    import requests

    def install(script):
        fake = FakeOpenAI(script)
        monkeypatch.setattr(requests, "post", fake)
        return fake

    return install


def luna(tmp_path, *, cap=1.0, max_output=1000):
    model = OpenAIResponses(OPENAI, "gpt-x", api_key="sk-test", temperature=None, reasoning_effort="medium",
                            max_completion_tokens=max_output)
    return Metered(model, PRICE, Ledger(tmp_path / "ledger.jsonl", cap), max_output, sleep=lambda s: None)


def one_try(model, tools):
    specs = asyncio.run(tools.list_tools())
    return asyncio.run(run_try(model, tools, specs, arm="agent-graph", system="sys", question_id="q1",
                               question="Why do transfers fail?", attempt=1, caps=Caps(min_call_seconds=0)))


def search_then_answer(url, body, n):
    if n == 1:
        return 200, response(REASONING, function_call("search", {"query": "transfer failures"}))
    return 200, response(message("Stuck transfers.\nSources: jira:DEMO-1"))


# --- which API -------------------------------------------------------------------------------


@pytest.mark.parametrize("url, choice, api", [
    (OPENAI, "auto", "responses"), ("https://API.openai.com/v1/", "auto", "responses"),
    (LOCAL, "auto", "chat"), ("http://ollama:11434/v1", "auto", "chat"), ("https://api.example.com/v1", "auto", "chat"),
    (OPENAI, "chat", "chat"), (LOCAL, "responses", "responses"),
])
def test_auto_picks_the_responses_api_for_openai_only(url, choice, api):
    assert model_api(url, choice) == api
    assert type(make_model(url, "m", api=choice)) is (OpenAIResponses if api == "responses" else OpenAIChat)


def test_an_unknown_api_is_refused():
    with pytest.raises(ModelError, match="auto, chat, responses"):
        model_api(OPENAI, "completions")


# --- the request and the round trip ----------------------------------------------------------


def test_a_tool_call_round_trips_with_its_reasoning_sent_back(tmp_path, server):
    fake = server(search_then_answer)
    tools = FakeMcp({"transfer failures": search_rows("jira:DEMO-1")})
    result = one_try(luna(tmp_path), tools)
    assert result.status == ANSWERED and result.answer.startswith("Stuck transfers")
    assert result.tool_calls == 1 and tools.calls[0][:2] == ("search", {"query": "transfer failures"})

    (url1, first, headers), (url2, second, _) = fake.requests
    assert url1 == url2 == OPENAI + "/responses" and headers["Authorization"] == "Bearer sk-test"
    assert first["model"] == "gpt-x" and first["store"] is False
    assert first["reasoning"] == {"effort": "medium"} and first["max_output_tokens"] == 1000
    # none of the chat/completions fields, and no sampling fields for a remote reasoning model
    assert not {"messages", "reasoning_effort", "max_completion_tokens", "temperature", "seed", "stream"} & set(first)
    assert first["tools"][0] == {"type": "function", "name": "search", "description": SEARCH.description,
                                 "parameters": dict(SEARCH.input_schema), "strict": False}
    assert first["input"] == [{"role": "system", "content": "sys"},
                              {"role": "user", "content": "Why do transfers fail?"}]
    # the next request carries the whole conversation: the reasoning and the call exactly as they came,
    # then the tool's output for the same call id
    assert second["input"][:2] == first["input"]
    assert second["input"][2:4] == [REASONING, function_call("search", {"query": "transfer failures"})]
    assert second["input"][4:] == [{"type": "function_call_output", "call_id": "call_1",
                                    "output": search_rows("jira:DEMO-1")}]


def test_a_chat_format_assistant_turn_becomes_input_items():
    """A history without Responses output items (written by hand, or from chat/completions) still maps."""
    items = agent_mod.responses_input([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "looking",
         "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "search", "arguments": '{"q": 1}'}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "rows"},
    ])
    assert items == [{"role": "user", "content": "q"}, {"role": "assistant", "content": "looking"},
                     {"type": "function_call", "call_id": "c0", "name": "search", "arguments": '{"q": 1}'},
                     {"type": "function_call_output", "call_id": "c0", "output": "rows"}]


# --- usage, cost and the cap -----------------------------------------------------------------


def test_usage_with_cached_and_reasoning_tokens_is_priced_in_the_ledger(tmp_path, server):
    server(search_then_answer)
    model = luna(tmp_path)
    result = one_try(model, FakeMcp({"transfer failures": search_rows("jira:DEMO-1")}))
    ledger = rows(tmp_path / "ledger.jsonl")
    assert [r["outcome"] for r in ledger] == ["answered", "answered"]
    first = ledger[0]
    assert (first["prompt_tokens"], first["cached_tokens"], first["completion_tokens"], first["reasoning_tokens"]) \
        == (1200, 1000, 300, 250)
    assert first["finish_reason"] == "tool_calls" and ledger[1]["finish_reason"] == "stop"
    # 200 uncached input at $0.10/M, 1000 cached at $0.01/M, 300 output (reasoning included) at $0.50/M
    assert first["cost_usd"] == pytest.approx((200 * 0.10 + 1000 * 0.01 + 300 * 0.50) / 1e6)
    assert model.ledger.spent == pytest.approx(sum(r["cost_usd"] for r in ledger)) == pytest.approx(result.cost)
    assert "sk-test" not in (tmp_path / "ledger.jsonl").read_text()


def test_a_reply_without_usage_is_charged_at_its_bound(tmp_path, server):
    server(lambda url, body, n: (200, response(message("hi"), usage=None)))
    model = luna(tmp_path)
    result = one_try(model, FakeMcp())
    row = rows(tmp_path / "ledger.jsonl")[0]
    assert result.status == ANSWERED and row["cost_usd"] == row["bound_usd"] > 0 and "no token usage" in row["note"]


def test_the_cap_stops_the_run_before_a_call_that_could_cross_it(tmp_path, server):
    # A call's bound is about 1000 output tokens at $0.50/M = $0.0005; the first call costs $0.00018, so
    # with a $0.0006 cap the second call could cross it and is never sent.
    fake = server(lambda url, body, n: (200, response(REASONING, function_call("search", {"query": "x"}, f"c{n}"))))
    model = luna(tmp_path, cap=0.0006)
    result = one_try(model, FakeMcp())
    assert result.status == STOPPED and "cap" in result.reason
    assert len(fake.requests) == 1 and len(rows(tmp_path / "ledger.jsonl")) == 1
    assert model.ledger.spent <= 0.0006


def test_an_answer_cut_by_max_output_tokens_is_the_named_failure(tmp_path, server):
    server(lambda url, body, n: (200, response(REASONING, status="incomplete", incomplete="max_output_tokens")))
    result = one_try(luna(tmp_path), FakeMcp())
    assert result.status == ERROR and result.reason == OUTPUT_LIMIT
    assert rows(tmp_path / "ledger.jsonl")[0]["finish_reason"] == "length"


# --- errors ----------------------------------------------------------------------------------


def test_an_error_body_is_surfaced_and_a_400_stops_the_run(tmp_path, server):
    error = {"error": {"message": "Unsupported value: 'reasoning.effort' does not support 'max' with this model.",
                       "type": "invalid_request_error", "param": "reasoning.effort", "code": "unsupported_value"}}
    server(lambda url, body, n: (400, error))
    model = luna(tmp_path)
    with pytest.raises(FatalModelError) as err:
        one_try(model, FakeMcp())
    text = str(err.value)
    assert err.value.status == 400 and "HTTP 400" in text and "/responses" in text
    assert "does not support 'max' with this model" in text
    assert "param reasoning.effort" in text and "code unsupported_value" in text
    row = rows(tmp_path / "ledger.jsonl")[0]
    assert row["outcome"] == "failed" and row["status"] == 400 and "does not support 'max'" in row["error"]


def test_a_response_that_failed_is_an_error_with_its_reason(tmp_path, server):
    failed = response(status="failed", usage=None)
    failed["error"] = {"code": "server_error", "message": "The model had a problem."}
    server(lambda url, body, n: (200, failed))
    result = one_try(luna(tmp_path), FakeMcp())
    assert result.status == ERROR and "The model had a problem." in result.reason and "server_error" in result.reason


class NotJson(Answer):
    def __init__(self, status, text):
        super().__init__(status, {})
        self.text = text

    def json(self):
        raise ValueError("not json")


def test_an_error_body_that_is_not_json_is_shown_as_text(server):
    server(lambda url, body, n: NotJson(502, "<html>bad gateway"))
    with pytest.raises(ModelError) as err:
        OpenAIResponses(OPENAI, "m", api_key="k").complete([], [])
    assert err.value.status == 502 and "<html>bad gateway" in str(err.value)


# --- chat/completions is unchanged -----------------------------------------------------------


def test_a_local_model_still_gets_the_same_chat_completions_request(server):
    fake = server(lambda url, body, n: (200, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                                              "usage": {"prompt_tokens": 3, "completion_tokens": 1}}))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    tools = openai_tools([SEARCH])
    model = make_model(LOCAL, "qwen", reasoning_effort="medium", max_completion_tokens=50)
    reply = model.complete(messages, tools)
    assert type(model) is OpenAIChat and reply.content == "hi" and reply.output_items == ()
    url, body, _ = fake.requests[0]
    assert url == LOCAL + "/chat/completions"
    assert body == {"model": "qwen", "messages": messages, "stream": False, "temperature": 0.0, "seed": 0,
                    "tools": tools, "reasoning_effort": "medium", "max_completion_tokens": 50}


# --- the command -----------------------------------------------------------------------------


@pytest.fixture
def command(monkeypatch, tmp_path, server):
    Endpoint.opened = 0
    monkeypatch.setattr(agent_mod, "McpToolBox", Endpoint)
    monkeypatch.setenv("OKG_CHAT_MCP_TOKEN", "t0ken")
    monkeypatch.delenv("ARCHI_CONTEXT_EVAL_QUESTIONS", raising=False)
    key = tmp_path / "model.key"
    key.write_text("sk-model-test\n")
    monkeypatch.setenv("ARCHI_EVAL_MODEL_KEY_FILE", str(key))
    questions = tmp_path / "questions.yaml"
    questions.write_text(yaml.safe_dump({"bundle": "demo", "k": 3, "questions": QUESTIONS}))

    def either_api(url, body, n):
        """Search for the question, then answer, in whichever API the request came on."""
        if url.endswith("/responses"):
            last = body["input"][-1]
            if last.get("role") == "user":
                return 200, response(REASONING, function_call("search", {"query": last["content"]}, f"c{n}"))
            return 200, response(message("Stuck transfers.\nSources: jira:DEMO-1"))
        if body["messages"][-1]["role"] == "user":
            call = {"id": f"c{n}", "type": "function",
                    "function": {"name": "search", "arguments": json.dumps({"query": body["messages"][-1]["content"]})}}
            return 200, {"choices": [{"message": {"content": "", "tool_calls": [call]}, "finish_reason": "tool_calls"}],
                         "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
        return 200, {"choices": [{"message": {"content": "Stuck transfers.\nSources: jira:DEMO-1"},
                                  "finish_reason": "stop"}], "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    return questions, write_corpus(tmp_path), server(either_api)


def test_the_command_runs_an_openai_model_on_the_responses_api(command, tmp_path, capsys):
    questions, corpus, fake = command
    code = main(agent_args(questions, corpus, tmp_path / "out", "--model-url", OPENAI, "--allow-remote-model",
                           "--reasoning-effort", "medium", "--model-price", "0.10/0.50/0.01"))
    assert code == 0, capsys.readouterr().err
    assert fake.requests and {url for url, _, _ in fake.requests} == {OPENAI + "/responses"}
    assert all(body["reasoning"] == {"effort": "medium"} for _, body, _ in fake.requests)
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["agent_arms"]["model_api"] == "responses"
    assert "responses API" in (tmp_path / "out" / "report.md").read_text()
    ledger = rows(tmp_path / "out" / "ledger.jsonl")
    assert len(ledger) == len(fake.requests) and all(r["outcome"] == "answered" and r["cost_usd"] > 0 for r in ledger)
    assert sum(r["reasoning_tokens"] for r in ledger) == 250 * len(ledger)
    assert "sk-model-test" not in (tmp_path / "out" / "ledger.jsonl").read_text()
    assert "sk-model-test" not in (tmp_path / "out" / "report.json").read_text()


@pytest.mark.parametrize("extra", [[], ["--model-url", OPENAI, "--allow-remote-model", "--model-api", "chat"]])
def test_the_command_keeps_chat_completions_for_a_local_model_or_when_asked(command, tmp_path, capsys, extra):
    questions, corpus, fake = command
    assert main(agent_args(questions, corpus, tmp_path / "out", *extra)) == 0, capsys.readouterr().err
    assert fake.requests and all(url.endswith("/chat/completions") for url, _, _ in fake.requests)
    assert json.loads((tmp_path / "out" / "report.json").read_text())["agent_arms"]["model_api"] == "chat"
