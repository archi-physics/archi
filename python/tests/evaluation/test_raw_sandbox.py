"""acc.r3-context-quality.raw-arm-safety: the agent-bash arm's sandbox, with real bubblewrap.

Each test runs real commands under `bwrap` and is skipped, with the reason, on a host that
has none (macOS, most CI images: CI skips them). Set ARCHI_REQUIRE_BWRAP=1 to make them fail
instead of skip, on a host that must prove the sandbox. The refusals without a working bwrap
run everywhere.
"""
import os
import re
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

from archi.evaluation import raw_arm
from archi.evaluation.raw_arm import BashSandbox, SandboxError

REQUIRE_BWRAP = os.environ.get("ARCHI_REQUIRE_BWRAP") == "1"
needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None and not REQUIRE_BWRAP,
                                 reason="bubblewrap (bwrap) is not installed on this host "
                                        "(set ARCHI_REQUIRE_BWRAP=1 to fail instead)")
SECRET = "sk-sandbox-test-0000"


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "records"
    (root / "jira").mkdir(parents=True)
    (root / "jira" / "DEMO-1.json").write_text('{"key": "DEMO-1", "description": "Transfer failures at the test site"}\n')
    return root


def sandbox(corpus, **kwargs):
    try:
        return BashSandbox(corpus, **kwargs)
    except SandboxError as exc:  # pragma: no cover - a host where bwrap exists but cannot start
        if REQUIRE_BWRAP:
            raise
        pytest.skip(str(exc))


def test_without_bwrap_the_bash_arm_refuses_to_run(corpus, monkeypatch):
    monkeypatch.setattr(raw_arm.shutil, "which", lambda name: None)
    with pytest.raises(SandboxError, match="bubblewrap \\(bwrap\\) is not installed"):
        BashSandbox(corpus)


def test_a_sandbox_that_cannot_run_true_refuses_to_start(corpus):
    false = shutil.which("false")
    if false is None or not any(Path(p).is_file() for p in ("/usr/bin/bash", "/bin/bash")):
        pytest.skip("this host has no `false` or no bash")
    with pytest.raises(SandboxError, match="the sandbox cannot start: `true` in it gave exit 1"):
        BashSandbox(corpus, bwrap=false)


@needs_bwrap
def test_the_limits_are_set_inside(corpus):
    box = sandbox(corpus, max_procs=50, memory_bytes=2 * 1024 ** 3, file_bytes=8 * 1024 ** 2, command_seconds=20)
    result = box.run("ulimit -v; ulimit -f; ulimit -t; ulimit -u")
    assert result.ok, result.text
    memory, files, cpu, procs = result.text.split()
    assert int(memory) <= 2 * 1024 ** 2 and int(files) <= 8 * 1024 and int(cpu) <= 20
    if box.nproc_mode == "namespace":  # the kernel counts processes in bwrap's user namespace
        assert int(procs) == 50
    else:
        assert int(procs) <= raw_arm._user_tasks() + 50 + 64


@needs_bwrap
def test_a_large_allocation_is_stopped(corpus):
    box = sandbox(corpus, memory_bytes=1024 ** 3)
    if "python3" not in box.tools:
        pytest.skip("no python3 on this host to allocate with")
    result = box.run("python3 -c 'b = bytearray(3 * 1024 ** 3); print(\"ALLOCATED\")'")
    assert "ALLOCATED" not in result.text and "MemoryError" in result.text and not result.ok


@needs_bwrap
def test_a_fork_bomb_is_stopped(corpus):
    box = sandbox(corpus, max_procs=64, command_seconds=30)
    if "python3" not in box.tools:
        pytest.skip("no python3 on this host to fork with")
    # Forks until the limit says no (bounded at 2000 even if it never does), then reports how
    # many it got; the children sleep, and die with the sandbox's PID namespace.
    script = ("import os, time\n"
              "n = 0\n"
              "try:\n"
              "    while n < 2000:\n"
              "        if os.fork() == 0:\n"
              "            time.sleep(60); os._exit(0)\n"
              "        n += 1\n"
              "except OSError as e:\n"
              "    print(\"refused\", e.errno, \"forked\", n, flush=True)\n"
              "else:\n"
              "    print(\"forked\", n, flush=True)\n")
    started = time.monotonic()
    result = box.run(f"python3 -c '{script}'")
    assert time.monotonic() - started < 25, result.text
    assert "refused" in result.text, result.text
    match = re.search(r"forked (\d+)", result.text)
    assert match, result.text
    forked = int(match.group(1))
    if box.nproc_mode == "namespace":
        assert forked < 64, result.text
    else:  # the user's other jobs on the host come and go
        assert forked < 64 + 200, result.text


@needs_bwrap
def test_a_command_scope_bounds_memory_across_processes_and_tmpfs(corpus):
    """Three writers of 200 MiB to the tmpfs /tmp, each under the per-process limits, hold 600 MiB
    together: a 384 MiB command scope stops them; per-process limits alone would not."""
    box = sandbox(corpus, memory_bytes=384 * 1024 ** 2, file_bytes=256 * 1024 ** 2, command_seconds=60)
    if box.limit_scope != raw_arm.SCOPE_COMMAND:
        assert box.scope_note  # the run says why, and the CLI prints it
        pytest.skip(f"no per-command scope on this host: {box.scope_note}")
    assert box.argv("true")[:3] == [shutil.which("systemd-run"), "--user", "--scope"]
    result = box.run("for f in a b c; do head -c 200M /dev/zero > /tmp/$f || exit 9; done; echo WROTE-ALL")
    assert "WROTE-ALL" not in result.text and not result.ok, result.text


@needs_bwrap
def test_a_file_past_the_size_limit_is_stopped(corpus):
    box = sandbox(corpus, file_bytes=1024 ** 2)
    result = box.run("head -c 5000000 /dev/zero > /tmp/big; ls -l /tmp/big | awk '{print $5}'")
    assert "File size limit exceeded" in result.text or "File too large" in result.text, result.text
    size = int(result.text.strip().splitlines()[-1])
    assert size <= 1024 ** 2


@needs_bwrap
def test_the_corpus_is_read_at_slash_corpus(corpus):
    result = sandbox(corpus).run("pwd; cat /corpus/jira/DEMO-1.json")
    assert result.ok, result.text
    assert result.text.splitlines()[0] == "/corpus" and '"DEMO-1"' in result.text


@needs_bwrap
def test_there_is_no_network(corpus):
    # A server on the host's loopback: reachable from outside the sandbox, not from inside.
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    threading.Thread(target=lambda: server.accept(), daemon=True).start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass  # the host itself can connect
        result = sandbox(corpus).run(
            f"exec 3<>/dev/tcp/127.0.0.1/{port} && echo LOOPBACK-OPEN; "
            "exec 4<>/dev/tcp/1.1.1.1/53 && echo INTERNET-OPEN; "
            "cat /proc/net/dev")
    finally:
        server.close()
    assert "LOOPBACK-OPEN" not in result.text and "INTERNET-OPEN" not in result.text
    interfaces = re.findall(r"^\s*([\w.-]+):\s+\d", result.text, re.M)  # /proc/net/dev's rows
    assert interfaces == ["lo"], result.text


@needs_bwrap
def test_the_corpus_cannot_be_written(corpus):
    before = (corpus / "jira" / "DEMO-1.json").read_text()
    box = sandbox(corpus)
    for command in ("echo x > /corpus/new.txt", "echo x >> /corpus/jira/DEMO-1.json", "rm /corpus/jira/DEMO-1.json",
                    "mkdir /corpus/d", "touch /corpus/jira/DEMO-1.json"):
        result = box.run(command)
        assert not result.ok, command
        assert "Read-only file system" in result.text or "Permission denied" in result.text, result.text
    assert (corpus / "jira" / "DEMO-1.json").read_text() == before
    assert sorted(p.name for p in corpus.rglob("*")) == ["DEMO-1.json", "jira"]


@needs_bwrap
def test_home_is_a_tmpfs_that_is_gone_after_the_command(corpus):
    box = sandbox(corpus)
    result = box.run('echo "$HOME"; stat -f -c %T "$HOME"; echo note > "$HOME/n.txt" && cat "$HOME/n.txt"')
    assert result.ok, result.text
    home, fstype, note = result.text.splitlines()[:3]
    assert home == raw_arm.SANDBOX_HOME and fstype == "tmpfs" and note == "note"
    again = box.run('ls -A "$HOME"')
    assert again.ok and again.text.strip() == ""  # each command gets a fresh HOME


@needs_bwrap
def test_no_key_or_its_file_is_visible_inside(corpus, tmp_path, monkeypatch):
    key_file = tmp_path / "model.key"
    key_file.write_text(SECRET + "\n")
    monkeypatch.setenv("ARCHI_EVAL_MODEL_KEY_FILE", str(key_file))
    monkeypatch.setenv("ARCHI_EVAL_JUDGE_KEY_FILE", str(key_file))
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OKG_CHAT_MCP_TOKEN", SECRET)
    box = sandbox(corpus)
    result = box.run(f"env; cat {key_file}; ls {tmp_path}")
    env_lines = [line for line in result.text.splitlines() if "=" in line]
    names = {line.split("=", 1)[0] for line in env_lines}
    assert SECRET not in result.text and not any(str(key_file) in line for line in env_lines)
    assert names <= {"PATH", "HOME", "TMPDIR", "LANG", "PWD", "SHLVL", "_", "OLDPWD"}, names
    assert "No such file or directory" in result.text
    assert not any(SECRET in arg for arg in box.argv("true"))


@needs_bwrap
def test_a_command_is_stopped_at_its_time_limit(corpus):
    box = sandbox(corpus, command_seconds=1)
    started = time.monotonic()
    result = box.run("sleep 30; echo late")
    assert time.monotonic() - started < 10
    assert result.timed_out and not result.ok and "late" not in result.text
    assert "stopped after 1 seconds" in result.text


@needs_bwrap
def test_output_past_the_limit_is_cut_and_the_command_stopped(corpus):
    box = sandbox(corpus, output_bytes=1024)
    started = time.monotonic()
    result = box.run("yes corpus-line")  # never ends by itself
    assert time.monotonic() - started < 10
    assert result.cut and not result.ok
    assert result.responded  # the visible part is still a response the scorer reads
    body, note = result.text.rsplit("\n", 1)
    assert len(body.encode()) <= 1024 and note == "[output cut at 1024 bytes; the command was stopped]"


@needs_bwrap
def test_a_command_that_printed_nothing_is_not_a_response(corpus):
    result = sandbox(corpus).run("grep -r NOTHING-MATCHES /corpus")
    assert result.exit_code == 1 and not result.ok and not result.responded and "[exit code 1]" in result.text


@needs_bwrap
def test_a_nonzero_exit_that_printed_something_is_a_response(corpus):
    import asyncio

    tools = raw_arm.BashTools(sandbox(corpus))
    text, is_error = asyncio.run(tools.call("bash", {"command": "cat /corpus/jira/DEMO-1.json /corpus/nope"}))
    assert not is_error and "Transfer failures" in text and "[exit code 1]" in text


@needs_bwrap
def test_the_bash_tool_reports_failure_to_the_loop(corpus):
    import asyncio

    tools = raw_arm.BashTools(sandbox(corpus))
    text, is_error = asyncio.run(tools.call("bash", {"command": "ls /corpus"}))
    assert not is_error and text.strip() == "jira"
    text, is_error = asyncio.run(tools.call("bash", {"command": "true"}))
    assert is_error  # printed nothing
    text, is_error = asyncio.run(tools.call("bash", {}))
    assert is_error and "command" in text


@needs_bwrap
def test_the_host_environment_is_not_inherited_even_without_clearenv(corpus, monkeypatch):
    """Belt and braces: the subprocess itself starts with an empty environment, but for the two
    variables systemd-run needs to find the user's manager when a command scope is in use."""
    seen = {}
    real = raw_arm.subprocess.Popen

    def spy(argv, **kwargs):
        seen.update(kwargs)
        return real(argv, **kwargs)

    monkeypatch.setattr(raw_arm.subprocess, "Popen", spy)
    sandbox(corpus).run("true")
    assert set(seen["env"]) <= set(raw_arm._SCOPE_ENV) and seen["start_new_session"] is True
    assert os.environ  # the test process still has its own


@needs_bwrap
def test_both_agent_arms_run_on_one_scorer_with_the_bash_arm_in_the_sandbox(tmp_path):
    """agent-bash greps the corpus, agent-graph searches the fake MCP; one scripted model."""
    import asyncio

    from archi.evaluation.agent import ChatReply, ToolCall
    from archi.evaluation.context import parse_context
    from archi.evaluation.raw_arm import (AGENT_BASH, AGENT_GRAPH, AgentPlan, Caps, Ledger, Metered, Price,
                                         resolve_gold, run_agent_arms)
    from archi.evaluation.raw_corpus import verify_corpus
    from evaluation.raw_support import FakeMcp, ScriptedModel, search_rows, write_corpus

    corpus = verify_corpus(write_corpus(tmp_path))
    manifest = corpus.manifest_paths()
    cset = parse_context({"bundle": "demo", "questions": [
        {"id": "q1", "question": "Why do transfers fail?", "gold": ["jira:DEMO-1"]}]})
    tools = FakeMcp({"Why do transfers fail?": search_rows("jira:DEMO-1")})

    class Model(ScriptedModel):
        def complete(self, messages, tool_schema):
            self.calls += 1
            if messages[-1]["role"] != "user":
                return ChatReply("Stuck transfers.", (), 10, 5)
            if any(t["function"]["name"] == "bash" for t in tool_schema):
                return ChatReply("", (ToolCall("b1", "bash", {"command": "grep -r 'Transfer failures' /corpus"}),),
                                 10, 5)
            return ChatReply("", (ToolCall("s1", "search", {"query": "Why do transfers fail?"}),), 10, 5)

    ledger = Ledger(tmp_path / "ledger.jsonl", 1.0)
    model = Metered(Model(None), Price(1.0, 10.0), ledger, 100)
    plan = AgentPlan((AGENT_BASH, AGENT_GRAPH), model, Caps(), 1, corpus, sandbox=BashSandbox(corpus.records))
    gold = asyncio.run(resolve_gold(tools, cset.scored, corpus.keys(), manifest, corpus.records))
    data = asyncio.run(run_agent_arms(cset.scored, plan, tools, gold, manifest=manifest)).to_dict()
    tries = {arm: ts[0] for arm, ts in data["questions"][0]["arms"].items()}
    assert tries[AGENT_BASH]["calls"][0]["ok"] and tries[AGENT_BASH]["evidence"]["found"] == ["jira:DEMO-1"]
    assert tries[AGENT_GRAPH]["evidence"]["found"] == ["jira:DEMO-1"]
    assert data["arms"][AGENT_BASH]["evidence_recall"] == data["arms"][AGENT_GRAPH]["evidence_recall"] == 1.0
    assert data["sandbox"]["bwrap"] and set(data["sandbox"]["tools"]) | set(data["sandbox"]["missing_tools"]) == {
        "rg", "jq", "python3"}
