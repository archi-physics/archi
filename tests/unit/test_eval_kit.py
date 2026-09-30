"""Tests for the eval kit in scripts/eval (export, report, review page, answer import), on synthetic data."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[2] / "scripts" / "eval"


def load(name):
    spec = importlib.util.spec_from_file_location(f"eval_kit_{name}", KIT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return module.main()


ITEM = """
id: {id}
status: {status}
question: {question}
gold: The answer to {id}.
atoms:
- id: a1
  text: first fact
  required: true
  answer_source: doc
- id: a2
  text: second fact
  required: true
  answer_source: {src2}
- id: a3
  text: optional fact
  required: false
  answer_source: doc
past_answers:
- text: an old answer
  changed: replaced in 2025
labels:
  task_type: {task}
"""


def write_set(root: Path):
    root.mkdir()
    (root / "q1.yaml").write_text(ITEM.format(id="q1", status="approved", question="What is one?", src2="doc", task="procedure"))
    (root / "q2.yaml").write_text(ITEM.format(id="q2", status="approved", question="What is two?", src2="live_service", task="status"))
    (root / "q3.yaml").write_text(ITEM.format(id="q3", status="draft", question="Not ready?", src2="doc", task="status"))
    (root / "_notes.yaml").write_text("question: ignored\n")


def write_run(run: Path, verdicts: dict, failed=()):
    """verdicts: {item_id: {atom_id: outcome}}; items in `failed` get an execution failure."""
    run.mkdir(parents=True)
    answers, evals = [], []
    for item, outcomes in verdicts.items():
        aid = f"{item}-attempt-1"
        if item in failed:
            answers.append({"attempt_id": aid, "item_id": item, "ordinal": 1, "status": "execution_failed",
                            "error": {"type": "AnswerTimeLimitExceeded", "message": "time limit"}, "duration_ms": 900000})
            continue
        answers.append({"attempt_id": aid, "item_id": item, "ordinal": 1, "status": "answer_ready",
                        "answer": f"answer {item} <!-- html --> </script>",
                        "duration_ms": 12000, "usage": {"input_tokens": 1000, "output_tokens": 100},
                        "tool_calls": [{"ordinal": 1, "name": "search", "query": "q", "response": "r" * 50, "duration_ms": 5}]})
        evals.append({"attempt_id": aid, "item_id": item, "status": "scored",
                      "judgments": [{"atom_id": k, "outcome": v, "rationale": "because"} for k, v in outcomes.items()]})
    (run / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
    (run / "evaluation_results.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evals))


@pytest.fixture()
def dataset(tmp_path, monkeypatch):
    write_set(tmp_path / "golden")
    out = tmp_path / "ds"
    assert run_main(load("export_dataset"), ["--set", str(tmp_path / "golden"), "--out", str(out)], monkeypatch) == 0
    return out


def test_export_dataset(dataset):
    bench = json.loads((dataset / "benchmark.json").read_text())
    labels = json.loads((dataset / "labels.json").read_text())
    assert [r["id"] for r in bench] == ["q1", "q2"]  # the draft item is skipped
    assert bench[0]["expected_atoms"][0] == {"id": "a1", "text": "first fact", "required": True}
    assert "Outdated answers (no longer true):\n- an old answer (replaced in 2025)" in bench[0]["answer"]
    assert bench[0]["time_sensitive"] is False and bench[0]["category"] == "procedure"
    assert labels["q1"]["multi_source"] == "single"
    assert labels["q2"]["multi_source"] == "static_live"
    assert labels["q1"]["changed_over_time"] == "yes"
    assert labels["q2"]["atoms"][1]["answer_source"] == "live_service"


def test_export_stops_on_an_item_without_required_atoms(tmp_path, monkeypatch):
    root = tmp_path / "golden"
    root.mkdir()
    (root / "bad.yaml").write_text("id: b\nstatus: approved\nquestion: q?\ngold: g\natoms:\n- id: x\n  text: t\n")
    with pytest.raises(SystemExit, match="no required atom"):
        run_main(load("export_dataset"), ["--set", str(root), "--out", str(tmp_path / "o")], monkeypatch)


def test_report_scores_and_pairs(dataset, tmp_path, monkeypatch, capsys):
    write_run(tmp_path / "runs" / "a", {"q1": {"a1": "entailed", "a2": "not_mentioned"},
                                        "q2": {"a1": "entailed", "a2": "entailed"}})
    write_run(tmp_path / "runs" / "b", {"q1": {"a1": "entailed", "a2": "entailed"},
                                        "q2": {"a1": "contradicted", "a2": "entailed"}}, failed=())
    write_run(tmp_path / "runs" / "c", {"q1": {}, "q2": {"a1": "entailed", "a2": "entailed"}}, failed=("q1",))
    run_main(load("report"), ["--labels", str(dataset / "labels.json"), "--pair", "a:b",
                              f"a={tmp_path / 'runs' / 'a'}", f"b={tmp_path / 'runs' / 'b'}",
                              f"c={tmp_path / 'runs' / 'c'}"], monkeypatch)
    out = capsys.readouterr().out
    rows = {}
    for line in out.splitlines():  # the first line per setup is its row in the summary table
        name = line.split("|")[1].strip() if line.startswith("| ") else None
        if name in ("a", "b", "c"):
            rows.setdefault(name, line)
    assert "| 75% |" in rows["a"]            # q1 50%, q2 100%
    assert "| 75% |" in rows["b"] and "| 50% |" in rows["b"]  # q2 has a contradicted atom: 1 of 2 answers
    assert "| 50% |" in rows["c"] and "answer_timeout 1" in rows["c"]  # the failed attempt counts 0
    assert "1,000 / 100" in rows["a"]
    assert "b − a: **+0 points**" in out and "better on 1, worse on 1, equal on 0" in out
    assert "## Required atoms found, by answer source" in out
    assert "## Correct, by task_type" in out


def test_review_page(dataset, tmp_path, monkeypatch):
    write_run(tmp_path / "runs" / "a", {"q1": {"a1": "entailed", "a2": "entailed"}, "q2": {"a1": "entailed", "a2": "entailed"}})
    out = tmp_path / "review.html"
    run_main(load("review_page"), ["--dataset", str(dataset / "benchmark.json"), "--labels", str(dataset / "labels.json"),
                                   "--out", str(out), "--tool-output-chars", "10", f"a={tmp_path / 'runs' / 'a'}"], monkeypatch)
    page = out.read_text()
    blob = page.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
    data = json.loads(blob)
    assert data["setups"] == ["a"]
    assert data["items"]["q1"]["runs"]["a"][0]["score"] == 1.0
    assert data["items"]["q1"]["runs"]["a"][0]["tools"][0][4] == "r" * 10  # tool output cut to 10 characters
    assert data["items"]["q1"]["runs"]["a"][0]["answer"].endswith("<!-- html --> </script>")  # survives embedding


def test_import_answers(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"phases": {"prepare": {"status": "completed"}}, "artifacts": {}}))
    (run / "preparation.jsonl").write_text("".join(json.dumps({"id": q, "status": "prepared"}) + "\n" for q in ("q1", "q2")))
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"item_id": "q2", "ordinal": 1, "status": "answer_ready", "answer": "two"}) + "\n"
                       + json.dumps({"item_id": "q1", "ordinal": 1, "status": "execution_failed",
                                     "error": {"type": "X", "message": "m"}}) + "\n")
    (tmp_path / "cfg.yaml").write_text("agent: other\n")
    (tmp_path / "prompt.md").write_text("You answer.\n")
    run_main(load("import_answers"), [str(run), str(answers), str(tmp_path / "cfg.yaml"), str(tmp_path / "prompt.md"),
                                      "--agent-name", "other", "--provider", "openai", "--model", "m"], monkeypatch)
    rows = [json.loads(line) for line in (run / "answers.jsonl").read_text().splitlines()]
    assert [r["attempt_id"] for r in rows] == ["q1-attempt-1", "q2-attempt-1"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "run_completed" and manifest["phases"]["run"]["status"] == "completed"


def test_import_answers_refuses_missing_slots(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"phases": {"prepare": {"status": "completed"}}, "artifacts": {}}))
    (run / "preparation.jsonl").write_text("".join(json.dumps({"id": q, "status": "prepared"}) + "\n" for q in ("q1", "q2")))
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"item_id": "q1", "ordinal": 1, "status": "answer_ready", "answer": "one"}) + "\n")
    (tmp_path / "c").write_text("x")
    with pytest.raises(SystemExit):
        run_main(load("import_answers"), [str(run), str(answers), str(tmp_path / "c"), str(tmp_path / "c"),
                                          "--agent-name", "o", "--provider", "p", "--model", "m"], monkeypatch)
