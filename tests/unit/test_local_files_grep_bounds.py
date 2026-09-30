from src.archi.pipelines.agents.tools import local_files as lf


def _hit(text, before=(), after=()):
    return {"hash": "h", "path": "p", "metadata": {},
            "matches": [{"line": 1, "text": text, "before": list(before), "after": list(after)}]}


def test_small_hits_are_unchanged():
    out = lf._format_grep_hits([_hit("short line", ["b"], ["a"])])
    assert "L1: short line" in out and "B: b" in out and "A: a" in out
    assert "truncated" not in out


def test_huge_line_is_clipped():
    out = lf._format_grep_hits([_hit("x" * 2_000_000)])
    assert len(out) < lf.MAX_GREP_LINE_CHARS + 200
    assert "[line truncated, 2000000 chars]" in out


def test_total_output_is_bounded():
    hits = [_hit("y" * 900, ["b" * 900] * 3, ["a" * 900] * 3) for _ in range(40)]
    out = lf._format_grep_hits(hits)
    assert len(out) <= lf.MAX_GREP_OUTPUT_CHARS + 200
    assert "output truncated" in out
