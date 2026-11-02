"""req.w2.compops-unbroken — the comp-ops instance stays green.

Deterministic diff of the two committed lint artifacts: the pre-W2
baseline and the latest re-check. Allowed delta: additional
skipped-severity notices introduced by okg version bumps. Forbidden:
any new error/warning finding, or any baseline finding disappearing
(which would mean someone changed the cms deployment).
"""
import json
from pathlib import Path

CHANGE = (
    Path(__file__).resolve().parents[2]
    / "pact" / "changes" / "w2-consolidate-hep-sources"
)


def _findings(name):
    data = json.loads((CHANGE / name).read_text(encoding="utf-8"))
    return {
        (f.get("severity", ""), f.get("code", ""), str(f.get("message", ""))[:60])
        for f in data.get("findings", [])
    }


def test_compops_lint_unchanged_modulo_version_skips():
    baseline = _findings("compops-lint-baseline.json")
    recheck = _findings("compops-lint-recheck-20260821.json")
    gone = baseline - recheck
    new = recheck - baseline
    hard_new = {f for f in new if f[0] in ("error", "blocker", "warning")}
    assert not gone, f"baseline findings disappeared (cms was touched?): {sorted(gone)}"
    assert not hard_new, f"new error/warning findings vs baseline: {sorted(hard_new)}"
