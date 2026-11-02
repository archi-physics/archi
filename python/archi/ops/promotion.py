"""Promotion of an automation from shadow to live (PACT r3-ops-runtime req.shadow; ADR 0002 section 7.3).

"Promotion to live needs recorded shadow results and an operator's approval, recorded as a PACT
decision on the instance." The approval is a human step that archi never takes. `archi ops promote`
does the part before it:

1. refuse unless the shadow report passes `archi.ops.shadow.promotion_readiness` (the caller passes
   the report, the readiness verdict and the thresholds it was judged with);
2. refuse unless the instance directory is a git repository;
3. `okg pact new promote-<automation>-<time> --shape fast` in the instance repo. The change directory
   holds the shadow report (`shadow-report.json`) AND the thresholds it was judged against
   (`thresholds.json`), and a junit file (`readiness.xml`) written from the readiness verdict itself:
   one test, `test_promotion_readiness`, that fails with the reasons when the report is not ready. It
   is ingested as the change's evidence, and `okg pact gate task-done` brings the change to
   `awaiting_review`, the one state in which an approval is a legal verdict;
4. set `mode: live` and `promotion: <change>` in `ops.yaml` and commit those and the change directory,
   by name. Then it stops.

With no trial (`archi ops promote <automation> --no-trial --note "<decision>"`), step 1 is replaced: the
request names the decision that waived the shadow trial (for example a board decision to switch a queue
without one), and the change records it in its intent, in `promotion.json` (`no_trial`) and in its junit
evidence (`test_promotion_without_trial`, passing, with the decision and the shadow report at request time).
Everything after it is the same: an operator still approves the change, and until then the automation
runs in shadow.

An operator reads the change and approves it:

    okg pact decide <change> --verdict approve --principal <name> --dsn $OKG_DSN

(okg moves the approved change to `pact/archive/<change>`; commit that too.) The runtime counts
`mode: live` only while the named change is approved in okg's decision ledger AND is bound to this
automation (its `promotion.json` names it), is the newest promotion change for it, and was approved after the
latest demotion (`OpsIO.promotion_status`,
`archi.ops.loop` `Deps.promotions`); until then an automation runs in shadow and records why, and
hand-editing `ops.yaml` to say live changes nothing. The ledger's approval time is the `promoted_at`
that lifts an earlier demotion: a demotion recorded after it keeps the automation in shadow.

Nothing is written before every refusal check has passed. If `okg pact` fails, `ops.yaml` is untouched
and nothing is committed (the half-made change directory is left for the operator to look at).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from xml.sax.saxutils import escape, quoteattr

import yaml

from archi.ops.record import _okg
from archi.ops.runtime_config import OPS_FILE


class PromotionRefused(RuntimeError):
    """The automation cannot be promoted now; the message says why."""


Runner = Callable[..., subprocess.CompletedProcess]


def change_id(automation: str, now: datetime) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", automation.lower()).strip("-")
    return f"promote-{slug}-{now.astimezone(timezone.utc):%Y%m%dt%H%M%Sz}"


def summary(report: Mapping[str, Any]) -> str:
    """One line of the report for the PACT note."""
    keys = ("scored", "answered", "agreement_rate", "response_coverage", "false_proposal_rate_upper")
    return ", ".join(f"{k}={report.get(k)}" for k in keys)


def readiness_junit(readiness: Mapping[str, Any], report: Mapping[str, Any], thresholds: Mapping[str, Any],
                    no_trial: str = "") -> str:
    """A junit file that says what the readiness check said: one test, failing with the reasons if not ready.
    With `no_trial` it is `test_promotion_without_trial`, passing, and names the decision that waived the trial."""
    if no_trial:
        out = escape(f"no shadow evidence: no shadow trial, on the decision: {no_trial}\nreport at request time: {summary(report)}")
        return ('<?xml version="1.0" encoding="utf-8"?>\n<testsuites><testsuite name="archi-ops-promotion" tests="1" '
                'failures="0" errors="0"><testcase classname="promotion.no_trial" '
                f'name="test_promotion_without_trial"><system-out>{out}</system-out></testcase></testsuite></testsuites>\n')
    reasons = list(readiness.get("reasons") or [])
    failed = not readiness.get("ready")
    body = f"<failure message={quoteattr('; '.join(reasons) or 'not ready')}/>" if failed else ""
    out = escape(f"report: {summary(report)}\nthresholds: {json.dumps(dict(thresholds), sort_keys=True)}")
    return ('<?xml version="1.0" encoding="utf-8"?>\n<testsuites><testsuite name="archi-ops-promotion" tests="1" '
            f'failures="{int(failed)}" errors="0"><testcase classname="promotion.shadow_report" '
            f'name="test_promotion_readiness">{body}<system-out>{out}</system-out></testcase></testsuite></testsuites>\n')


PROMOTION_FILE = "promotion.json"


def promotion_record(instance_dir: Path, change: str) -> Optional[dict]:
    """The `promotion.json` of a change under pact/changes or pact/archive, or None if it is not an archi promotion."""
    for where in ("changes", "archive"):
        path = Path(instance_dir) / "pact" / where / change / PROMOTION_FILE
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


def latest_promotion_change(instance_dir: Path, automation: str) -> Optional[str]:
    """The newest promotion change opened for `automation` (by its recorded request time), approved or not."""
    found = []
    for where in ("changes", "archive"):
        base = Path(instance_dir) / "pact" / where
        for path in base.glob(f"*/{PROMOTION_FILE}") if base.is_dir() else ():
            record = promotion_record(instance_dir, path.parent.name)
            if record and record.get("automation") == automation:
                found.append((str(record.get("requested_at") or ""), path.parent.name))
    return max(found)[1] if found else None


def _git(directory: Path, *args: str, run: Runner) -> subprocess.CompletedProcess:
    """git in the instance repo, committing as archi does (`archi init` sets no user in the repo)."""
    from archi.cli.init import _git_identity_env

    return run(["git", "-C", str(directory), *args], capture_output=True, text=True,
               env={**os.environ, **_git_identity_env()})


def promote(
    instance_dir: Path,
    automation: str,
    *,
    report: Mapping[str, Any],
    readiness: Mapping[str, Any],
    thresholds: Optional[Mapping[str, Any]] = None,
    principal: str,
    note: str = "",
    dsn: Optional[str] = None,
    deployment: Optional[str] = None,
    now: Optional[datetime] = None,
    run: Runner = subprocess.run,
    okg: Optional[str] = None,
    no_trial: str = "",
) -> dict[str, Any]:
    """Open the promotion PACT change for `automation`; see the module docstring. Raises PromotionRefused.
    `no_trial` names the decision that waived the shadow trial; with it the readiness verdict is not required."""
    instance_dir = Path(instance_dir)
    thresholds = dict(thresholds or {})
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    no_trial = (no_trial or "").strip()
    if not no_trial and not readiness.get("ready"):
        raise PromotionRefused(
            f"{automation} is not ready to promote: " + "; ".join(readiness.get("reasons") or ["no shadow report"]))
    inside = _git(instance_dir, "rev-parse", "--is-inside-work-tree", run=run)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise PromotionRefused(f"{instance_dir} is not a git repository; the promotion is recorded in "
                               "the instance repo, so it must be one")
    ops_path = instance_dir / OPS_FILE
    data = yaml.safe_load(ops_path.read_text(encoding="utf-8")) if ops_path.is_file() else {}
    data = data or {}
    if automation not in (data.get("automations") or {}):
        raise PromotionRefused(f"{ops_path} has no automations.{automation}; there is nothing to promote")

    change = change_id(automation, now)
    exe = okg or _okg()
    basis = (f"No shadow evidence: there was no shadow trial, on the decision: {no_trial}. Shadow report at request time: "
             f"{summary(report)}."
             if no_trial else
             f"Shadow report: {summary(report)}. Thresholds: {json.dumps(thresholds, sort_keys=True)}.")
    text = (f"Promote automation {automation} from shadow to live. {basis}"
            + (f" Requested by {principal}." if principal else "")
            + (f" Note: {note}" if note else ""))
    pact_env = {**os.environ, "OKG_PACT_QUIET_NUDGE": "1"}
    if deployment:
        pact_env["OKG_PACT_GRAPH_DEPLOYMENT"] = deployment
        pact_env.setdefault("OKG_DEPLOYMENTS_DIR", str(instance_dir / "deployments"))
    if dsn:
        pact_env["OKG_DSN"] = dsn
    change_dir = instance_dir / "pact" / "changes" / change
    rel = f"pact/changes/{change}"
    test_id = ("promotion/no_trial.py::test_promotion_without_trial" if no_trial else
               "promotion/shadow_report.py::test_promotion_readiness")
    acceptance = (f"No shadow evidence: an operator approves going live with no shadow trial, on the decision: {no_trial}." if no_trial
                  else "The shadow report passes promotion_readiness and an operator approves.")

    def pact(*args: str, what: str) -> None:
        done = run([exe, "pact", "--root", str(instance_dir), *args], capture_output=True, text=True,
                   env=pact_env, cwd=str(instance_dir))
        if done.returncode != 0:
            detail = "\n".join(line for line in ((done.stderr or "") + "\n" + (done.stdout or "")).splitlines()
                               if line.strip() and not line.startswith("tip:"))[-1500:]
            raise PromotionRefused(f"okg pact {what} failed; {automation} is unchanged and nothing was "
                                   f"committed: {detail}")

    pact("new", change, "--shape", "fast", "--title", f"Promote {automation} to live", "--intent", text,
         "--acceptance", acceptance,
         "--task-title", f"Promote {automation}", "--test-id", test_id,
         "--test-command", f"archi ops promote {automation}", what="new")
    (change_dir / "shadow-report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str),
                                                   encoding="utf-8")
    (change_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True), encoding="utf-8")
    # What binds the change to this automation: the runtime reads it back, so an id in ops.yaml is not enough.
    (change_dir / PROMOTION_FILE).write_text(json.dumps(
        {"automation": automation, "change": change, "requested_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "requested_by": principal, **({"no_trial": no_trial} if no_trial else {})},
        indent=2, sort_keys=True), encoding="utf-8")
    (change_dir / "readiness.xml").write_text(readiness_junit(readiness, report, thresholds, no_trial),
                                                   encoding="utf-8")
    pact("evidence", "ingest", change, "--pytest-report", f"{rel}/readiness.xml", what="evidence ingest")
    pact("gate", "task-done", change, f"task.{change}", what="gate task-done")

    data["automations"][automation] = {**(data["automations"][automation] or {}), "mode": "live", "promotion": change}
    ops_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    added = _git(instance_dir, "add", "--", OPS_FILE, rel, run=run)
    committed = _git(instance_dir, "commit", "-q", "-m", f"request promotion of {automation} ({change})", "--",
                     OPS_FILE, rel, run=run)
    if added.returncode != 0 or committed.returncode != 0:
        raise PromotionRefused("the change is open and ops.yaml names it, but the commit failed: "
                               f"{(committed.stderr or added.stderr or committed.stdout)[-1500:]}")
    return {"automation": automation, "change": change, "state": "awaiting_review", "report": summary(report),
            "thresholds": thresholds,
            "no_trial": no_trial or None,
            "next": f"an operator approves it: okg pact decide {change} --verdict approve --principal <name>; "
                    "until then the automation runs in shadow"}
