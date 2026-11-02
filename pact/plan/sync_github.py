"""Create the Archi 3.0 milestones, labels and issues on GitHub from the plan.

Reads pact/plan/archi-3.0.yaml for milestones, labels and issue titles, and each
pact/changes/<id>/pact.yaml for the requirement and verification text. Creates one
issue per PACT task and one epic issue per PACT that lists its tasks.

Idempotent: milestones, labels and issues are matched by exact title, so a re-run
creates only what is missing and refreshes epic bodies. Needs an authenticated `gh`.

    python pact/plan/sync_github.py                # dry run: print what would change
    python pact/plan/sync_github.py --apply
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "pact" / "plan" / "archi-3.0.yaml"
BLOB = "https://github.com/{repo}/blob/{ref}/pact/changes/{pid}/pact.yaml"


def gh(*args, payload=None):
    cmd = ["gh", "api", *args]
    if payload is not None:
        cmd += ["--input", "-"]
    out = subprocess.run(
        cmd,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout) if out.stdout.strip() else None


def paged(repo, path):
    return gh("--paginate", "--slurp", f"repos/{repo}/{path}")


def flatten(pages):
    return [item for page in pages for item in page]


def load_pact(pid):
    return yaml.safe_load((ROOT / "pact" / "changes" / pid / "pact.yaml").read_text())


def task_body(pid, pact, task, link):
    req = next(r for r in pact["requirements"] if r["id"] == task["requirement"])
    return (
        f"Part of PACT [`{pid}`]({link}): {pact['title']}\n\n"
        f"**Requirement** `{req['id']}`\n\n{req['statement']}\n\n"
        f"**Task** `{task['id']}`\n\n{task['verification']}\n\n"
        "Done when the evidence is attached to the PACT and `okg pact check "
        f"{pid}` shows the requirement proven."
    )


def epic_body(pid, pact, link, task_refs):
    lines = "\n".join(f"- [ ] {ref}" for ref in task_refs)
    return (
        f"Tracks PACT [`{pid}`]({link}).\n\n**Why.** {pact['why']}\n\n"
        f"**Tasks**\n\n{lines}\n\n"
        "Design: `docs/adr/0002-archi-3.0-release.md`."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--ref", default="archi_v3", help="branch the PACT links point at")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    plan = yaml.safe_load(PLAN.read_text())
    repo = args.repo or plan["repo"]
    dry = not args.apply

    def act(msg, fn):
        print(("would " if dry else "") + msg)
        return None if dry else fn()

    have_labels = {lab["name"] for lab in flatten(paged(repo, "labels?per_page=100"))}
    for name, spec in plan["labels"].items():
        if name not in have_labels:
            act(f"create label {name}", lambda n=name, s=spec: gh(
                f"repos/{repo}/labels", payload={"name": n, **s}))
    for pid, entry in plan["pacts"].items():
        name = f"pact:{pid}"
        if name not in have_labels:
            act(f"create label {name}", lambda n=name: gh(
                f"repos/{repo}/labels", payload={"name": n, "color": "c5def5"}))

    milestones = {
        m["title"]: m["number"]
        for m in flatten(paged(repo, "milestones?state=all&per_page=100"))
    }
    for m in plan["milestones"]:
        if m["title"] not in milestones:
            made = act(f"create milestone {m['title']}", lambda m=m: gh(
                f"repos/{repo}/milestones", payload=m))
            milestones[m["title"]] = made["number"] if made else None

    issues = {
        i["title"]: i
        for i in flatten(paged(repo, "issues?state=all&labels=archi-3.0&per_page=100"))
        if "pull_request" not in i
    }

    for pid, entry in plan["pacts"].items():
        pact = load_pact(pid)
        link = BLOB.format(repo=repo, ref=args.ref, pid=pid)
        labels = ["archi-3.0", f"pact:{pid}", *entry.get("labels", [])]
        ms = milestones.get(entry["milestone"])
        manifest_tasks = {t["id"]: t for t in pact["tasks"]}
        refs = []
        for tid, title in entry["tasks"].items():
            task = manifest_tasks[tid]
            if title in issues:
                refs.append(f"#{issues[title]['number']}")
                continue
            made = act(f"create issue [{pid}] {title}", lambda t=title, tk=task: gh(
                f"repos/{repo}/issues",
                payload={"title": t, "body": task_body(pid, pact, tk, link),
                         "labels": labels, "milestone": ms}))
            refs.append(f"#{made['number']}" if made else title)
        epic_title = f"[epic] {pact['title']}"
        body = epic_body(pid, pact, link, refs)
        if epic_title in issues:
            num = issues[epic_title]["number"]
            act(f"refresh epic #{num}", lambda n=num: gh(
                "-X", "PATCH", f"repos/{repo}/issues/{n}", payload={"body": body}))
        else:
            act(f"create epic {epic_title}", lambda: gh(
                f"repos/{repo}/issues",
                payload={"title": epic_title, "body": body,
                         "labels": [*labels, "epic"], "milestone": ms}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
