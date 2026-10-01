#!/usr/bin/env python3
"""Build one self-contained HTML page to review `archi eval qa` runs question by question.

Usage:
  python scripts/eval/review_page.py --dataset datasets/v1/benchmark.json --labels datasets/v1/labels.json \
      --out review.html prod=runs/prod new-model=runs/new-model

For every question the page shows the gold answer and its atoms (required or optional, answer source), and for
every setup: the score, each atom's verdict with the judge's reason, the answer, and every tool call with its
input and the start of its output. The question list can be searched, filtered by label and sorted by any
setup's score. No network access is needed to open it.

The page contains your questions, answers and tool outputs: share it only where those may go.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def build_data(dataset: list, labels: dict, setups: list, tool_chars: int, answer_chars: int) -> dict:
    items = {}
    for row in dataset:
        lab = labels.get(row["id"], {})
        src = {a["id"]: a for a in lab.get("atoms", [])}
        items[row["id"]] = {
            "question": row["question"], "gold": row.get("answer") or "",
            "labels": {k: v for k, v in lab.items() if k not in ("atoms", "source_file") and not isinstance(v, (dict, list))},
            "atoms": [{"id": a["id"], "text": a["text"], "required": bool(a.get("required")),
                       "source": src.get(a["id"], {}).get("answer_source")} for a in row.get("expected_atoms") or []],
            "runs": {},
        }
    for name, run in setups:
        evals = {e["attempt_id"]: e for e in read_jsonl(run / "evaluation_results.jsonl")}
        for a in read_jsonl(run / "answers.jsonl"):
            item = items.get(a["item_id"])
            if item is None:
                continue
            e = evals.get(a["attempt_id"], {})
            verdicts = {j.get("atom_id"): j for j in e.get("judgments") or []}
            req = [x["id"] for x in item["atoms"] if x["required"]]
            answered = a.get("status") == "answer_ready"
            score = (sum(verdicts.get(x, {}).get("outcome") == "entailed" for x in req) / len(req)
                     if answered and verdicts and req else (0.0 if not answered else None))
            item["runs"].setdefault(name, []).append({
                "n": a.get("ordinal"), "status": a.get("status"), "score": score,
                "s": round(int(a.get("duration_ms") or 0) / 1000),
                "answer": (a.get("answer") or (a.get("error") or {}).get("message", "") or "")[:answer_chars],
                "verdicts": {k: [v.get("outcome"), (v.get("rationale") or "")[:600]] for k, v in verdicts.items()},
                "tools": [[t.get("name"), str(t.get("query") or "")[:2000], int(t.get("duration_ms") or 0),
                           t.get("status") or "", str(t.get("response") or t.get("error") or "")[:tool_chars]]
                          for t in a.get("tool_calls") or []],
            })
    return {"setups": [n for n, _ in setups], "items": items}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mute:#6b6b70;--line:#e3e3e8;--card:#f7f7f9;--good:#1f7a3a;--bad:#b3261e;--mid:#8a6d00;--acc:#2458d6}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececf0;--mute:#9a9aa2;--line:#303036;--card:#1f1f23;--good:#5cc47a;--bad:#ff7b72;--mid:#e3b341;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:16px 16px 8px}h1{font-size:18px;margin:0 0 4px}p.note{color:var(--mute);margin:0}
.bar{display:flex;flex-wrap:wrap;gap:8px;padding:8px 16px;border-bottom:1px solid var(--line)}
input,select{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.wrap{display:grid;grid-template-columns:minmax(260px,420px) 1fr;min-height:70vh}
@media (max-width:900px){.wrap{grid-template-columns:1fr}}
.list{border-right:1px solid var(--line);overflow:auto;max-height:85vh}
table{border-collapse:collapse;width:100%}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--bg);cursor:pointer;font-weight:600;white-space:nowrap}
tr.q{cursor:pointer}tr.q:hover,tr.q.sel{background:var(--card)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.detail{padding:12px 16px;overflow:auto;max-height:85vh}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
pre{white-space:pre-wrap;word-break:break-word;margin:6px 0;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
.good{color:var(--good)}.bad{color:var(--bad)}.mid{color:var(--mid)}.mute{color:var(--mute)}.tag{font-size:12px;color:var(--mute)}
details{margin:4px 0}summary{cursor:pointer}.pill{display:inline-block;padding:0 6px;border:1px solid var(--line);border-radius:10px;font-size:12px;margin-right:4px}
</style></head><body>
<header><h1>__TITLE__</h1><p class="note">Score = share of a question's required atoms judged entailed (mean over attempts; a failed attempt counts 0). Click a column to sort, a row to open it.</p></header>
<div class="bar"><input id="search" placeholder="Search questions" size="28"><select id="lkey"><option value="">Filter by label…</option></select><select id="lval" hidden></select><span id="count" class="tag"></span></div>
<div class="wrap"><div class="list"><table><thead id="head"></thead><tbody id="rows"></tbody></table></div><div class="detail" id="detail"><p class="mute">Select a question.</p></div></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent), S = D.setups, IDS = Object.keys(D.items);
const el = (tag, attrs, ...kids) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs || {})) { if (k === 'class') e.className = v; else e.setAttribute(k, v); } for (const c of kids) e.append(c instanceof Node ? c : document.createTextNode(c == null ? '' : String(c))); return e; };
const mean = xs => { const v = xs.filter(x => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const score = (q, s) => mean((D.items[q].runs[s] || []).map(r => r.score));
const pct = x => x == null ? '–' : Math.round(100 * x) + '%';
const cls = x => x == null ? 'mute' : x >= 0.999 ? 'good' : x < 0.5 ? 'bad' : 'mid';
let sortKey = 'id', sortDir = 1, current = null;
const keys = [...new Set(IDS.flatMap(q => Object.keys(D.items[q].labels)))].sort();
for (const k of keys) document.getElementById('lkey').append(el('option', {value: k}, k));
document.getElementById('lkey').onchange = e => { const k = e.target.value, sel = document.getElementById('lval'); sel.replaceChildren(el('option', {value: ''}, 'any')); sel.hidden = !k; if (k) for (const v of [...new Set(IDS.map(q => String(D.items[q].labels[k])))].sort()) sel.append(el('option', {value: v}, v)); draw(); };
document.getElementById('lval').onchange = draw; document.getElementById('search').oninput = draw;
function visible() { const s = document.getElementById('search').value.toLowerCase(), k = document.getElementById('lkey').value, v = document.getElementById('lval').value; return IDS.filter(q => (!s || (q + ' ' + D.items[q].question).toLowerCase().includes(s)) && (!k || !v || String(D.items[q].labels[k]) === v)); }
function draw() {
  const head = el('tr', {}, el('th', {'data-k': 'id'}, 'question'), ...S.map(s => el('th', {'data-k': s}, s)));
  head.querySelectorAll('th').forEach(th => th.onclick = () => { const k = th.getAttribute('data-k'); sortDir = sortKey === k ? -sortDir : (k === 'id' ? 1 : -1); sortKey = k; draw(); });
  document.getElementById('head').replaceChildren(head);
  const qs = visible().sort((a, b) => sortKey === 'id' ? sortDir * a.localeCompare(b) : sortDir * ((score(a, sortKey) ?? -1) - (score(b, sortKey) ?? -1)));
  document.getElementById('rows').replaceChildren(...qs.map(q => { const tr = el('tr', {class: 'q' + (q === current ? ' sel' : '')}, el('td', {}, el('b', {}, q), el('div', {class: 'tag'}, D.items[q].question.slice(0, 110))), ...S.map(s => el('td', {class: 'n ' + cls(score(q, s))}, pct(score(q, s))))); tr.onclick = () => show(q); return tr; }));
  const all = S.map(s => s + ' ' + pct(mean(qs.map(q => score(q, s)))));
  document.getElementById('count').textContent = qs.length + ' questions · mean: ' + all.join(' · ');
}
function show(q) {
  current = q; draw(); const it = D.items[q], d = document.getElementById('detail');
  const labels = el('div', {}, ...Object.entries(it.labels).map(([k, v]) => el('span', {class: 'pill'}, k + ': ' + v)));
  const atomRows = it.atoms.map(a => el('tr', {}, el('td', {}, a.id, el('div', {class: 'tag'}, (a.required ? 'required' : 'optional') + (a.source ? ' · ' + a.source : ''))), el('td', {}, a.text), ...S.map(s => { const rs = it.runs[s] || []; return el('td', {}, ...rs.map(r => { const v = r.verdicts[a.id]; const o = v ? v[0] : (r.status === 'answer_ready' ? 'not scored' : 'failed'); return el('div', {class: o === 'entailed' ? 'good' : o === 'contradicted' ? 'bad' : 'mute', title: v ? v[1] : ''}, o); })); })));
  const setups = S.map(s => el('div', {class: 'card'}, el('b', {}, s), ...(it.runs[s] || []).map(r => el('div', {},
    el('div', {class: 'tag'}, 'attempt ' + r.n + ' · ' + r.status + ' · score ' + pct(r.score) + ' · ' + r.s + ' s · ' + r.tools.length + ' tool calls'),
    el('details', {open: ''}, el('summary', {}, 'Answer'), el('pre', {}, r.answer)),
    el('details', {}, el('summary', {}, 'Judge reasons'), ...Object.entries(r.verdicts).map(([k, v]) => el('div', {}, el('b', {}, k + ': '), el('span', {class: v[0] === 'entailed' ? 'good' : v[0] === 'contradicted' ? 'bad' : 'mute'}, v[0]), ' — ', v[1]))),
    el('details', {}, el('summary', {}, 'Tool calls (' + r.tools.length + ')'), ...r.tools.map((t, i) => el('details', {}, el('summary', {}, (i + 1) + '. ' + t[0] + ' · ' + t[2] + ' ms' + (t[3] && t[3] !== 'success' ? ' · ' + t[3] : '')), el('div', {class: 'tag'}, 'input'), el('pre', {}, t[1]), el('div', {class: 'tag'}, 'output (start)'), el('pre', {}, t[4]))))))));
  d.replaceChildren(el('h2', {}, q), labels, el('div', {class: 'card'}, el('b', {}, 'Question'), el('pre', {}, it.question)), el('div', {class: 'card'}, el('details', {}, el('summary', {}, el('b', {}, 'Gold answer')), el('pre', {}, it.gold))),
    el('div', {class: 'card'}, el('b', {}, 'Atoms'), el('table', {}, el('thead', {}, el('tr', {}, el('th', {}, 'atom'), el('th', {}, 'text'), ...S.map(s => el('th', {}, s)))), el('tbody', {}, ...atomRows))), ...setups);
  try { localStorage.setItem('eval-review-q', q); } catch (e) {}
}
draw(); let start = IDS[0]; try { const s = localStorage.getItem('eval-review-q'); if (s && D.items[s]) start = s; } catch (e) {} if (start) show(start);
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="benchmark.json")
    ap.add_argument("--labels", type=Path, help="labels.json (answer sources and item labels)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="Evaluation review")
    ap.add_argument("--tool-output-chars", type=int, default=1000, help="characters kept of each tool output")
    ap.add_argument("--answer-chars", type=int, default=20000)
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR")
    args = ap.parse_args()
    setups = []
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=RUN_DIR, got {spec!r}")
        setups.append((name, Path(path)))
    labels = json.loads(args.labels.read_text()) if args.labels else {}
    data = build_data(json.loads(args.dataset.read_text()), labels, setups, args.tool_output_chars, args.answer_chars)
    # every "<" as \u003c: valid JSON, and no "</script>" or "<!--" can end the data block early
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = PAGE.replace("__TITLE__", html.escape(args.title)).replace("__DATA__", blob)
    args.out.write_text(page)
    print(f"wrote {args.out} ({len(page) / 1e6:.1f} MB, {len(data['items'])} questions, {len(setups)} setups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
