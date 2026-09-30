#!/usr/bin/env python3
"""Export a golden set (one YAML file per question) to an `archi eval qa` dataset.

Input: a directory of item files, one question per file (see golden_item_template.yaml).
Output, in --out:

  benchmark.json        the dataset for `archi eval qa --dataset` (id, question, answer,
                        time_sensitive, category, answer_mode, answer_source, expected_atoms)
  labels.json           per-item and per-atom labels used by report.py and review_page.py
                        (the dataset schema rejects extra fields, so they live here)
  export_manifest.json  counts, skipped items, warnings, input and output sha256

Mapping:
  - answer          = the item's `gold`, plus its `past_answers` under
                      "Outdated answers (no longer true):". The judge sees the answer as its
                      reference, so it can tell an outdated claim from a current one.
  - expected_atoms  = every atom with its id, text and required flag. Supplying atoms means
                      the evaluator never extracts them itself.
  - category        = labels.task_type (or "general").
  - time_sensitive  = false. Refresh the gold of live questions on the run day instead; for
                      Archi's live-oracle items (Dataset V2), write those rows by hand.
  - labels.json     = the item's `labels`, plus computed `multi_source` (single, multi_static,
                      multi_live, static_live) and `changed_over_time`, plus each atom's
                      answer_source / implicit / reachable.

Only items with `status: approved` are exported, unless --all is given.

Usage:
  python scripts/eval/export_dataset.py --set golden/ --out datasets/v1
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

OUTDATED_ANSWERS_HEADER = "Outdated answers (no longer true):"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_live(source: str) -> bool:
    return source.startswith("live") or source == "monitoring"


def source_options(value) -> list:
    """'a or b' -> [{a}, {b}] (either suffices); 'a+b' -> [{a, b}] (both needed)."""
    return [{p.strip() for p in alt.split("+") if p.strip()} for alt in str(value).split(" or ")]


def multi_source(required_atoms: list) -> str:
    """The smallest set of sources that answers every required atom, as a label."""
    per_atom = [source_options(a["answer_source"]) for a in required_atoms if a.get("answer_source")]
    if not per_atom:
        return "unknown"
    sources = sorted(set().union(*(opt for opts in per_atom for opt in opts)))
    for size in range(1, len(sources) + 1):
        for cover in itertools.combinations(sources, size):
            chosen = set(cover)
            if all(any(opt <= chosen for opt in opts) for opts in per_atom):
                if size == 1:
                    return "single"
                live = {s for s in chosen if is_live(s)}
                if live and chosen - live:
                    return "static_live"
                return "multi_live" if live else "multi_static"
    return "unknown"


def export_answer(doc: dict) -> str:
    gold = str(doc.get("gold") or "").strip()
    past = doc.get("past_answers") or []
    if not past:
        return gold
    lines = [f"- {p['text']} ({p.get('changed', 'no longer true')})" for p in past]
    return gold + "\n\n" + OUTDATED_ANSWERS_HEADER + "\n" + "\n".join(lines)


def check_item(doc: dict, path: Path) -> list:
    """Hard errors for one item (the export stops on any)."""
    errors = []
    for key in ("id", "question", "gold", "atoms"):
        if not doc.get(key):
            errors.append(f"{path.name}: `{key}` missing or empty")
    atoms = doc.get("atoms") or []
    ids = [str(a.get("id")) for a in atoms]
    if len(ids) != len(set(ids)):
        errors.append(f"{path.name}: duplicate atom ids")
    for a in atoms:
        if not a.get("id") or not str(a.get("text") or "").strip():
            errors.append(f"{path.name}: every atom needs `id` and `text`")
    if atoms and not any(a.get("required") for a in atoms):
        errors.append(f"{path.name}: no required atom")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, type=Path, help="directory with one YAML file per question")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--all", action="store_true", help="also export items that are not `status: approved`")
    args = ap.parse_args()

    files = sorted(p for p in args.set.rglob("*") if p.suffix in (".yaml", ".yml") and not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"no .yaml files under {args.set}")
    dataset, labels, warnings, errors, skipped, inputs, seen = [], {}, [], [], [], {}, set()
    for path in files:
        doc = yaml.safe_load(path.read_text()) or {}
        if not isinstance(doc, dict) or "question" not in doc:
            continue  # not an item file (e.g. a template or notes)
        if not args.all and str(doc.get("status", "")).lower() != "approved":
            skipped.append(str(doc.get("id") or path.name))
            continue
        item_errors = check_item(doc, path)
        if item_errors:
            errors += item_errors
            continue
        item_id = str(doc["id"])
        if item_id in seen:
            errors.append(f"{path.name}: duplicate item id {item_id}")
            continue
        seen.add(item_id)
        inputs[str(path)] = sha256(path)
        lab = dict(doc.get("labels") or {})
        atoms = doc["atoms"]
        required = [a for a in atoms if a.get("required")]
        no_source = [a["id"] for a in required if not a.get("answer_source")]
        if no_source:
            warnings.append(f"{item_id}: required atoms without answer_source {no_source} (left out of per-source scores)")
        computed = {"multi_source": multi_source(required),
                    "changed_over_time": "yes" if doc.get("past_answers") else "no"}
        for key, value in computed.items():
            if key in lab and lab[key] != value:
                warnings.append(f"{item_id}: labels.{key}={lab[key]!r} differs from computed {value!r}; computed used")
        dataset.append({
            "id": item_id,
            "question": str(doc["question"]).strip(),
            "answer": export_answer(doc),
            "time_sensitive": False,
            "category": str(lab.get("task_type") or "general"),
            "answer_mode": "direct_answer",
            "answer_source": path.name,
            "expected_atoms": [{"id": str(a["id"]), "text": str(a["text"]).strip(), "required": bool(a.get("required"))}
                               for a in atoms],
        })
        labels[item_id] = {
            **{k: v for k, v in lab.items() if k not in computed},
            **computed,
            "tier": doc.get("tier"),
            "time_sensitive": bool(doc.get("time_sensitive")),
            "provenance": lab.get("provenance") or doc.get("provenance"),
            "source_file": path.name,
            "atoms": [{"id": str(a["id"]), "required": bool(a.get("required")), "answer_source": a.get("answer_source"),
                       "implicit": bool(a.get("implicit")), "reachable": a.get("reachable", True)} for a in atoms],
        }
    if errors:
        raise SystemExit("export stopped:\n  " + "\n  ".join(errors))

    args.out.mkdir(parents=True, exist_ok=True)
    dataset.sort(key=lambda r: r["id"])
    bench, lab_path = args.out / "benchmark.json", args.out / "labels.json"
    bench.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    lab_path.write_text(json.dumps(dict(sorted(labels.items())), ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "set": str(args.set),
        "items": len(dataset),
        "required_atoms": sum(1 for r in dataset for a in r["expected_atoms"] if a["required"]),
        "skipped_not_approved": skipped,
        "warnings": warnings,
        "inputs_sha256": inputs,
        "outputs_sha256": {bench.name: sha256(bench), lab_path.name: sha256(lab_path)},
    }
    (args.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("items", "required_atoms", "skipped_not_approved", "warnings")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
