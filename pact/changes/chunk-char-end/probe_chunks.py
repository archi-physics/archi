"""Rebuild each real record from its emitted chunks the way okg dev reads it.

Read-only. Runs the real archi reader (PYTHONPATH picks main or branch),
captures the exact text each emitter hands to _chunks, collects the emitted
document_chunk attrs, then applies okg's own order_segments + trim_overlap
(imported from okg origin/dev) and joins the segment texts. Compares the
result to the chunked text.

usage: probe_chunks.py <kind> <out.json> [args...]
  twiki  <eos_root> [web_root] [physics_filter 0/1]
  jira   <records.json>
  hn     <records.json>
  docs   <records.json>
"""
import hashlib
import json
import sys
import time
from collections import Counter

import archi
from okg.deployment import NodeFact
from okg.substrate.mcp import record_text as rt

kind, out_path, *args = sys.argv[1:]
CURRENT = {"text": None}


def recorder(module):
    real = module._chunks

    def wrapped(text):
        CURRENT["text"] = text
        yield from real(text)

    module._chunks = wrapped


if kind == "twiki":
    import archi.sources.twiki as mod
    kwargs = {"eos_root": args[0]}
    if len(args) > 1:
        kwargs["web_root"] = args[1]
    if len(args) > 2:
        kwargs["physics_filter"] = args[2] == "1"
    source = mod.TwikiEOSSource(**kwargs)
    run = lambda: source.run("probe", mode="scope_complete")  # noqa: E731
elif kind == "jira":
    import archi.sources.jira as mod
    meta = args[0].rsplit("/", 1)[0] + "/meta.json"
    source = mod.JiraIssueSource(records_path=args[0], meta_path=meta)
    run = lambda: source.run("probe", mode="scope_complete")  # noqa: E731
elif kind == "hn":
    import archi.sources.hypernews as mod
    source = mod.HyperNewsSource(records_path=args[0])
    run = lambda: source.run("probe", mode="scope_complete")  # noqa: E731
elif kind == "docs":
    import archi.sources.docs as mod
    source = mod.DocumentationSource(records_path=args[0])
    run = lambda: source.run("probe", mode="scope_complete")  # noqa: E731
else:
    raise SystemExit(f"unknown kind {kind}")
recorder(mod)

t0 = time.time()
result = run()
pages = {}
for fact in result.facts:
    if not (isinstance(fact, NodeFact) and fact.subtype == "document_chunk"):
        continue
    key_fields = {k: v for k, v in fact.source_record_id.items() if k != "chunk_index"}
    key = json.dumps(key_fields, sort_keys=True)
    page = pages.setdefault(key, {"source": CURRENT["text"], "chunks": []})
    assert page["source"] is CURRENT["text"] or page["source"] == CURRENT["text"]
    page["chunks"].append((fact.node_id, dict(fact.attrs)))


def pair_status(prev, cur):
    if prev.char_end is None or cur.char_end is None:
        return "no_char_end"
    if not (rt._exact(prev) and rt._exact(cur)):
        return "not_exact"
    if not prev.char_offset < cur.char_offset < prev.char_end:
        return "no_overlap_gap" if cur.char_offset >= prev.char_end else "offsets_not_increasing"
    overlap = prev.char_end - cur.char_offset
    if overlap > len(cur.text):
        return "overlap_longer_than_chunk"
    if not prev.text.endswith(cur.text[:overlap]):
        return "text_mismatch"
    return "trimmed"


summary = Counter()
causes = Counter()
pair_counts = Counter()
examples = {}
ids = hashlib.sha256()
total_chunks = 0
for key in sorted(pages):
    page = pages[key]
    source_text = page["source"]
    segs = []
    for node_id, a in page["chunks"]:
        ids.update(node_id.encode())
        ids.update(a["content_sha256"].encode())
        segs.append(rt._Segment(
            chunk_id=node_id, text=a["text"], source=None, untrusted_origin=None,
            chunk_ordinal=None, chunk_total=None,
            char_offset=rt._int_or_none(a.get("char_offset")),
            char_end=rt._int_or_none(a.get("char_end")),
            heading_path=None, redacted=False,
        ))
    total_chunks += len(segs)
    summary["chunks_offset_not_window_multiple"] += sum(
        1 for s in segs if s.char_offset is not None and s.char_offset % 3800
    )
    ordered, basis = rt.order_segments(segs)
    statuses = [pair_status(p, c) for p, c in zip(ordered, ordered[1:])]
    pair_counts.update(statuses)
    for (p, c), st in zip(zip(ordered, ordered[1:]), statuses):
        if st == "no_overlap_gap":
            gap = source_text[p.char_end:c.char_offset]
            summary["gap_chars"] += len(gap)
            summary["gap_whitespace_only" if not gap.strip() else "gap_has_text"] += 1
    rt.trim_overlap(ordered)
    rebuilt = "".join(s.text for s in ordered)
    summary["pages"] += 1
    summary["multi_chunk_pages"] += len(segs) > 1
    summary["basis_" + basis] += 1
    if len(rebuilt) == len(source_text):
        summary["length_equal"] += 1
    if rebuilt == source_text:
        summary["byte_identical"] += 1
        continue
    # Group the mismatch by cause.
    bad = sorted({s for s in statuses if s != "trimmed"})
    if bad:
        cause = "pairs_not_trimmed:" + "+".join(bad)
    elif rebuilt == source_text.strip():
        cause = "page_edge_whitespace_only"
    elif rebuilt.replace("\x00", " ") == source_text.replace("\x00", " "):
        cause = "nul_replaced"
    elif rebuilt == source_text.strip().replace("\x00", " "):
        cause = "page_edge_whitespace_and_nul"
    else:
        cause = "other"
    causes[cause] += 1
    ex = examples.setdefault(cause, [])
    if len(ex) < 5:
        ex.append({"key": key, "source_len": len(source_text),
                   "rebuilt_len": len(rebuilt), "chunks": len(segs)})

named = {}
for key, page in pages.items():
    if len(page["source"]) == 39173:
        named.setdefault("source_len_39173", []).append(key)
    if "EXO16010" in key:
        named.setdefault("EXO16010", []).append(key)

out = {
    "archi": archi.__file__,
    "okg_record_text": rt.__file__,
    "kind": kind,
    "args": args,
    "health": getattr(result.health, "status", None),
    "record_count": getattr(result.health, "record_count", None),
    "total_chunks": total_chunks,
    "chunk_id_and_sha_digest": ids.hexdigest(),
    "summary": dict(summary),
    "pair_status": dict(pair_counts),
    "mismatch_causes": dict(causes),
    "examples": examples,
    "named": named,
    "seconds": round(time.time() - t0, 1),
}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("archi", "health", "record_count", "total_chunks",
      "summary", "pair_status", "mismatch_causes", "named", "seconds")}, indent=1))
