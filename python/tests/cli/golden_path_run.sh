#!/usr/bin/env bash
# req.r3-golden-path.init / .up / .nightly: an empty directory to a cited answer.
#
# Usage: golden_path_run.sh <empty work dir>
#
# Runs the user commands of the golden path with the cern-team bundle, times
# each one, and checks each result:
#   1. archi init gp --bundle cern-team     no answers: the bundle's defaults (docsite,
#                                           indico) publish; instance repo, secrets, graph
#   2. archi up                             graph-db, chat-db, mcp, webui; okg chat sync
#      (then, not a user command: archi down + archi up keeps the same published graph)
#   3. archi doctor --json                  every check passes
#   4. mcp_cited_answer.py "<question>"     a search over MCP that cites node ids
# (The nightly workflow adds the install as its first command.)
#
# Needs docker or podman with compose, network access (images, the public
# documentation pages and Indico event the defaults read, the okg wheel), and okg + archi installed in the Python
# environment whose bin/ holds `python` (set V to that bin/; default: PATH).
# KEEP=1 leaves the stack running; otherwise it ends with `archi down --purge --yes`.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
V="${V:-$(dirname "$(command -v python)")}"
RUN="$1"
QUESTION="${QUESTION:-What is the Knowledge Bases and Agents for CERN workshop?}"
mkdir -p "$RUN"
[ -z "$(ls -A "$RUN")" ] || { echo "$RUN is not empty"; exit 1; }
RUN="$(cd "$RUN" && pwd)"
INST="$RUN/gp"
unset OKG_PROFILES_DIR

started=$(date +%s)
step() { echo; echo "== $(( $(date +%s) - started ))s: $*"; }

cleanup() {
  if [ "${KEEP:-0}" != 1 ] && [ -f "$INST/archi.yaml" ]; then
    "$V/archi" down --instance "$INST" --purge --yes || true
  fi
}
trap cleanup EXIT

step "1. archi init"
"$V/archi" init gp --bundle cern-team --dir "$INST" --data-root "$RUN/data"

step "2. archi up"
"$V/archi" up --instance "$INST"

# Not a user command: the graph must survive `archi down` + `archi up`
# (the published generation and its node count are unchanged).
published() {
  "$V/archi" doctor --instance "$INST" --json | "$V/python" -c '
import json, sys
c = next(c for c in json.load(sys.stdin)["checks"] if c["id"] == "published")
print(c["message"], (c.get("data") or {}).get("nodes"))'
}
step "durability: archi down, archi up, same graph"
before="$(published)"
"$V/archi" down --instance "$INST"
"$V/archi" up --instance "$INST"
after="$(published)"
echo "  before: $before"
echo "  after:  $after"
[ "$before" = "$after" ] && [ "${before#published }" != "$before" ] || {
  echo "the published graph did not survive archi down + archi up"; exit 1; }

step "3. archi doctor --json"
"$V/archi" doctor --instance "$INST" --json > "$RUN/doctor.json" || {
  "$V/archi" doctor --instance "$INST"; exit 1; }
"$V/python" - "$RUN/doctor.json" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
for check in report["checks"]:
    print(f"  {check['status']:4}  {check['id']}: {check['message'][:110]}")
assert report["ok"], "archi doctor reported a failure"
EOF

step "4. a cited answer over MCP"
"$V/python" "$REPO/python/tests/cli/mcp_cited_answer.py" --instance "$INST" "$QUESTION" \
  | tee "$RUN/answer.json"

step "evidence"
git -C "$INST" log --format='  %h %s'
echo "  .env tracked: $(git -C "$INST" ls-files .env | wc -l | tr -d ' ')"
echo "  total: $(( $(date +%s) - started ))s"
