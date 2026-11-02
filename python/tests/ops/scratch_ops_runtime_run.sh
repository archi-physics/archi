#!/usr/bin/env bash
# PACT r3-ops-runtime on a real scratch instance (okg 0.2.0 worker, stub proposer, marker mediator):
#   acc.r3-ops-runtime.e2e      L0, L1 and L2 (approve, reject) run end to end; records published and read back
#   acc.r3-ops-runtime.timeout  a timed-out proposal does not execute; a late approval is refused, also after a
#                               worker restart (the mediator step's own guard)
#   acc.r3-ops-runtime.halt     `archi ops halt` stops an approved run before its mediator, and a halt that does
#                               not cancel is still refused by the mediator step itself
#   acc.r3-ops-runtime.demote   a stale source drops a live automation to shadow, with the reason recorded;
#                               the demotion persists until `archi ops promote`
#   acc.r3-ops-runtime.score    a shadow report that passes promotion_readiness, then a PACT-recorded promotion
#   acc.r3-ops-runtime.no-l3    L3 in archi.yaml is refused at run start
#   req.r3-ops-runtime.control  per-run and per-day budgets; `archi ops status|pending|approve|reject|halt|resume`
#   at-most-once mediators      a replay after a crash mid-call does not call the tool again
#
# Usage: ARCHI_SCRATCH_DSN=postgresql://... scratch_ops_runtime_run.sh <empty work dir>
#
# Run it from a git checkout of the commit under test: it prints that commit and refuses a dirty checkout.
# Needs an EMPTY database on an AKMON Postgres (ghcr.io/mitdbg/akmon-postgres), okg 0.2.0 and archi installed
# (editable, from this checkout) in the Python environment whose bin/ holds `python` (set V to that bin/).
# Starts one `okg runtime worker` at a time in the background; its PID goes to <work dir>/pids and it is the only
# process this script kills. No paid LLM calls: the proposer is a stub.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
V="${V:-$(dirname "$(command -v python)")}"
RUN="$1"
DSN="${ARCHI_SCRATCH_DSN:?set ARCHI_SCRATCH_DSN to an empty AKMON database}"
F="$REPO/python/tests/ops/fixtures/scratch"
mkdir -p "$RUN"
[ -z "$(ls -A "$RUN")" ] || { echo "$RUN is not empty"; exit 1; }
RUN="$(cd "$RUN" && pwd)"
cd "$RUN"
unset OKG_PROFILES_DIR
D="$(basename "$DSN")"   # the deployment is named after the database, as archi init names its own
INST="$RUN/instance"
export ARCHI_INSTANCE="$INST" OKG_DSN="$DSN" ARCHI_OPS_DEPLOYMENT="$D" OKG_DEPLOYMENTS_DIR="$INST/deployments"
export PYTHONDONTWRITEBYTECODE=1
T="$V/python $REPO/python/tests/ops/scratch_ops_runtime_tools.py"
OPS="$V/archi ops --instance $INST --dsn $DSN --deployment $D"
PIDS="$RUN/pids"
FAILS=0

check() {  # check <label> <python expression over stdin as `d`/`rows`>
  "$V/python" -c "
import json, sys
text = sys.stdin.read()
try:
    d = json.loads(text)
except ValueError:
    d = None
rows = [json.loads(l) for l in text.splitlines() if l.startswith('{') and l.rstrip().endswith('}')]
ok = $2
print(('PASS' if ok else 'FAIL') + ': $1')
sys.exit(0 if ok else 1)
" || FAILS=$((FAILS + 1))
}

start_worker() {
  # exec, so the subshell's PID is the worker's own PID (a plain `a && b &` forks a shell around it)
  ( cd "$INST" && exec nohup "$V/okg" runtime worker --deployment $D >> "$RUN/worker.log" 2>&1 ) &
  echo $! > "$RUN/worker.pid"
  cat "$RUN/worker.pid" >> "$PIDS"
  echo "worker started, pid $(cat "$RUN/worker.pid")"
  sleep "${1:-12}"
}
kill_worker() {  # kill -9 only the PID this script started
  kill -9 "$(cat "$RUN/worker.pid")" 2>/dev/null || true
  sleep 3
  if ps -p "$(cat "$RUN/worker.pid")" > /dev/null 2>&1; then echo "worker STILL RUNNING"; else echo "worker stopped"; fi
}
trigger() { "$V/okg" runtime trigger-schedule --deployment $D archi_ops_scan > "$RUN/trigger.out" 2>&1 || tail -3 "$RUN/trigger.out" >&2; }
publish() { "$V/okg" run --once --deployment $D --apply --json > "$RUN/publish-$1.json" 2>&1 || tail -3 "$RUN/publish-$1.json"; }
node() {  # node <type> <record id>: the published node, one JSON line
  "$V/okg" filter --deployment $D --type "$1" --where "attrs.record_id=$2" --all --jsonl 2>/dev/null
}
markers() { $T markers; }
new_run() {  # new_run <signal id>: add the signal, let the scanner start it, print the run id
  local wid; wid="$($T signal "$1")"; trigger; echo "$wid"
}
configure() { $T configure "$@"; }
logpos() { wc -c < "$RUN/worker.log" | tr -d ' '; }
in_log() {  # in_log <offset> <text>: does the worker log hold text after the offset? (prints a JSON bool)
  if tail -c +"$(($1 + 1))" "$RUN/worker.log" | grep -qF -- "$2"; then echo '{"found": true}'; else echo '{"found": false}'; fi
}
rb() { node "$1" "$2" > "$RUN/rb-$2.jsonl"; }

echo "== 0. the commit under test"
SHA="$(git -C "$REPO" rev-parse HEAD)"
echo "commit $SHA"
echo "{\"dirty\": $(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')}" | check "the checkout is clean" "d['dirty'] == 0"
[ "$FAILS" = 0 ] || exit 1

echo "== 1. inputs: cern-team regrouped into packs, the scratch automation's jira cache"
"$V/python" -c "
import sys; sys.path.insert(0, '$REPO/python/tests')
from pathlib import Path
from bundles.bundle_support import build_cern_team_catalog
build_cern_team_catalog(Path('$RUN/catalog'))
"
mkdir -p data/jira
cp "$REPO/python/tests/packs/fixtures/ops_record/jira_records_open.json" data/jira/records.json
echo '{"record_count": 2}' > data/jira/meta.json

echo "== 2. a fresh archi init (archi.cli.init.init itself; only the container start is skipped, the lab database runs)"
$T init-instance 2>&1 | tee "$RUN/install-flow.log" | grep -v "^\s*$" | tail -25
check "init rendered the scratch automation and the runtime line" \
  "'archi.ops.workflow' in open('$INST/profiles/cern-team/deployment-defaults.yaml').read() and 'scratch-triage' in open('$INST/archi.lock').read()" < /dev/null
check "init ran provision and then the runtime bootstrap by itself (no hand step)" \
  "'okg provision --deployment' in open('$RUN/install-flow.log').read() and 'okg runtime bootstrap --deployment' in open('$RUN/install-flow.log').read()" < /dev/null
grep -n "^runtime" -A4 "$OKG_DEPLOYMENTS_DIR/$D/deployment.yaml" | tee "$RUN/runtime-line.txt"
check "deployment.yaml has runtime.enabled and archi.ops.workflow (written before okg install)" \
  "'enabled: true' in open('$RUN/runtime-line.txt').read() and 'archi.ops.workflow' in open('$RUN/runtime-line.txt').read()" < /dev/null
$T tables | tee "$RUN/tables.json"
check "init created the archi_ops tables; nothing in this script did" \
  "d == {'signals': True, 'halts': True, 'mediations': True, 'demotions': True}" < "$RUN/tables.json"
ENSURE="from archi.ops.okg_io import OpsIO; OpsIO('$DSN', '$D').ensure_tables()"
"$V/python" -c "$ENSURE" & P1=$!; echo $P1 >> "$PIDS"
"$V/python" -c "$ENSURE"; R2=$?
wait $P1; R1=$?
echo "{\"a\": $R1, \"b\": $R2}" | check "two concurrent ensure_tables (archi up and a worker start) both succeed" "d == {'a': 0, 'b': 0}"
"$V/okg" runtime bootstrap --deployment $D > "$RUN/bootstrap-2.out" 2>&1; echo "{\"exit\": $?}" | check "okg runtime bootstrap is idempotent (archi up repeats it)" "d['exit'] == 0"
git -C "$INST" log --oneline | head -5

echo "== 3. start the worker; the scanner schedule exists"
start_worker 15
$V/python -c "
from archi.ops.okg_io import OpsIO
print(OpsIO('$DSN', '$D').schedules())" | tee "$RUN/schedules-0.txt"
check "the archi_ops_scan schedule is registered and active" "'archi_ops_scan' in open('$RUN/schedules-0.txt').read() and 'False' in open('$RUN/schedules-0.txt').read()" < /dev/null

echo "== 4. data: a seed incident the proposals cite"
"$V/python" -m archi.ops.record submit --deployment $D --records "$F/seed.jsonl" > "$RUN/seed.json"
check "seed incident admitted" "d['output']['status'] == 'accepted'" < "$RUN/seed.json"
publish seed
node ops_incident inc-seed-1 > "$RUN/seed-node.jsonl"
check "the seed incident is published" "len(rows) == 1 and rows[0]['attrs']['entity'] == 'T2_US_MIT'" < "$RUN/seed-node.jsonl"
"$V/okg" doctor --check sources --deployment $D --json > "$RUN/doctor-sources.json" 2>&1 || true
check "okg doctor lists the jira source with a watermark" \
  "any(r.get('name') == 'jira.jira' and ((r.get('coverage') or {}).get('watermark') or {}).get('last_reconcile_at') for r in d['sources'])" < "$RUN/doctor-sources.json"

echo "== 4b. live is a request: a hand-written mode: live is shadow until a person approves a promotion"
configure --level L2 --mode live --timeout 120 --proposal marker --cost 0.1
WH=$(new_run hand-1); $T wait "$WH" | tee "$RUN/hand.out"
check "ops.yaml says live with no promotion: the run is in shadow" \
  "d['status'] == 'SUCCESS' and 'observed' in str(d['output']) and \"'mode': 'shadow'\" in str(d['output'])" < "$RUN/hand.out"
publish hand
$T human-action "$WH"; publish human0
$OPS promote scratch-triage --as operator-x --min-proposals 1 --min-answered 1 --note "scratch: the shadow proposal matched a person" --json > "$RUN/promote1.json" 2> "$RUN/promote1.err"
cat "$RUN/promote1.json"; cat "$RUN/promote1.err"
CHANGE1=$("$V/python" -c "import json; print(json.load(open('$RUN/promote1.json'))['change'])")
check "promote opened a PACT change awaiting review and did not approve it" \
  "d['change'].startswith('promote-scratch-triage-') and d['state'] == 'awaiting_review' and d['thresholds']['min_proposals'] == 1" < "$RUN/promote1.json"
echo "{\"subject\": \"$(git -C "$INST" log -1 --format=%s)\", \"files\": \"$(git -C "$INST" show --name-only --format= HEAD | tr '\n' ' ')\"}" > "$RUN/promote1-commit.json"
check "the instance repo has the request, with the report and the thresholds" \
  "d['subject'] == 'request promotion of scratch-triage ($CHANGE1)' and 'pact/changes/$CHANGE1/shadow-report.json' in d['files'] and 'pact/changes/$CHANGE1/thresholds.json' in d['files']" < "$RUN/promote1-commit.json"
WP=$(new_run pre-1); $T wait "$WP" | tee "$RUN/pre.out"
check "ops.yaml names the change but nobody approved it: still shadow" \
  "d['status'] == 'SUCCESS' and \"'mode': 'shadow'\" in str(d['output'])" < "$RUN/pre.out"
$T pact "$CHANGE1" > "$RUN/pact0.json"
check "the ledger has no approval yet" "d == []" < "$RUN/pact0.json"
$T approve-promotion "$CHANGE1"
$T pact "$CHANGE1" > "$RUN/pact1.json"; cat "$RUN/pact1.json"
check "the operator approval (okg pact decide) is in the ledger" "len(d) == 1 and d[0]['verdict'] == 'approve' and d[0]['principal'] == 'operator-x'" < "$RUN/pact1.json"

echo "== 5. L0 (acc e2e): observed, nothing escalated, no marker"
configure --level L0
W0=$(new_run l0-1); $T wait "$W0" | tee "$RUN/l0.out"
check "L0 run finished observed" "d['status'] == 'SUCCESS' and 'observed' in str(d['output'])" < "$RUN/l0.out"
check "no marker at L0" "d == 0" < <(markers)

echo "== 6. L1: the draft is recorded, nothing is sent or run"
configure --level L1 --proposal draft
W1=$(new_run l1-1); $T wait "$W1" | tee "$RUN/l1.out"
check "L1 run finished drafted" "d['status'] == 'SUCCESS' and 'drafted' in str(d['output'])" < "$RUN/l1.out"
check "no marker at L1" "d == 0" < <(markers)

echo "== 7. L2 approve: escalation waits; pending and approve answer it; marker appears"
configure --level L2 --proposal marker --timeout 180
W2=$(new_run l2a-1); ESC2=$($T pending "$W2" --timeout 120); echo "escalation $ESC2"
$OPS pending | tee "$RUN/pending-2.txt"
check "ops pending lists the escalation" "'$ESC2' in open('$RUN/pending-2.txt').read()" < /dev/null
$OPS status --json > "$RUN/status-2.json" 2>"$RUN/status-2.err" || cat "$RUN/status-2.err"
check "status shows level L2, mode live, one pending escalation and the pack budget" \
  "d['automations'][0]['level'] == 'L2' and d['automations'][0]['mode'] == 'live' and len(d['pending']) == 1 and d['scanner'] == {'archi_ops_scan': 'active'} and d['automations'][0]['budget']['per_day_cap'] == 10" < "$RUN/status-2.json"
check "no marker before approval" "d == 0" < <(markers)
$OPS approve "$ESC2" --as scratch-op --note "scratch approve"
$T wait "$W2" | tee "$RUN/l2a.out"
check "L2 approved run executed and the outcome resolved" "d['status'] == 'SUCCESS' and 'executed' in str(d['output']) and 'resolved' in str(d['output'])" < "$RUN/l2a.out"
check "the marker was written exactly once" "d == 1" < <(markers)
$T escalation "$W2" > "$RUN/esc-2.json"
check "the escalation is applied, decided by the OS user for scratch-op" "d['status'] == 'applied' and d['decided_by'].endswith(' for scratch-op')" < "$RUN/esc-2.json"
$OPS approve "$ESC2" > "$RUN/re-approve.out" 2>&1; echo "exit $?" >> "$RUN/re-approve.out"; cat "$RUN/re-approve.out"
check "approving an already-applied escalation exits non-zero with a clear message" \
  "'is applied' in open('$RUN/re-approve.out').read() and 'can no longer be answered' in open('$RUN/re-approve.out').read() and 'exit 2' in open('$RUN/re-approve.out').read()" < /dev/null

echo "== 8. L2 reject: no marker"
W3=$(new_run l2r-1); ESC3=$($T pending "$W3" --timeout 120)
$OPS reject "$ESC3" --as scratch-op --note "not now"
$T wait "$W3" | tee "$RUN/l2r.out"
check "rejected run ends rejected" "d['status'] == 'SUCCESS' and 'rejected' in str(d['output'])" < "$RUN/l2r.out"
check "still one marker" "d == 1" < <(markers)

echo "== 9. L2 timeout (acc timeout): nobody answers; a late approval is refused; no marker"
configure --level L2 --proposal marker --timeout 15
W4=$(new_run l2t-1); ESC4=$($T pending "$W4" --timeout 60)
$T wait "$W4" --timeout 90 | tee "$RUN/l2t.out"
check "timed-out run ends timed_out" "d['status'] == 'SUCCESS' and 'timed_out' in str(d['output'])" < "$RUN/l2t.out"
$OPS approve "$ESC4" --as scratch-op > "$RUN/late-approve.out" 2>&1; echo "exit $?" >> "$RUN/late-approve.out"
cat "$RUN/late-approve.out"
check "the late approval was refused" "'timed_out' in open('$RUN/late-approve.out').read() and 'exit 2' in open('$RUN/late-approve.out').read()" < /dev/null
check "still one marker after the late approval" "d == 1" < <(markers)

echo "== 10. budget: a proposal costing more than the 0.5 per-run cap is refused by the cap"
configure --level L2 --proposal marker --timeout 120 --cost 0.6
W6=$(new_run l2b-1); $T wait "$W6" | tee "$RUN/l2b.out"
check "run ends budget_exhausted" "d['status'] == 'SUCCESS' and 'budget_exhausted' in str(d['output'])" < "$RUN/l2b.out"
check "no escalation" "d is None" < <($T escalation "$W6")
"$V/python" -c "
import psycopg
with psycopg.connect('$DSN') as c:
    print({r[0]: float(r[1]) for r in c.execute(\"select scope, sum(usd_consumed) from okg.deployment_cost_ledger group by scope\")})" > "$RUN/ledger.txt"
cat "$RUN/ledger.txt"
check "the cap was enforced: nothing was charged to the run scope" "'run:$W6' not in open('$RUN/ledger.txt').read()" < /dev/null

echo "== 11. the per-day cap: a second run is refused once the day's budget is used"
configure --level L0 --proposal marker --cost 0.45
$T day-fill 9.3 | tee "$RUN/dayfill.json"
WD1=$(new_run cap-1); $T wait "$WD1" | tee "$RUN/cap1.out"
check "the first run, still under the 10 per day cap, finishes" "d['status'] == 'SUCCESS' and 'observed' in str(d['output'])" < "$RUN/cap1.out"
WD2=$(new_run cap-2); $T wait "$WD2" | tee "$RUN/cap2.out"
check "the second run is refused: budget_exhausted by the day cap" \
  "d['status'] == 'SUCCESS' and 'budget_exhausted' in str(d['output']) and 'day budget exhausted' in str(d['output'])" < "$RUN/cap2.out"
$T day-fill 1.0 > /dev/null   # put the day's spend back, so the later cases are not held by the cap

echo "== 12. L3 in archi.yaml (acc no-l3): refused at run start, nothing runs"
configure --level L3 --proposal marker --cost 0.1
W7=$(new_run l3-1); $T wait "$W7" | tee "$RUN/l3.out"
check "run ends refused, citing D13" "d['status'] == 'SUCCESS' and 'refused' in str(d['output']) and 'D13' in str(d['output'])" < "$RUN/l3.out"
$OPS status --json > "$RUN/status-l3.json" 2>"$RUN/status-l3.err" || cat "$RUN/status-l3.err"
check "status names the L3 problem" "'L3' in str(d['automations'][0]['level_problem'])" < "$RUN/status-l3.json"
check "still one marker" "d == 1" < <(markers)

echo "== 13. the mediator step's own guard (a test gate holds the run after the loop decided, before the step)"
configure --level L2 --proposal marker --timeout 300 --gate 300 --cost 0.1
BASE=$(markers)
echo "-- 13a. approve, then halt WITHOUT cancelling, then release: the step refuses (halted)"
POS=$(logpos); WG1=$(new_run gate-halt-1); ESCG1=$($T pending "$WG1" --timeout 120)
$OPS approve "$ESCG1" --as scratch-op
$T wait-log "$RUN/worker.log" "archi_ops test gate: waiting" --after "$POS" --timeout 90
$OPS halt --no-cancel --as scratch-op --json > "$RUN/halt-gate.json" 2> "$RUN/halt-gate.err"; cat "$RUN/halt-gate.json"
check "halt paused the scanner and left the approved run alone" "d['paused'] == ['archi_ops_scan'] and d['cancelled'] == [] and d['left_running'] == ['$WG1']" < "$RUN/halt-gate.json"
$T release-gate "$WG1"
$T wait "$WG1" --timeout 90 | tee "$RUN/gate-halt.out"
check "the run was not cancelled and ends halted" "d['status'] == 'SUCCESS' and 'halted' in str(d['output'])" < "$RUN/gate-halt.out"
in_log "$POS" "archi_ops_mediate refused (halted)" | check "the worker log shows the refusal coming from the mediator step" "d['found']"
check "no marker was written" "d == $BASE" < <(markers)
$OPS resume > /dev/null
echo "-- 13b. approve in time, make the escalation row late, release: the step refuses (timed_out)"
configure --timeout 40   # the deadline passes while the run waits at the gate
POS=$(logpos); WG2=$(new_run gate-late-1); ESCG2=$($T pending "$WG2" --timeout 120)
$OPS approve "$ESCG2" --as scratch-op
$T wait-log "$RUN/worker.log" "archi_ops test gate: waiting" --after "$POS" --timeout 90
echo "waiting past the 40 s deadline at the gate"; sleep 45
$T tamper late "$WG2"; $T release-gate "$WG2"
$T wait "$WG2" --timeout 90 | tee "$RUN/gate-late.out"
check "the run ends timed_out" "d['status'] == 'SUCCESS' and 'timed_out' in str(d['output'])" < "$RUN/gate-late.out"
in_log "$POS" "archi_ops_mediate refused (timed_out)" | check "the worker log shows the mediator step refusing the late approval" "d['found']"
check "no marker was written" "d == $BASE" < <(markers)
configure --timeout 300
echo "-- 13c. approve in time, supersede the decision (reject), release: the step refuses (error)"
POS=$(logpos); WG3=$(new_run gate-super-1); ESCG3=$($T pending "$WG3" --timeout 120)
$OPS approve "$ESCG3" --as scratch-op
$T wait-log "$RUN/worker.log" "archi_ops test gate: waiting" --after "$POS" --timeout 90
$T tamper superseded "$WG3"; $T release-gate "$WG3"
$T wait "$WG3" --timeout 90 | tee "$RUN/gate-super.out"
check "the run ends error" "d['status'] == 'SUCCESS' and \"'error'\" in str(d['output'])" < "$RUN/gate-super.out"
in_log "$POS" "archi_ops_mediate refused (error): decision is rejected" | check "the worker log shows the mediator step refusing a superseded decision" "d['found']"
check "no marker was written" "d == $BASE" < <(markers)

echo "== 14. at most once: the worker dies in the middle of the tool call; the replay does not call it again"
configure --level L2 --proposal slow --timeout 300 --gate 0 --cost 0.1
WS=$(new_run slow-1); ESCS=$($T pending "$WS" --timeout 120)
$OPS approve "$ESCS" --as scratch-op
for i in $(seq 1 60); do [ "$(markers)" = "$((BASE + 1))" ] && break; sleep 1; done
check "the tool wrote its marker and is still inside the call" "d == $((BASE + 1))" < <(markers)
kill_worker
start_worker 15
$T wait "$WS" --timeout 150 | tee "$RUN/slow.out"
check "the recovered run did not call the tool again: action unknown, outcome unknown" \
  "d['status'] == 'SUCCESS' and 'unknown' in str(d['output'])" < "$RUN/slow.out"
check "still exactly one marker for that run" "d == $((BASE + 1))" < <(markers)
$T mediation "$WS" > "$RUN/mediation-slow.json"; cat "$RUN/mediation-slow.json"
check "the attempt marker has no recorded end" "d['status'] == 'attempted'" < "$RUN/mediation-slow.json"
BASE=$(markers)

echo "== 15. worker-restart late approval: the guard refuses it"
configure --level L2 --proposal marker --timeout 40 --cost 0.1
W8=$(new_run l2w-1); ESC8=$($T pending "$W8" --timeout 90)
kill_worker
echo "waiting past the deadline (40 s) with the worker down"; sleep 50
$OPS approve "$ESC8" --as scratch-op > "$RUN/restart-approve.out" 2>&1; cat "$RUN/restart-approve.out"
check "the late approval was ACCEPTED by okg (worker down, no sweep)" "'approved' in open('$RUN/restart-approve.out').read()" < /dev/null
start_worker 15
$T wait "$W8" --timeout 120 | tee "$RUN/l2w.out"
check "the recovered run did not execute: timed_out" "d['status'] == 'SUCCESS' and 'timed_out' in str(d['output'])" < "$RUN/l2w.out"
check "no new marker (the guard refused)" "d == $BASE" < <(markers)

echo "== 16. halt during an approved run before its mediator (acc halt)"
configure --level L2 --proposal marker --timeout 300
W9=$(new_run l2h-1); ESC9=$($T pending "$W9" --timeout 90)
kill_worker
$OPS approve "$ESC9" --as scratch-op | tee "$RUN/halt-approve.out"
$OPS halt --as scratch-op --json > "$RUN/halt.json" 2> "$RUN/halt.err"; cat "$RUN/halt.json"; cat "$RUN/halt.err"
check "halt paused the scanner and cancelled the approved run" "d['paused'] == ['archi_ops_scan'] and d['cancelled'] == ['$W9'] and d['stayed_open'] == []" < "$RUN/halt.json"
start_worker 15
if ps -p "$(cat "$RUN/worker.pid")" > /dev/null 2>&1; then echo '{"up": true}'; else echo '{"up": false}'; fi \
  | check "the worker restarted after halt (no queued admission blocks it)" "d['up']"
sleep 15
$T wait "$W9" --timeout 60 | tee "$RUN/l2h.out"
check "the run is CANCELLED" "d['status'] == 'CANCELLED'" < "$RUN/l2h.out"
check "no new marker" "d == $BASE" < <(markers)
$T escalation "$W9" > "$RUN/esc-9.json"
check "the escalation is closed (decided, not pending)" "d['status'] in ('decided', 'applied')" < "$RUN/esc-9.json"
$OPS status --json > "$RUN/status-halted.json"
check "status shows the scanner paused" "d['scanner'] == {'archi_ops_scan': 'paused'}" < "$RUN/status-halted.json"

echo "== 17. resume: the halted signal runs again as attempt 2; then halt while a run waits for approval"
$OPS resume --json | tee "$RUN/resume.json"
check "resume un-paused the scanner and queued the halted signal for attempt 2" \
  "d['resumed'] == ['archi_ops_scan'] and d['requeued'] == [['scratch-triage', 'l2h-1', 2]]" < "$RUN/resume.json"
trigger
WR="scratch-triage~l2h-1~2"; ESCR=$($T pending "$WR" --timeout 90)
$OPS reject "$ESCR" --as scratch-op
$T wait "$WR" --timeout 90 | tee "$RUN/requeued.out"
check "the second attempt is a new run and ran to rejected" "d['status'] == 'SUCCESS' and 'rejected' in str(d['output'])" < "$RUN/requeued.out"
W10=$(new_run l2h2-1); ESC10=$($T pending "$W10" --timeout 90)
$OPS halt --as scratch-op --json > "$RUN/halt2.json" 2> "$RUN/halt2.err"; cat "$RUN/halt2.json"
check "halt closed the open escalation and cancelled the run" "d['closed'] == ['$ESC10'] and d['cancelled'] == ['$W10'] and d['stayed_open'] == []" < "$RUN/halt2.json"
$T wait "$W10" --timeout 60 | tee "$RUN/l2h2.out"
check "the waiting run is CANCELLED" "d['status'] == 'CANCELLED'" < "$RUN/l2h2.out"
$T escalation "$W10" > "$RUN/esc-10.json"
check "its escalation was answered reject by archi ops halt" "d['status'] in ('decided', 'applied') and d['decided_by'] == 'archi ops halt'" < "$RUN/esc-10.json"
check "no new marker" "d == $BASE" < <(markers)
$OPS resume --json > "$RUN/resume2.json" 2>/dev/null
trigger
WQ="scratch-triage~l2h2-1~2"; ESCQ=$($T pending "$WQ" --timeout 90)
$OPS reject "$ESCQ" --as scratch-op > /dev/null
$T wait "$WQ" --timeout 90 > /dev/null

echo "== 18. a stale source demotes a live automation at once, the demotion persists, a new promotion lifts it"
configure --level L2 --proposal marker --timeout 120 --cost 0.1
$T source stale
WS1=$(new_run dem-1); $T wait "$WS1" | tee "$RUN/dem1.out"
check "demoted run ends observed, in shadow" "d['status'] == 'SUCCESS' and 'observed' in str(d['output']) and 'shadow' in str(d['output'])" < "$RUN/dem1.out"
check "no escalation was opened" "d is None" < <($T escalation "$WS1")
$T demotions | tee "$RUN/demotions.json"
check "the demotion is in archi_ops.demotions the moment it happens (one row, from this run)" "len(d) == 1 and d[0]['run_id'] == '$WS1' and d[0]['reason'].startswith('demoted to shadow: ') and 'freshness:jira.jira' in d[0]['reason']" < "$RUN/demotions.json"
$T source fresh
WS2=$(new_run dem-2); $T wait "$WS2" | tee "$RUN/dem2.out"
check "the second run starts at once, the source is fresh, yet it stays in shadow (the log, not a publish, carries the demotion)" \
  "d['status'] == 'SUCCESS' and 'observed' in str(d['output']) and \"'mode': 'shadow'\" in str(d['output'])" < "$RUN/dem2.out"
check "no escalation, no new marker" "d is None" < <($T escalation "$WS2")
publish persisted
$T human-action "$WP"; $T human-action "$WS1"; $T human-action "$WS2"
publish human
$OPS promote scratch-triage --as operator-x > "$RUN/promote-refused.out" 2>&1; echo "exit $?" >> "$RUN/promote-refused.out"; cat "$RUN/promote-refused.out"
check "promote is refused with the default thresholds" \
  "'not ready to promote' in open('$RUN/promote-refused.out').read() and 'need 30' in open('$RUN/promote-refused.out').read() and 'exit 2' in open('$RUN/promote-refused.out').read()" < /dev/null
$OPS promote scratch-triage --as operator-x --min-proposals 4 --min-answered 4 --note "scratch: four shadow proposals matched a person" --json > "$RUN/promote2.json" 2> "$RUN/promote2.err"; cat "$RUN/promote2.json"; cat "$RUN/promote2.err"
CHANGE2=$("$V/python" -c "import json; print(json.load(open('$RUN/promote2.json'))['change'])")
check "a second promotion change is open, with four scored incidents" "d['state'] == 'awaiting_review' and d['thresholds']['min_proposals'] == 4 and 'scored=4' in d['report']" < "$RUN/promote2.json"
WS3=$(new_run dem-3); $T wait "$WS3" | tee "$RUN/dem3.out"
check "the new change is not approved yet: still shadow" "d['status'] == 'SUCCESS' and \"'mode': 'shadow'\" in str(d['output'])" < "$RUN/dem3.out"
$T approve-promotion "$CHANGE2"
WS4=$(new_run dem-4); ESCP=$($T pending "$WS4" --timeout 90)
check "after the approval the run is live: it waits for a person" "'$ESCP'.startswith('esc-')" < /dev/null
$OPS reject "$ESCP" --as scratch-op
$T wait "$WS4" --timeout 90 | tee "$RUN/dem4.out"
check "the promoted automation ran live, the demotion lifted" "d['status'] == 'SUCCESS' and 'rejected' in str(d['output']) and \"'mode': 'live'\" in str(d['output'])" < "$RUN/dem4.out"
$T pact promote-scratch-triage > "$RUN/pact2.json"; cat "$RUN/pact2.json"
check "both approvals are PACT decisions in the okg ledger, by operator-x" "len(d) == 2 and all(r['verdict'] == 'approve' and r['principal'] == 'operator-x' for r in d)" < "$RUN/pact2.json"

echo "-- 18b. hand-edit ops.yaml to name the older, approved promotion: it is superseded, so the run is shadow"
$T set-promotion "$CHANGE1"
WS5=$(new_run dem-5); $T wait "$WS5" | tee "$RUN/dem5.out"
check "an old approved promotion that is not the newest does not make the automation live" \
  "d['status'] == 'SUCCESS' and \"'mode': 'shadow'\" in str(d['output'])" < "$RUN/dem5.out"
$T set-promotion "$CHANGE2"

echo "== 19. publish and read the ops records back"
sleep 20   # let the queued halted-record admissions run
publish final
rb ops_decision "dec-$W0";  check "L0: decision observed" "rows[0]['attrs']['verdict'] == 'observed' and rows[0]['attrs']['actor_kind'] == 'automation' and rows[0]['attrs']['generation_read']" < "$RUN/rb-dec-$W0.jsonl"
rb ops_proposal "prop-$W1"; check "L1: the draft is on the proposal" "'looking at the SRM' in rows[0]['attrs']['draft'] and rows[0]['attrs']['mode'] == 'live'" < "$RUN/rb-prop-$W1.jsonl"
rb ops_decision "dec-$W1";  check "L1: decision drafted" "rows[0]['attrs']['verdict'] == 'drafted'" < "$RUN/rb-dec-$W1.jsonl"
rb ops_decision "dec-$W2";  check "L2 approve: decision approved, decided_by the OS user for scratch-op, escalation id recorded" "rows[0]['attrs']['verdict'] == 'approved' and rows[0]['attrs']['actor'].endswith(' for scratch-op') and rows[0]['attrs']['escalation_id'].startswith('esc-')" < "$RUN/rb-dec-$W2.jsonl"
rb ops_action "act-$W2";    check "L2 approve: action succeeded" "rows[0]['attrs']['status'] == 'succeeded' and rows[0]['attrs']['action_class'] == 'write_marker'" < "$RUN/rb-act-$W2.jsonl"
rb ops_outcome "out-$W2";   check "L2 approve: outcome resolved" "rows[0]['attrs']['result'] == 'resolved'" < "$RUN/rb-out-$W2.jsonl"
rb ops_decision "dec-$W3";  check "L2 reject: decision rejected" "rows[0]['attrs']['verdict'] == 'rejected'" < "$RUN/rb-dec-$W3.jsonl"
rb ops_decision "dec-$W4";  check "timeout: decision timed_out" "rows[0]['attrs']['verdict'] == 'timed_out'" < "$RUN/rb-dec-$W4.jsonl"
rb ops_decision "dec-$W6";  check "budget: decision budget_exhausted" "rows[0]['attrs']['verdict'] == 'budget_exhausted'" < "$RUN/rb-dec-$W6.jsonl"
rb ops_decision "dec-$WD2"; check "per-day cap: decision budget_exhausted naming the day budget" "rows[0]['attrs']['verdict'] == 'budget_exhausted' and 'day budget exhausted' in rows[0]['attrs']['reason']" < "$RUN/rb-dec-$WD2.jsonl"
rb ops_incident "inc-$W7";  check "L3: a run-level incident says why nothing ran" "'run not started' in rows[0]['attrs']['summary'] and 'D13' in rows[0]['attrs']['summary'] and not rows[0]['attrs'].get('generation_read')" < "$RUN/rb-inc-$W7.jsonl"
rb ops_decision "dec-$WG1"; check "gate halt: decision halted, superseding the approval" "rows[0]['attrs']['verdict'] == 'halted' and 'supersedes the approval' in rows[0]['attrs']['reason']" < "$RUN/rb-dec-$WG1.jsonl"
rb ops_decision "dec-$WG2"; check "gate late: decision timed_out from the step" "rows[0]['attrs']['verdict'] == 'timed_out' and 'after the deadline' in rows[0]['attrs']['reason']" < "$RUN/rb-dec-$WG2.jsonl"
rb ops_decision "dec-$WG3"; check "gate superseded: decision error from the step" "rows[0]['attrs']['verdict'] == 'error' and 'decision is rejected' in rows[0]['attrs']['reason']" < "$RUN/rb-dec-$WG3.jsonl"
rb ops_action "act-$WS";    check "at most once: the action is recorded as unknown" "rows[0]['attrs']['status'] == 'unknown' and 'possible duplicate prevented' in rows[0]['attrs']['result']" < "$RUN/rb-act-$WS.jsonl"
rb ops_outcome "out-$WS";   check "at most once: the outcome is unknown, with the reason" "rows[0]['attrs']['result'] == 'unknown' and 'check the external system' in rows[0]['attrs']['resolution']" < "$RUN/rb-out-$WS.jsonl"
rb ops_decision "dec-$W8";  check "restart: decision timed_out, the late approval was not executed" "rows[0]['attrs']['verdict'] == 'timed_out' and 'after the deadline' in rows[0]['attrs']['reason']" < "$RUN/rb-dec-$W8.jsonl"
rb ops_decision "dec-$W10-halt"; check "halt: a halted decision for the waiting run" "rows[0]['attrs']['verdict'] == 'halted'" < "$RUN/rb-dec-$W10-halt.jsonl"
rb ops_proposal "prop-$WH";  check "hand-edited live: proposal is shadow, reason says there is no promotion change" "rows[0]['attrs']['mode'] == 'shadow' and rows[0]['attrs']['mode_reason'].startswith('not promoted: ops.yaml names no promotion change')" < "$RUN/rb-prop-$WH.jsonl"
rb ops_proposal "prop-$WP";  check "named but unapproved: shadow, reason says the change is awaiting review" "rows[0]['attrs']['mode'] == 'shadow' and 'is awaiting_review' in rows[0]['attrs']['mode_reason']" < "$RUN/rb-prop-$WP.jsonl"
rb ops_proposal "prop-$WS1"; check "demotion: proposal is shadow with the failing check as the reason" "rows[0]['attrs']['mode'] == 'shadow' and rows[0]['attrs']['mode_reason'].startswith('demoted to shadow: ') and 'freshness:jira.jira' in rows[0]['attrs']['mode_reason']" < "$RUN/rb-prop-$WS1.jsonl"
rb ops_proposal "prop-$WS2"; check "persisted demotion: shadow, reason says the demotion persists" "rows[0]['attrs']['mode'] == 'shadow' and rows[0]['attrs']['mode_reason'].startswith('demoted to shadow: demotion persists')" < "$RUN/rb-prop-$WS2.jsonl"
rb ops_proposal "prop-$WS3"; check "second promotion unapproved: shadow, not promoted" "rows[0]['attrs']['mode'] == 'shadow' and rows[0]['attrs']['mode_reason'].startswith('not promoted: ')" < "$RUN/rb-prop-$WS3.jsonl"
rb ops_proposal "prop-$WS5"; check "hand-edited to the old promotion: shadow, reason says it is not the newest" "rows[0]['attrs']['mode'] == 'shadow' and 'not the newest promotion' in rows[0]['attrs']['mode_reason']" < "$RUN/rb-prop-$WS5.jsonl"
rb ops_proposal "prop-$WS4"; check "after the second approval the proposal is live" "rows[0]['attrs']['mode'] == 'live'" < "$RUN/rb-prop-$WS4.jsonl"

echo "== 20. cleanup: stop the worker this script started"
kill_worker
if [ "$FAILS" = 0 ]; then echo "== done: every check passed (commit $SHA)"; else echo "== done: $FAILS check(s) FAILED (commit $SHA)"; exit 1; fi
