#!/usr/bin/env bash
# PACT r3-auto-ticket-triage on a real scratch instance (okg 0.2.0 worker, cluster-manager with servicenow, jira and
# ticket-triage, fixture caches, a fake ticket system on a loopback port):
#   acc.r3-auto-ticket-triage.fixture  a fixture queue yields cited drafts (shadow, live L1) and gated status or
#                                      assignment proposals (L2): approved runs the live tool once, rejected, timed
#                                      out and halted never reach the ticket system; L3 is refused at run start
#
# Usage: ARCHI_SCRATCH_DSN=postgresql://... scratch_ticket_triage_run.sh <empty work dir>
#
# Run it from a clone of the commit under test: it prints that commit and refuses a dirty checkout. Needs an EMPTY
# database on an AKMON Postgres (ghcr.io/mitdbg/akmon-postgres), okg 0.2.0 and archi installed (editable, from this
# checkout) in the Python environment whose bin/ holds `python` (set V to that bin/). Starts one `okg runtime worker`
# and the fake ticket system in the background; their PIDs go to <work dir>/pids and they are the only processes
# this script kills. Every signal comes from the worker's own scanner reading the published graph: a ticket
# arrives in a source cache, `okg run` publishes it, and the scanner sees it as new. No model call, no network.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
V="${V:-$(dirname "$(command -v python)")}"
RUN="$1"
DSN="${ARCHI_SCRATCH_DSN:?set ARCHI_SCRATCH_DSN to an empty AKMON database}"
PORT="${TT_PORT:-18931}"
mkdir -p "$RUN"
[ -z "$(ls -A "$RUN")" ] || { echo "$RUN is not empty"; exit 1; }
RUN="$(cd "$RUN" && pwd)"
cd "$RUN"
unset OKG_PROFILES_DIR
D="$(basename "$DSN")"
INST="$RUN/instance"
export ARCHI_INSTANCE="$INST" OKG_DSN="$DSN" ARCHI_OPS_DEPLOYMENT="$D" OKG_DEPLOYMENTS_DIR="$INST/deployments"
export PYTHONDONTWRITEBYTECODE=1 TT_STATE="$RUN/tickets-state.json"
export OKG_ROLE_SECRETS_FILE="$INST/.secrets/role-secrets.json"
export TT_REDMINE_TOKEN=fixture-token   # what the fake ticket system accepts; the worker inherits it
T="$V/python $REPO/python/tests/ops/scratch_ticket_triage_tools.py"
OPS="$V/archi ops --instance $INST --dsn $DSN --deployment $D"
PIDS="$RUN/pids"
FAILS=0
# the worker and the fake ticket system are the only processes this script starts, and the only ones it stops
trap '[ -f "$PIDS" ] && kill -9 $(cat "$PIDS") 2>/dev/null' EXIT

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
" || { FAILS=$((FAILS + 1)); echo "$1" >> "$RUN/fails"; }
}
start_worker() {
  ( cd "$INST" && exec nohup "$V/okg" runtime worker --deployment $D >> "$RUN/worker.log" 2>&1 ) &
  echo $! > "$RUN/worker.pid"; cat "$RUN/worker.pid" >> "$PIDS"
  echo "worker started, pid $(cat "$RUN/worker.pid")"
  sleep "${1:-15}"
}
publish() { "$V/okg" run --once --deployment $D --apply --json > "$RUN/publish-$1.json" 2>&1 || tail -3 "$RUN/publish-$1.json"; }
ingest() {  # read the caches again (a publish alone finds nothing new), then publish
  "$V/okg" ingest --deployment $D --inline --json > "$RUN/ingest-$1.json" 2>&1 || tail -3 "$RUN/ingest-$1.json"
  publish "$1"
}
node() { "$V/okg" filter --deployment $D --type "$1" --where "attrs.record_id=$2" --all --jsonl 2>/dev/null; }
writes() { $T ticket "$1" "$2" | "$V/python" -c "import json,sys; t=json.load(sys.stdin); print(len(t['notes']))"; }
arrive() {  # arrive <system> <key> <subject>: the ticket arrives, is published, and the scanner adds its signal; prints the run id
  local sid; sid=$($T arrive "$@"); ingest "arrive-$2" >&2; $T wait-signal "$sid" --timeout 240
}
configure() { $T configure --port "$PORT" "$@" > /dev/null; }

echo "== 0. the commit under test"
SHA="$(git -C "$REPO" rev-parse HEAD)"
echo "commit $SHA"
echo "{\"dirty\": $(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')}" | check "the checkout is clean" "d['dirty'] == 0"
[ "$FAILS" = 0 ] || exit 1

echo "== 1. inputs: the shipped packs and cluster-manager, the fixture queue, the fake ticket system"
$T inputs
( exec nohup "$V/python" "$REPO/python/tests/ops/ticket_fakes.py" serve "$TT_STATE" "$PORT" >> "$RUN/tickets.log" 2>&1 ) &
echo $! > "$RUN/tickets.pid"; cat "$RUN/tickets.pid" >> "$PIDS"

echo "== 2. archi init cluster-manager --with servicenow,jira,ticket-triage (the container start and docsite's download are skipped)"
$T init-instance 2>&1 | tee "$RUN/install-flow.log" | grep -v "^\s*$" | tail -12
[ -f "$INST/automations/ticket-triage.yaml" ] || { echo "archi init did not finish; see install-flow.log"; exit 1; }
"$V/python" -c "
import json, yaml
b = yaml.safe_load(open('$INST/automations/ticket-triage.yaml'))
print(json.dumps({'trigger': b['trigger'], 'sources': b['context_gate']['sources'], 'not_selected': b.get('not_selected')}))" \
  | tee "$RUN/block.json" | check "the rendered automation triggers on all three systems and gates on their sources" \
  "d['trigger']['system'] == ['redmine', 'jira', 'servicenow'] and d['sources'] == ['tickets.tickets', 'jira.jira', 'servicenow.servicenow'] and d['not_selected'] is None"
configure --level L1 --mode shadow --timeout 120
cat "$INST/ops.yaml"

echo "== 3. the worker starts; the first scan of the published queue sets the cursors and triages nothing"
start_worker 15
publish first
"$V/okg" runtime trigger-schedule --deployment $D archi_ops_scan > /dev/null 2>&1
for i in $(seq 1 24); do [ "$($T cursors | "$V/python" -c 'import json,sys; print(sum(k.endswith("/*") for k in json.load(sys.stdin)))')" = 3 ] && break; sleep 5; done
$T cursors | tee "$RUN/cursors-0.json" | check "each system read, and a cursor per source (subtype at a site), from the published generation" \
  "sorted(k for k in d if k.endswith('/*')) == ['jira/*', 'redmine/*', 'servicenow/*'] and any(':tracker_ticket@' in k for k in d) and all(v['generation'] and v['updated'] for k, v in d.items() if not k.endswith('/*'))"
$T signals | check "no signal for the tickets already in the queue" "d == []"

echo "== 4. SHADOW (the default): three tickets arrive (Redmine, JIRA from the jira pack, JIRA from the tickets pack); each becomes one signal and a run that records a cited draft"
WS=$(arrive redmine 4820 "Scratch filesystem full on node051")
WJ=$(arrive jira OPS-15 "Transfers failing with SRM_TIMEOUT after the upgrade" --description "FTS reports SRM_TIMEOUT on srm-put.")
WG=$(arrive redmine SUP-3 "Jobs held: home quota exceeded" --tracker jira --description "Jobs stay held, home quota exceeded.")
# a ticket_queue automation runs one at a time per ticket, so the three runs may overlap
$T wait "$WS" --timeout 400 | tee "$RUN/shadow-redmine.out" | check "the Redmine run ends observed in shadow" \
  "d['status'] == 'SUCCESS' and d['output']['status'] == 'observed' and d['output']['mode'] == 'shadow'"
$T wait "$WJ" --timeout 400 | tee "$RUN/shadow-jira.out" | check "the JIRA run ends observed in shadow" \
  "d['status'] == 'SUCCESS' and d['output']['status'] == 'observed' and d['output']['mode'] == 'shadow'"
$T wait "$WG" --timeout 400 | tee "$RUN/shadow-generic-jira.out" | check "the generic JIRA run (tickets pack) ends observed in shadow" \
  "d['status'] == 'SUCCESS' and d['output']['status'] == 'observed' and d['output']['mode'] == 'shadow'"
$T signals | tee "$RUN/signals-1.json" | check "exactly the three arrivals became signals, with their ticket entities" \
  "sorted(s['entity'] for s in d) == ['jira:OPS-15', 'jira:SUP-3', 'redmine:4820'] and all(s['source'] == 'ticket' for s in d)"
echo "{\"n\": $(writes redmine 4820)}" | check "shadow sent nothing to the ticket system" "d['n'] == 0"
publish shadow
node ops_proposal "prop-$WS" > "$RUN/prop-shadow.jsonl"
check "the shadow proposal is a draft that cites the ticket, related tickets and the precedent-free record" \
  "rows[0]['attrs']['mode'] == 'shadow' and rows[0]['attrs']['action_class'] == 'draft_reply' and 'tracker_ticket:redmine:4820' in rows[0]['attrs']['evidence_ids'] and 'tracker_ticket:redmine:4799' in rows[0]['attrs']['evidence_ids'] and 'Sources:' in rows[0]['attrs']['draft']" < "$RUN/prop-shadow.jsonl"
node ops_incident "inc-$WS" > "$RUN/inc-shadow.jsonl"
check "the incident names the ticket (entity and ticket key)" \
  "rows[0]['attrs']['entity'] == 'redmine:4820' and rows[0]['attrs']['ticket_keys'] in (['4820'], '4820')" < "$RUN/inc-shadow.jsonl"
node ops_proposal "prop-$WJ" > "$RUN/prop-shadow-jira.jsonl"
check "the JIRA draft cites the related resolved issues" \
  "'jira:OPS-11' in rows[0]['attrs']['evidence_ids'] and 'jira:OPS-15' in rows[0]['attrs']['evidence_ids']" < "$RUN/prop-shadow-jira.jsonl"

node ops_proposal "prop-$WG" > "$RUN/prop-shadow-generic.jsonl"
check "the generic JIRA draft cites its own tracker_ticket and the related generic JIRA tickets, no Redmine ticket" \
  "'tracker_ticket:jira:SUP-3' in rows[0]['attrs']['evidence_ids'] and 'tracker_ticket:jira:SUP-1' in rows[0]['attrs']['evidence_ids'] and not any(e.startswith('tracker_ticket:redmine:') for e in rows[0]['attrs']['evidence_ids'])" < "$RUN/prop-shadow-generic.jsonl"

echo "== 5. promotion, as an operator does it (a PACT change a person approves)"
$T human-action "$WS"; $T human-action "$WJ"; publish human
$OPS promote ticket-triage --as operator-x --min-proposals 1 --min-answered 1 \
  --note "lab: the shadow drafts matched a person" --json > "$RUN/promote.json" 2> "$RUN/promote.err"
cat "$RUN/promote.json"; cat "$RUN/promote.err"
CHANGE=$("$V/python" -c "import json; print(json.load(open('$RUN/promote.json'))['change'])")
$T approve-promotion "$CHANGE"
configure --mode live

echo "== 6. live L1: a cited draft, nothing sent"
W1=$(arrive redmine 4821 "Scratch filesystem full on node063")
$T wait "$W1" | tee "$RUN/l1.out" | check "the live L1 run ends drafted" \
  "d['status'] == 'SUCCESS' and d['output']['status'] == 'drafted' and d['output']['mode'] == 'live' and not d['output']['mediated']"
echo "{\"n\": $(writes redmine 4821)}" | check "L1 sent nothing to the ticket system" "d['n'] == 0"

echo "== 7. L2 approved: an assignment proposal waits for a person; the approval runs the live tool once"
configure --level L2 --timeout 600
W2=$(arrive redmine 4822 "Scratch filesystem full on node070")
ESC2=$($T pending "$W2" --timeout 360); echo "escalation $ESC2"
$T escalation "$W2" | tee "$RUN/esc-2.json" | check "the escalation asks to assign the ticket to Grace Hopper" \
  "d['escalation_id'] == '$ESC2' and 'Action: assign_ticket' in d['reason'] and 'Grace Hopper' in d['reason']"
echo "{\"n\": $(writes redmine 4822)}" | check "nothing reached the ticket before the approval" "d['n'] == 0"
$OPS approve "$ESC2" --as operator-x --note "lab approve"
$T wait "$W2" | tee "$RUN/l2a.out" | check "the approved run executed" \
  "d['status'] == 'SUCCESS' and d['output']['status'] == 'executed' and d['output']['mediated']"
$T ticket redmine 4822 | tee "$RUN/ticket-4822.json" | check "the ticket is assigned, with one note naming the escalation and the approver" \
  "d['assignee'] == 'Grace Hopper' and len(d['notes']) == 1 and '$ESC2' in d['notes'][0] and 'operator-x' in d['notes'][0]"

echo "== 8. L2 rejected: the ticket system is not touched"
W3=$(arrive redmine 4823 "Scratch filesystem full on node071")
ESC3=$($T pending "$W3" --timeout 360)
$OPS reject "$ESC3" --as operator-x --note "not now"
$T wait "$W3" | check "the rejected run ends rejected" "d['status'] == 'SUCCESS' and d['output']['status'] == 'rejected'"
$T ticket redmine 4823 | check "ticket 4823 unchanged" "d['assignee'] == '' and d['notes'] == []"

echo "== 9. L2 timed out: nobody answers; a late approval is refused; the ticket is not touched"
configure --timeout 120
W4=$(arrive redmine 4824 "Scratch filesystem full on node072")
ESC4=$($T pending "$W4" --timeout 360)
$T wait "$W4" --timeout 400 | check "the run ends timed_out" "d['status'] == 'SUCCESS' and d['output']['status'] == 'timed_out'"
$OPS approve "$ESC4" --as operator-x > "$RUN/late.out" 2>&1; echo "exit $?" >> "$RUN/late.out"; cat "$RUN/late.out"
check "the late approval was refused" "'exit 2' in open('$RUN/late.out').read()" < /dev/null
$T ticket redmine 4824 | check "ticket 4824 unchanged" "d['assignee'] == '' and d['notes'] == []"

echo "== 10. halt while a proposal waits: the run is cancelled and its escalation closed; the ticket is not touched"
configure --timeout 300
W5=$(arrive redmine 4825 "Scratch filesystem full on node073")
ESC5=$($T pending "$W5" --timeout 360)
$OPS halt --as operator-x --json > "$RUN/halt.json" 2> "$RUN/halt.err"; cat "$RUN/halt.json"
check "halt paused the scanner, closed the escalation and cancelled the run" \
  "d['paused'] == ['archi_ops_scan'] and d['closed'] == ['$ESC5'] and d['cancelled'] == ['$W5']" < "$RUN/halt.json"
$T wait "$W5" --timeout 60 | check "the run is CANCELLED" "d['status'] == 'CANCELLED'"
$T ticket redmine 4825 | check "ticket 4825 unchanged" "d['assignee'] == '' and d['notes'] == []"
$OPS resume --json > "$RUN/resume.json"; cat "$RUN/resume.json"
W5B="${W5%~1}~2"   # resume requeues the cancelled signal as attempt 2; a person rejects it so the queue moves on
"$V/okg" runtime trigger-schedule --deployment $D archi_ops_scan > /dev/null 2>&1
ESC5B=$($T pending "$W5B" --timeout 360) && $OPS reject "$ESC5B" --as operator-x > /dev/null
$T wait "$W5B" --timeout 120 > /dev/null

echo "== 11. L3 in archi.yaml: refused at run start, nothing runs"
configure --level L3
W6=$(arrive redmine 4826 "Scratch filesystem full on node074")
$T wait "$W6" | tee "$RUN/l3.out" | check "the run ends refused, citing D13" \
  "d['status'] == 'SUCCESS' and d['output']['status'] == 'refused' and 'D13' in d['output']['reason']"
$T ticket redmine 4826 | check "ticket 4826 unchanged" "d['assignee'] == '' and d['notes'] == []"
configure --level L1

echo "== 12. the records, published and read back"
sleep 15; publish final
node ops_proposal "prop-$W2" > "$RUN/rb-prop-2.jsonl"
check "L2: the proposal is assign_ticket to Grace Hopper, live, citing the related tickets" \
  "rows[0]['attrs']['action_class'] == 'assign_ticket' and json.loads(rows[0]['attrs']['payload'])['value'] == 'Grace Hopper' and rows[0]['attrs']['mode'] == 'live' and 'tracker_ticket:redmine:4799' in rows[0]['attrs']['evidence_ids']" < "$RUN/rb-prop-2.jsonl"
node ops_decision "dec-$W2" > "$RUN/rb-dec-2.jsonl"
check "L2: decision approved, by a person" "rows[0]['attrs']['verdict'] == 'approved' and rows[0]['attrs']['actor_kind'] == 'person'" < "$RUN/rb-dec-2.jsonl"
node ops_action "act-$W2" > "$RUN/rb-act-2.jsonl"
check "L2: the action succeeded through the live tool" \
  "rows[0]['attrs']['status'] == 'succeeded' and rows[0]['attrs']['mediator'] == 'archi.tools.tickets:assign_ticket'" < "$RUN/rb-act-2.jsonl"
node ops_proposal "prop-$W1" > "$RUN/rb-prop-1.jsonl"
check "L1: the live draft is on the proposal" "rows[0]['attrs']['mode'] == 'live' and 'Suggested reply:' in rows[0]['attrs']['draft']" < "$RUN/rb-prop-1.jsonl"
node ops_decision "dec-$W3" > "$RUN/rb-dec-3.jsonl"; check "reject recorded" "rows[0]['attrs']['verdict'] == 'rejected'" < "$RUN/rb-dec-3.jsonl"
node ops_decision "dec-$W4" > "$RUN/rb-dec-4.jsonl"; check "timeout recorded" "rows[0]['attrs']['verdict'] == 'timed_out'" < "$RUN/rb-dec-4.jsonl"
"$V/python" -c "
import json
s = json.load(open('$TT_STATE'))
issues = s['redmine']['issues']
print(json.dumps({k: len(v['notes']) for k, v in issues.items()}))" | tee "$RUN/notes.json" \
  | check "across the run, only the approved ticket was ever written" "d == {'4820': 0, '4821': 0, '4822': 1, '4823': 0, '4824': 0, '4825': 0, '4826': 0}"

echo "== 13. no traceback through archi code in the worker log"
"$V/python" -c "
import re
text = open('$RUN/worker.log', errors='replace').read()
blocks = re.split(r'(?=Traceback \(most recent call last\))', text)[1:]
mine = [b for b in blocks if '/python/archi/' in b.split('\n\n')[0]]
print('{\"all\": %d, \"through_archi\": %d}' % (len(blocks), len(mine)))" | tee "$RUN/tracebacks.json" | check "no traceback through archi code" "d['through_archi'] == 0"

echo "== 14. cleanup: the exit trap stops the worker and the fake ticket system this script started"
FAILS=$( [ -f "$RUN/fails" ] && wc -l < "$RUN/fails" | tr -d ' ' || echo 0 )
echo "== result: $FAILS failed check(s) (commit $SHA)"
[ "$FAILS" = 0 ]
