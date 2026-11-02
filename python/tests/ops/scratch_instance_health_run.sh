#!/usr/bin/env bash
# PACT r3-ops-self-monitoring on a real scratch instance (okg 0.2.0 worker, the instance-health pack, local tracker):
#   acc.r3-ops-self-monitoring.issue  a stale fixture source yields exactly one issue across two runs; when the
#                                     source is refreshed the next run marks the issue resolved
#
# Usage: ARCHI_SCRATCH_DSN=postgresql://... scratch_instance_health_run.sh <empty work dir>
#
# Run it from a clone of the commit under test: it prints that commit and refuses a dirty checkout. Needs an EMPTY
# database on an AKMON Postgres (ghcr.io/mitdbg/akmon-postgres), okg 0.2.0 and archi installed (editable, from this
# checkout) in the Python environment whose bin/ holds `python` (set V to that bin/). Starts one `okg runtime worker`
# in the background; its PID goes to <work dir>/pids and it is the only process this script kills.
# Every run is started by the worker's own scanner from the pack's schedule trigger (archi.ops.schedule): nothing
# here adds a signal. To get several ticks in minutes, the lab's COPY of the pack (not the shipped one) fires every
# minute (cron "* * * * *", runs_per_hour 60); the shipped hourly cron is covered by the unit tests. So runs keep
# coming once the worker is up, and the script counts them instead of choosing them. No model call and no network:
# the proposer is algorithmic and the tracker is local.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
V="${V:-$(dirname "$(command -v python)")}"
RUN="$1"
DSN="${ARCHI_SCRATCH_DSN:?set ARCHI_SCRATCH_DSN to an empty AKMON database}"
mkdir -p "$RUN"
[ -z "$(ls -A "$RUN")" ] || { echo "$RUN is not empty"; exit 1; }
RUN="$(cd "$RUN" && pwd)"
cd "$RUN"
unset OKG_PROFILES_DIR
D="$(basename "$DSN")"
INST="$RUN/instance"
export ARCHI_INSTANCE="$INST" OKG_DSN="$DSN" ARCHI_OPS_DEPLOYMENT="$D" OKG_DEPLOYMENTS_DIR="$INST/deployments"
export PYTHONDONTWRITEBYTECODE=1
# where `archi init` put the role passwords; okg (worker and CLI) reads them from here, as archi.stack.instance.host_env sets it
export OKG_ROLE_SECRETS_FILE="$INST/.secrets/role-secrets.json"
T="$V/python $REPO/python/tests/ops/scratch_instance_health_tools.py"
PIDS="$RUN/pids"
FAILS=0

check() {  # check <label> <python expression over stdin as `d`>
  "$V/python" -c "
import json, sys
text = sys.stdin.read()
try:
    d = json.loads(text)
except ValueError:
    d = None
ok = $2
print(('PASS' if ok else 'FAIL') + ': $1')
sys.exit(0 if ok else 1)
" || { FAILS=$((FAILS + 1)); echo "$1" >> "$RUN/fails"; }   # a check in a pipeline runs in a subshell: the file counts it
}
start_worker() {
  ( cd "$INST" && exec nohup "$V/okg" runtime worker --deployment $D >> "$RUN/worker.log" 2>&1 ) &
  echo $! > "$RUN/worker.pid"; cat "$RUN/worker.pid" >> "$PIDS"
  echo "worker started, pid $(cat "$RUN/worker.pid")"
  sleep "${1:-15}"
}
trigger() { "$V/okg" runtime trigger-schedule --deployment $D archi_ops_scan > "$RUN/trigger.out" 2>&1 || tail -3 "$RUN/trigger.out" >&2; }
publish() { "$V/okg" run --once --deployment $D --apply --json > "$RUN/publish-$1.json" 2>&1 || tail -3 "$RUN/publish-$1.json"; }
runs_where() {  # runs_where <python predicate over a run dict r>: how many runs match
  $T runs | "$V/python" -c "
import json, sys
runs = json.load(sys.stdin)
print(sum(1 for r in runs if $1))"
}
wait_runs() {  # wait_runs <min count> <predicate> <timeout s>: wait until that many runs match
  local deadline=$((SECONDS + $3)) n=0
  while [ $SECONDS -lt $deadline ]; do
    n=$(runs_where "$2"); [ "$n" -ge "$1" ] && { echo "$n runs match"; return 0; }
    sleep 5
  done
  echo "only $n runs match after $3 s"; return 1
}
issues() { $T issues > "$RUN/issues-$1.json"; cat "$RUN/issues-$1.json"; }

echo "== 0. the commit under test"
SHA="$(git -C "$REPO" rev-parse HEAD)"
echo "commit $SHA"
echo "{\"dirty\": $(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')}" | check "the checkout is clean" "d['dirty'] == 0"
[ "$FAILS" = 0 ] || exit 1

echo "== 1. inputs: the shipped packs and cern-team, and a jira cache for the source that will go stale"
"$V/python" -c "
import sys; sys.path.insert(0, '$REPO/python/tests')
from pathlib import Path
from bundles.bundle_support import build_cern_team_catalog
build_cern_team_catalog(Path('$RUN/catalog'))
"
sed -i 's/cron: "7 \* \* \* \*"/cron: "* * * * *"/; s/runs_per_hour: 2/runs_per_hour: 60/' "$RUN/catalog/packs/instance-health/pack.yaml"
grep -n "cron:\|runs_per_hour" "$RUN/catalog/packs/instance-health/pack.yaml"
mkdir -p data/jira
cp "$REPO/python/tests/packs/fixtures/ops_record/jira_records_open.json" data/jira/records.json
echo '{"record_count": 2}' > data/jira/meta.json

echo "== 2. archi init (archi.cli.init.init itself; only the container start is skipped, the lab database runs)"
$T init-instance 2>&1 | tee "$RUN/install-flow.log" | grep -v "^\s*$" | tail -15
check "init rendered the automation and the runtime line" \
  "'archi.ops.workflow' in open('$INST/profiles/cern-team/deployment-defaults.yaml').read() and 'instance-health' in open('$INST/archi.lock').read()" < /dev/null
test -f "$INST/automations/instance-health.yaml" && sed -n 1,12p "$INST/automations/instance-health.yaml"
echo "{\"pack\": \"$(grep -c 'type: schedule' "$INST/automations/instance-health.yaml")\"}" | check "the rendered trigger is a schedule" "d['pack'] == '1'"

echo "== 3. the stack facts a lab without containers cannot have are left out of the monitor; everything else is read"
$V/archi doctor --instance "$INST" --json > "$RUN/doctor-0.json" 2>"$RUN/doctor-0.err"; echo "doctor exit $?"
"$V/python" -c "
import json
r = json.load(open('$RUN/doctor-0.json'))
for c in r['checks']:
    print(c['status'].upper().ljust(5), c['id'], '-', c['message'][:110])
"
IGNORE="${IGNORE:-}"
$T configure --level L1 --mode shadow ${IGNORE:+--ignore "$IGNORE"}
cat "$INST/ops.yaml"

echo "== 4. start the worker, ingest the jira cache and publish (a published graph and a watermark for jira.jira)"
start_worker 15
publish first
"$V/okg" doctor --check sources --deployment $D --json > "$RUN/doctor-sources.json" 2>&1 || true
check "okg lists the jira source with a watermark" \
  "any(r.get('name') == '$D.jira' or r.get('name') == 'jira.jira' for r in d['sources']) and any(((r.get('coverage') or {}).get('watermark') or {}).get('last_reconcile_at') for r in d['sources'])" < "$RUN/doctor-sources.json"
$V/archi doctor --instance "$INST" --json > "$RUN/doctor-1.json" 2>/dev/null || true
check "archi doctor reports the sources as fresh before anything is made stale" \
  "[c for c in d['checks'] if c['id'] == 'source-freshness'][0]['status'] == 'pass'" < "$RUN/doctor-1.json"

echo "== 5. SHADOW: the scanner makes the first tick by itself; the run records a proposal and writes no issue"
trigger   # the schedule would fire within a minute; this only saves the wait
wait_runs 1 "r['mode'] == 'shadow' and r['state'] == 'SUCCESS'" 240
$T runs > "$RUN/runs-shadow.json"
check "the first run finished observed in shadow" "d[0]['status'] == 'observed' and d[0]['mode'] == 'shadow'" < "$RUN/runs-shadow.json"
W0=$("$V/python" -c "import json; print(json.load(open('$RUN/runs-shadow.json'))[0]['id'])")
$T signals > "$RUN/signals-0.json"
check "every signal was made by the scanner from the schedule trigger" \
  "all(s['source'] == 'schedule' and s['signal_id'].startswith('tick-') for s in d) and len(d) >= 1" < "$RUN/signals-0.json"
issues shadow | check "shadow wrote no issue file" "d['files'] == []"

echo "== 6. promote it as an operator would (a PACT change a person approves); the source is already stale when it goes live"
publish shadow
$T human-action "$W0"; publish human0
$V/archi ops --instance "$INST" --dsn "$DSN" --deployment $D promote instance-health --as operator-x --min-proposals 1 \
  --min-answered 1 --note "lab: the shadow proposal matched a person" --json > "$RUN/promote.json" 2> "$RUN/promote.err"
cat "$RUN/promote.json"; cat "$RUN/promote.err"
CHANGE=$("$V/python" -c "import json; print(json.load(open('$RUN/promote.json'))['change'])")
check "promote opened a PACT change awaiting review" "d['state'] == 'awaiting_review'" < "$RUN/promote.json"
echo "-- a source goes stale: the jira watermark is aged 30 days"
$T source stale
$V/archi doctor --instance "$INST" --json > "$RUN/doctor-2.json" 2>/dev/null || true
"$V/python" -c "
import json
r = json.load(open('$RUN/doctor-2.json'))
f = [c for c in r['checks'] if c['id'] == 'source-freshness'][0]
print(f['status'], f['message'])"
$T approve-promotion "$CHANGE"
$T pact "$CHANGE" | check "the operator approval is in the ledger" "len(d) == 1 and d[0]['verdict'] == 'approve'"

echo "== 7. two live runs see the stale source (acc.issue); more may come, each one minute apart"
wait_runs 2 "r['mode'] == 'live' and r['status'] == 'drafted' and r['mediated']" 600
issues a
check "after the live runs: exactly one issue for the stale source, open" \
  "[f['key'] for f in d['files'] if f['key'].startswith('source-freshness/')] == ['source-freshness/jira.jira'] and [f['state'] for f in d['files'] if f['key'] == 'source-freshness/jira.jira'] == ['open'] and d['dirty'] == False" < "$RUN/issues-a.json"
sleep 70   # at least one more run
issues b
"$V/python" -c "
import json, subprocess
a = json.load(open('$RUN/issues-a.json')); b = json.load(open('$RUN/issues-b.json'))
mine = [c for c in b['commits'] if 'source-freshness/jira.jira' in c]
print(json.dumps({'one_file': len([f for f in b['files'] if f['key'].startswith('source-freshness/')]) == 1,
                  'one_commit_for_it': len(mine) == 1,
                  'same_file': [f for f in a['files'] if f['key'] == 'source-freshness/jira.jira'] == [f for f in b['files'] if f['key'] == 'source-freshness/jira.jira']}))" \
  | check "later runs changed nothing for it: one file, one commit that opened it, same content" "d == {'one_file': True, 'one_commit_for_it': True, 'same_file': True}"
N=$(runs_where "r['mode'] == 'live' and r['status'] == 'drafted' and r['mediated']"); echo "live drafted runs so far: $N"
check "at least two live runs saw the stale source" "d >= 2" <<< "$N"

echo "== 8. the source is refreshed (a re-ingest; if okg leaves the watermark alone for an unchanged cache, the watermark is set directly, as scratch_ops_runtime_run.sh does); a later run resolves the issue"
publish refresh
$V/archi doctor --instance "$INST" --json > "$RUN/doctor-3.json" 2>/dev/null || true
if "$V/python" -c "
import json, sys
r = json.load(open('$RUN/doctor-3.json'))
sys.exit(0 if [c for c in r['checks'] if c['id'] == 'source-freshness'][0]['status'] == 'pass' else 1)"; then
  echo "the re-ingest moved the watermark"
else
  echo "the re-ingest left the watermark alone; setting it directly"; $T source fresh
fi
wait_runs $((N + 2)) "r['mode'] == 'live' and r['status'] == 'drafted' and r['mediated']" 400
issues c
check "the issue is marked resolved, not deleted, and there is still one file for it" \
  "[(f['key'], f['state']) for f in d['files'] if f['key'].startswith('source-freshness/')] == [('source-freshness/jira.jira', 'resolved')] and 'resolved' in [f for f in d['files'] if f['key'] == 'source-freshness/jira.jira'][0]['events']" < "$RUN/issues-c.json"
cat "$INST/.archi/issues/source-freshness--jira.jira.md"
$T runs > "$RUN/runs-final.json"
check "no run ended in an error" "all(r['status'] != 'error' for r in d)" < "$RUN/runs-final.json"

echo "== 9. the records the runs left"
publish records
for kind in ops_proposal ops_decision ops_action; do
  "$V/okg" filter --deployment $D --type $kind --all --jsonl 2>/dev/null | "$V/python" -c "
import json, sys
rows = [json.loads(l) for l in sys.stdin if l.strip()]
print('$kind', len(rows), sorted({(r['attrs'].get('verdict') or r['attrs'].get('status') or r['attrs'].get('mode')) for r in rows}))"
done

echo "== 10. the worker log has no traceback from the automation"
# okg's own publisher cycle may refuse to run while the instance repo has uncommitted PACT files (the promotion
# window in step 6); that is okg waiting for the operator's commit, not this automation. Count only tracebacks that
# pass through archi's code.
"$V/python" -c "
import re
text = open('$RUN/worker.log', errors='replace').read()
blocks = re.split(r'(?=Traceback \(most recent call last\))', text)[1:]
mine = [b for b in blocks if '/python/archi/' in b.split('\n\n')[0]]
print('{\"all\": %d, \"through_archi\": %d}' % (len(blocks), len(mine)))" | tee "$RUN/tracebacks.json" | check "no traceback through archi code in the worker log" "d['through_archi'] == 0"
git -C "$INST" log --oneline | head -12

FAILS=$( [ -f "$RUN/fails" ] && wc -l < "$RUN/fails" | tr -d ' ' || echo 0 )
echo "== result: $FAILS failed check(s)"
[ "$FAILS" = 0 ]
