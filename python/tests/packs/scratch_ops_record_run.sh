#!/usr/bin/env bash
# PACT r3-ops-record on a real scratch instance:
#   acc.r3-ops-record.load   the catalog loads (--apply) with the ops-record slice
#   acc.r3-ops-record.admit  a workflow run's records are admitted and searchable
#                            in the next generation; a ticket resolution from the
#                            jira connector links to its incident
#   acc.r3-ops-record.precedent (live counterpart of the pytest)  the new
#                            incident retrieves its precedent with the outcome,
#                            and the `precedents` search profile ranks ops records
#
# Usage: ARCHI_SCRATCH_DSN=postgresql://... scratch_ops_record_run.sh <empty work dir>
#
# Needs an EMPTY database on an AKMON Postgres (ghcr.io/mitdbg/akmon-postgres),
# network access for the CMSSW release map (cern-team's default pack), and
# okg + archi installed in the Python environment whose bin/ holds `python`
# (set V to that bin/ directory; default: the one on PATH).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
V="${V:-$(dirname "$(command -v python)")}"
RUN="$1"
DSN="${ARCHI_SCRATCH_DSN:?set ARCHI_SCRATCH_DSN to an empty AKMON database}"
F="$REPO/python/tests/packs/fixtures/ops_record"
mkdir -p "$RUN"
[ -z "$(ls -A "$RUN")" ] || { echo "$RUN is not empty"; exit 1; }
RUN="$(cd "$RUN" && pwd)"
cd "$RUN"
unset OKG_PROFILES_DIR
D=r3ops

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
"
}

echo "== 0. inputs: cern-team regrouped into packs, ops-record from the repo, a jira cache"
"$V/python" -c "
import sys; sys.path.insert(0, '$REPO/python/tests')
from pathlib import Path
from bundles.bundle_support import build_cern_team_catalog
build_cern_team_catalog(Path('$RUN/catalog'))
"
mkdir -p data/jira
cp "$F/jira_records_open.json" data/jira/records.json
echo '{"record_count": 2}' > data/jira/meta.json

echo "== 1. render cern-team --with ops-record,jira and install with okg"
mkdir instance
"$V/archi" render --bundle cern-team --with ops-record,jira --instance instance \
  --pack-root "$RUN/catalog/packs" --bundle-root "$RUN/catalog"
mkdir -p instance/deployments
export OKG_DEPLOYMENTS_DIR="$RUN/instance/deployments"
cd instance
"$V/okg" install --profile cern-team --deployment-name $D --postgres-dsn "$DSN" \
  --deployments-root "$OKG_DEPLOYMENTS_DIR" --archi-data-root "$RUN/data" \
  --non-interactive --json > "$RUN/install.json" || { echo "install failed"; tail -c 3000 "$RUN/install.json"; exit 1; }
"$V/okg" db role-passwords --apply --deployment $D > /dev/null   # read roles for okg filter/search
"$V/okg" status --deployment $D --json > "$RUN/status-0.json"
check "published generation; catalog applied with the ops-record slice" \
  "d['publishing']['latest_published_status'] == 'published' and d['publishing']['latest_published_catalog_version'] >= 1" < "$RUN/status-0.json"
"$V/okg" catalog load --deployment "$OKG_DEPLOYMENTS_DIR/$D" --json > "$RUN/catalog.json"
check "catalog composes the 7 ops-record narrowings" \
  "d['narrowings_by_origin'].get('bridge:ops_record.yaml') == 7" < "$RUN/catalog.json"
sql() { "$V/python" -c "
import psycopg, sys
with psycopg.connect('$DSN') as c:
    print(c.execute(sys.argv[1]).fetchone()[0])" "$1"; }
[ "$(sql "select count(distinct name) from okg.subtypes where name like 'ops\_%'")" = 5 ] \
  && echo "PASS: catalog applied: 5 ops_* subtypes in okg.subtypes" \
  || { echo "FAIL: ops_* subtypes missing from okg.subtypes"; exit 1; }

echo "== 2. a workflow run's records, admitted through okg admission"
"$V/python" -m archi.ops.record submit --deployment $D --records "$F/run1.jsonl" > "$RUN/submit-1.json"
check "run 1 admitted (fast_track, 6 nodes, 6 edges)" \
  "d['output']['status'] == 'accepted' and d['output']['emitted']['node_facts_inserted'] == 6 and d['output']['emitted']['edge_facts_inserted'] == 6" < "$RUN/submit-1.json"
"$V/okg" run --once --deployment $D --apply --json > "$RUN/publish-1.json"
"$V/okg" admission list --deployment $D --json > "$RUN/admissions-1.json"
check "the admission is published in the next generation" \
  "any(a['producer_id'] == 'workflow:archi-ops' and a['status'] == 'published' for a in d['admissions'])" < "$RUN/admissions-1.json"
"$V/okg" search --deployment $D --query "SRM timeout T2_US_MIT" --json > "$RUN/search-1.json"
check "search finds the incident and its outcome" \
  "{'ops_incident:inc-20260914-t2usmit-srm', 'ops_outcome:out-20260914-t2usmit-srm-1'} <= {r['node_id'] for r in d['results']}" < "$RUN/search-1.json"
"$V/okg" expand ops_incident:inc-20260914-t2usmit-srm --deployment $D --direction in --json > "$RUN/expand-1.json"
check "the incident has its proposal and outcome as in-edges" \
  "{(n['subtype'], n['link']['edge_type']) for n in d['neighbors']} == {('ops_proposal', 'addresses'), ('ops_outcome', 'outcome_of')}" < "$RUN/expand-1.json"

echo "== 3. a second run opens an incident tracked in OPS-12; a person resolves OPS-12"
"$V/python" -m archi.ops.record submit --deployment $D --records "$F/run2.jsonl" > "$RUN/submit-2.json"
check "run 2 admitted" "d['output']['status'] == 'accepted'" < "$RUN/submit-2.json"
cp "$F/jira_records_resolved.json" "$RUN/data/jira/records.json"
"$V/okg" ingest --deployment $D --include jira.jira --json > "$RUN/ingest-2.json"
"$V/okg" run --once --deployment $D --apply --json > "$RUN/publish-2.json"
"$V/okg" filter --deployment $D --type jira_issue --where attrs.resolution=Fixed --all --jsonl > "$RUN/tickets-2.jsonl" 2>/dev/null
check "the jira connector reports OPS-12 resolved" \
  "'jira:OPS-12' in {r['node_id'] for r in rows}" < "$RUN/tickets-2.jsonl"

echo "== 4. link ticket resolutions to their incidents"
"$V/python" -m archi.ops.record link-tickets --deployment $D > "$RUN/link-1.json"
check "resolution outcomes admitted" "d['output']['status'] == 'accepted'" < "$RUN/link-1.json"
"$V/okg" run --once --deployment $D --apply --json > "$RUN/publish-3.json"
"$V/okg" filter --deployment $D --type ops_outcome --where attrs.ticket_key=OPS-12 --all --jsonl > "$RUN/outcome-12.jsonl" 2>/dev/null
check "the OPS-12 resolution is the incident outcome, by a person" \
  "len(rows) == 1 and rows[0]['attrs']['incident_id'] == 'inc-20261003-t2usmit-srm' and rows[0]['attrs']['result'] == 'resolved' and rows[0]['attrs']['actor_kind'] == 'person' and rows[0]['attrs']['check_method'] == 'ticket_resolution'" < "$RUN/outcome-12.jsonl"
"$V/okg" filter --deployment $D --type ops_incident --where attrs.record_id=inc-20261003-t2usmit-srm --all --jsonl > "$RUN/incident-12.jsonl" 2>/dev/null
check "the incident is now resolved" "rows[0]['attrs']['status'] == 'resolved'" < "$RUN/incident-12.jsonl"
"$V/okg" expand ops_incident:inc-20261003-t2usmit-srm --deployment $D --direction in --json > "$RUN/expand-12.json"
check "the outcome is linked to the incident" \
  "[(n['node_id'], n['link']['edge_type']) for n in d['neighbors']] == [('ops_outcome:inc-20261003-t2usmit-srm.ticket.OPS-12', 'outcome_of')]" < "$RUN/expand-12.json"
"$V/python" -m archi.ops.record link-tickets --deployment $D > "$RUN/link-2.json"
check "linking again finds nothing new" "d == {'status': 'nothing_to_link'}" < "$RUN/link-2.json"

echo "== 5. recall: the new incident's precedents"
"$V/python" -m archi.ops.precedents --deployment $D --incident inc-20261003-t2usmit-srm > "$RUN/precedents.json"
check "the precedent comes back with its proposal, decision, action and outcome" \
  "[p['record_id'] for p in d['precedents']] == ['inc-20260914-t2usmit-srm'] and d['precedents'][0]['same_error_signature'] and d['precedents'][0]['proposals'][0]['decisions'][0]['verdict'] == 'approved' and d['precedents'][0]['proposals'][0]['actions'][0]['status'] == 'succeeded' and 'resolved' in [o['result'] for o in d['precedents'][0]['outcomes']]" < "$RUN/precedents.json"
"$V/okg" search --deployment $D --query "SRM timeout T2_US_MIT" --scope precedents --json > "$RUN/search-precedents.json"
check "the precedents profile ranks ops records only" \
  "d['search_profile'] == 'precedents' and d['results'] and all(r['node_id'].startswith('ops_') for r in d['results'])" < "$RUN/search-precedents.json"

echo "== done: every check passed"
