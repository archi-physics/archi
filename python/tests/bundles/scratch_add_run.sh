#!/usr/bin/env bash
# acc.r3-bundle-format.add: archi add on a published scratch instance publishes
# again without manual steps.
#
# Usage: ARCHI_SCRATCH_DSN=postgresql://... scratch_add_run.sh <empty work dir>
#
# Needs an EMPTY database on an AKMON Postgres (the quickstart's
# ghcr.io/mitdbg/akmon-postgres image), and okg + archi installed in the Python environment whose bin/ holds
# `python` (set V to that bin/ directory; default: the one on PATH).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
V="${V:-$(dirname "$(command -v python)")}"
RUN="$1"
DSN="${ARCHI_SCRATCH_DSN:?set ARCHI_SCRATCH_DSN to an empty AKMON database}"
mkdir -p "$RUN"
[ -z "$(ls -A "$RUN")" ] || { echo "$RUN is not empty"; exit 1; }
RUN="$(cd "$RUN" && pwd)"
cd "$RUN"
unset OKG_PROFILES_DIR

echo "== 0. inputs: the shipped cern-team packs, a docsite cache and a local git repo"
"$V/python" -c "
import sys; sys.path.insert(0, '$REPO/python/tests')
from pathlib import Path
from bundles.bundle_support import build_cern_team_catalog
build_cern_team_catalog(Path('$RUN/catalog'))
"
mkdir -p data/docsite
printf '[{"title": "Start", "url": "https://docs.example.org/start", "body": "How to start."}]\n' \
  > data/docsite/records.json
mkdir demo-repo
git -C demo-repo init -q -b main
printf '# demo\n\nA tiny repository for the archi add run.\n' > demo-repo/README.md
printf 'def hello():\n    return "hello"\n' > demo-repo/hello.py
git -C demo-repo add -A
git -C demo-repo -c user.email=t@example.org -c user.name=t commit -qm "demo"

ROOTS=(--pack-root "$RUN/catalog/packs"
       --pack-root "$REPO/python/tests/bundles/fixtures/thirdparty/archi_fixture_packs/packs"
       --bundle-root "$RUN/catalog")

echo "== 1. render the instance (cern-team without indico: docsite + playbooks)"
mkdir instance
"$V/archi" render --bundle cern-team --without indico --instance instance "${ROOTS[@]}"

echo "== 2. install and publish with okg, from the instance dir (no OKG_PROFILES_DIR)"
# Not a git repo: the rendered profile declares standalone release provenance.
mkdir -p instance/deployments
export OKG_DEPLOYMENTS_DIR="$RUN/instance/deployments"
(cd instance && "$V/okg" install --profile cern-team --deployment-name r3bf \
   --postgres-dsn "$DSN" --deployments-root "$OKG_DEPLOYMENTS_DIR" \
   --archi-data-root "$RUN/data" --non-interactive --json > "$RUN/install.json") || {
  echo "install failed"; tail -c 3000 "$RUN/install.json"; exit 1; }
"$V/python" -c "import json; d=json.load(open('$RUN/install.json')); print('install status:', d.get('status'), d.get('generation_id') or d.get('published_generation'))"
"$V/okg" status --deployment r3bf --json > "$RUN/status-before.json" || true

echo "== 3. archi diff on the fresh install"
"$V/archi" diff --instance instance --deployment "$OKG_DEPLOYMENTS_DIR/r3bf" "${ROOTS[@]}"

echo "== 4. archi add repo-mirror: one command, no manual step"
git config --global --get user.email >/dev/null || export GIT_AUTHOR_EMAIL=t@example.org GIT_COMMITTER_EMAIL=t@example.org GIT_AUTHOR_NAME=t GIT_COMMITTER_NAME=t
"$V/archi" add repo-mirror --instance instance --deployment "$OKG_DEPLOYMENTS_DIR/r3bf" \
  --answer repo_mirror_url="file://$RUN/demo-repo" --answer repo_mirror_name=demo "${ROOTS[@]}"

echo "== 5. the new source's records are in the published graph"
"$V/okg" status --deployment r3bf --json > "$RUN/status-after.json"
"$V/okg" filter --deployment r3bf --json --where '{"type": "source_file"}' > "$RUN/filter-after.json" 2>&1 || true
"$V/okg" search --deployment r3bf --query hello --json > "$RUN/search-after.json" 2>&1 || true
"$V/python" - "$RUN" <<'EOF'
import json, sys
run = sys.argv[1]
for name in ("status-before", "status-after"):
    try:
        d = json.load(open(f"{run}/{name}.json"))
    except Exception as exc:
        print(name, "unreadable", exc); continue
    gen = d.get("published_generation") or d.get("generation") or {}
    print(name, json.dumps(gen)[:400] if gen else json.dumps(d)[:600])
print("search:", open(f"{run}/search-after.json").read()[:800])
EOF

echo "== 6. archi diff after the add: clean, then a hand edit to the registry shows"
"$V/archi" diff --instance instance "${ROOTS[@]}" && echo "diff clean"
sed -i '' 's/manage_checkout: true/manage_checkout: false/' "$OKG_DEPLOYMENTS_DIR/r3bf/source_registry.yaml"
if "$V/archi" diff --instance instance "${ROOTS[@]}"; then echo "UNEXPECTED: no drift"; exit 1; fi
echo "== done"
