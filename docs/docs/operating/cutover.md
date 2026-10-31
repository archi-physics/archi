# Cutover runbook: Archi 3.0 becomes `main`

This page is the order of work for the day the 3.0 line replaces v2 as
`archi-physics/archi` `main`, and `archi-kb` 3.0.0 reaches PyPI. It follows
[ADR 0002](../reference/adr.md) decision D1 and the cutover rules of ADR 0001
section 14. Each step has the exact commands, a check that proves it, and a
way back.

**Every step marked `JASON` needs the repository owner**: an admin setting, a
PyPI account, a written decision or an announcement. Nobody else runs
those. A step with no mark can be run by any maintainer with push rights.

Run the steps in order. Do not start step 7 (the branch swap) until steps 1 to
6 are done, and stop at the stop point inside step 7 if any check fails.

## State of the conditions

ADR 0001 section 14 says the cutover happens when four things hold, "and not
before". Where each stands when 3.0.0 is prepared:

| Condition | State | What closes it |
|---|---|---|
| 1. Every live v2 deployment has a named owner (R1). | **Answered** 2026-10-27 (board d384): two deployments, both moving to 3.0. See step 1. | Step 1 (done). |
| 2. Each of those deployments is migrated or stays on a tagged `v2.x`. | Not started. | Step 4. |
| 3. Parity for comp-ops and the chat migration hold, or the losses are accepted in writing. | **Not met.** `ticket-triage` ships in 3.0 (board d701) and replaces SubMIT's v2 queue bots at the cutover with no trial (board d717), but mail intake and the resolution email stay with the v2 bots until 3.1 (board d704), so those bots stay on `v2.x` until then; and the live comp-ops instance and SubMIT have not been rebuilt from their bundles. See [what has no equivalent yet](../migration-from-v2.md#what-has-no-equivalent-yet). | Step 2 (the written acceptance), `JASON`. |
| 4. The last v2 release is tagged and documented. | The newest tag is `v2.5.0`. `main` has had 52 commits since (check: step 0). | Step 3. |

State today of the repository settings this runbook changes (read with `gh`
on 2026-10-26, so re-check at the time):

- the default branch is `main`, and it is protected: one approving review,
  no force pushes, no deletions, admins not enforced;
- `archi_v3` is not protected;
- there is no `v2` branch;
- the repository has no GitHub environment named `pypi`;
- `archi-kb` has no pending publisher on PyPI. (Nobody has looked at
  pypi.org for this page. Check it in step 6.)

This page was last brought up to date on 2026-10-27, after #849 to #860
merged. `ticket-triage` (#746) ships in 3.0 (board d701); mail intake and the
resolution email stay with the v2 bots until 3.1 (board d704). If anything else
lands before the freeze in step 3, update the "not in 3.0.0" lists in the
changelog, the release notes and the [v2 guide](../migration-from-v2.md) first.

## Step 0. Look at the starting point

Any maintainer. Read-only. (Record the branch tips later, in step 7: step 5
moves `archi_v3`, so a tip written down now would be stale.)

```bash
git clone https://github.com/archi-physics/archi.git archi-cutover && cd archi-cutover
git fetch --tags origin
git tag -l 'v2.*' --sort=-v:refname | head -3
git log --oneline "$(git tag -l 'v2.*' --sort=-v:refname | head -1)..origin/main" | wc -l
gh pr list --repo archi-physics/archi --base main --state open --json number --jq length
gh issue create --repo archi-physics/archi --title "Archi 3.0 cutover" --body "Tracking issue for docs/docs/operating/cutover.md"
```

**Check:** the newest v2 tag and the count of commits after it are known, and
the tracking issue exists. Its number is `<issue>` below.

## Step 1. Inventory the live v2 deployments (R1)

`JASON` decides who is asked. Anyone can gather the answers.

For each machine that may run v2, on that machine:

```bash
archi list-deployments          # v2 command; run it in the v2 environment
```

Record, in one table in the tracking issue: deployment name, host, services
(`chatbot`, `uploader`, queue bots), the v2 tag or commit it runs, the owner
(a named person), and the decision: **move to 3.0** or **stay on a tagged
`v2.x`**. Ask the people who run `archi create` on shared machines and on
SubMIT. A deployment nobody claims is a deployment nobody can move.

**Check:** every row has a named owner and a decision.
**Rollback:** nothing to undo.

**Recorded inventory** (2026-10-27, confirmed by Jason on board d384; also on
issue #758). This list was found read-only, from process and port listings on
two SubMIT hosts, the repository's forks and GitHub environments, and the
v2 configs on `main`. Jason confirmed that FASRC, a CMS comp-ops bot at CERN,
WisDQM, CRAB and MIT course deployments do not run 2.x, and that no other hosts
do.

| Deployment | Host | Services | v2 version | Owner | Decision |
|---|---|---|---|---|---|
| SubMIT A2rchi | a SubMIT host; linked from SubMIT's public A2rchi page | chatbot, data manager, Redmine bot, mailbox bot (a SubMIT service account) | not read (run `archi list-deployments` as the service account) | SubMIT team | move to 3.0 |
| Second chat on SubMIT | a second SubMIT host | chatbot, data manager (her own account) | not read | Maria D'Alfonso (per her fork) | move to 3.0 |

Jason named no person for the SubMIT row. The v2 version of each is still to
be read, in its owner's environment, when the switch-over plan for it is made.
Neither owner has been contacted yet: that waits for the plan and a separate
go from Jason.

## Step 2. Accept the losses in writing

`JASON`. Condition 3 holds only with parity, or with the losses accepted in
writing. 3.0.0 does not have parity. Write the acceptance and keep it with the
inventory. It lists the known limits in the changelog, which are the canonical
list, and the v2 features in [what has no equivalent yet](../migration-from-v2.md#what-has-no-equivalent-yet). In short:

- mail intake and the resolution email stay with the v2 bots until 3.1
  (board d704), so those bots stay on a tagged `v2.x`; `ticket-triage` itself
  ships in 3.0 (board d701) and SubMIT switches to it at the cutover, with no
  trial run first (board d717);
- the live comp-ops instance and SubMIT have not been rebuilt from their
  bundles;
- the smoke sets pass on GPT-6 Luna, not on a local model, and the
  context-eval acceptance run on a live instance has not happened;
- no upload, SSO-page, status-board or Helm equivalents, public Indico events
  only, and one repository per `git-code` pack.

**Check:** the written acceptance is linked from the tracking issue and from
the `r3-release` PACT evidence.
**Rollback:** nothing to undo. Without the acceptance, stop here.

## Step 3. Freeze `main` and tag the last v2 release

`JASON` announces the freeze. Any maintainer with push rights tags. Do this
before step 4, because the owners pin the tag it creates.

1. **Announce the freeze** in the tracking issue and wherever contributors
   read: no merges to `main` from now until step 7 is done. Record the time
   and the tip:

   ```bash
   git fetch origin
   FREEZE_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
   FREEZE_TIP=$(git rev-parse origin/main)
   gh issue comment <issue> --repo archi-physics/archi --body "main frozen at $FREEZE_AT, tip $FREEZE_TIP"
   ```

2. **Check nothing merged since**, just before you tag:

   ```bash
   git fetch origin
   test "$(git rev-parse origin/main)" = "$FREEZE_TIP" && echo "tip unchanged"
   gh pr list --repo archi-physics/archi --base main --state merged \
       --search "merged:>$FREEZE_AT" --json number --jq length      # must print 0
   ```

   If either check fails, stop: the new commits must be in the release, so
   take a new `FREEZE_TIP` and tell the people who merged.

3. **Tag the release.** Pick the next number if `v2.6.0` is taken (step 0
   lists the tags).

   ```bash
   git tag -a v2.6.0 "$FREEZE_TIP" -m "Last v2 release before the 3.0 cutover"
   git push origin v2.6.0
   gh release create v2.6.0 --repo archi-physics/archi --target "$FREEZE_TIP" \
       --title "v2.6.0 (last v2 release)" \
       --notes "The last v2 release. From the cutover this line lives on the v2 branch. See docs/docs/migration-from-v2.md."
   ```

**Check:** `gh release view v2.6.0 --repo archi-physics/archi` shows it, and
`git rev-parse 'v2.6.0^{commit}'` equals `$FREEZE_TIP`.
**Rollback:** `gh release delete v2.6.0 --repo archi-physics/archi --yes` and
`git push origin :refs/tags/v2.6.0`. (Deleting a release or tag is
destructive. `JASON` approves it.) Do it only if nobody has pinned the tag.
Lift the freeze by saying so in the tracking issue.

## Step 4. Move or pin each v2 deployment

Each owner, on the host. This is the real migration. Every deployment in the
inventory takes one of two routes.

**Route A. Move to 3.0.** Follow [Moving from v2 to 3.0](../migration-from-v2.md):
back up v2, stand 3.0 up beside it, move the sources, compare answers. v2 stays
up until the owner is satisfied.

**Route B. Stay on a tagged v2 release.** Pin the deployment to the tag, so a
later change on the v2 branch cannot reach it. In v2, `archi restart` needs
`--name` and `--env-file`. `--config` or `--config-dir` is optional: without
it the deployment keeps the configuration it was rendered with. Add
`--podman` if the deployment runs on podman. The source is that tag's
`src/cli/cli_main.py`.

```bash
cd <the v2 checkout>
git fetch --tags origin
git checkout v2.6.0                    # the tag step 3 created
pip install -e .                       # in the v2 Python environment
archi restart --name <name> --env-file <the secrets .env the deployment was created with> \
    --service chatbot [--podman]
```

Repeat for each service the deployment runs (`--service data_manager`, and so
on). `archi restart` rebuilds the image unless you pass `--no-build`. See the
[v2 CLI reference](../v2/cli_reference.md#archi-restart); that page lists
`--env-file` as optional, which is wrong for the code.

**Check:** for a pinned deployment, `git describe --tags` in its checkout prints the
tag, and the service answers a question. For a moved one, `archi doctor`
exits 0 on the new instance and its owner signs off on the comparison.
**Rollback:** a moved deployment: `archi down --purge --yes` on the 3.0
instance, v2 is untouched. A pinned one: `git checkout` the previous tag or
commit and run the same `archi restart` again.

## Step 5. Merge the release-readiness change into `archi_v3`

A maintainer merges the pull request that holds this page and the version
bump to 3.0.0, after CI is green, not before:

```bash
gh pr checks <number> --repo archi-physics/archi
gh pr merge <number> --repo archi-physics/archi --merge --match-head-commit <sha>
```

Use `--merge`, never `--squash`: a squash drops the approval trailers PACT
reads. `archi_v3` has no required checks, so the merge happens at once. That is why
you check first.

Then stop merging to `archi_v3` as well. Announce that in the tracking issue.

**Check:** on `archi_v3`, `grep '^version' python/pyproject.toml` prints
`version = "3.0.0"`, and `grep __version__ python/archi/__init__.py` prints
the same.
**Rollback:** `git revert -m 1 <merge sha>` on `archi_v3`, as a pull request.

## Step 6. Prepare PyPI (once)

`JASON`. Nothing here can be done from the repository. The trusted
publisher is how `release.yml` publishes without a token.

1. On pypi.org, signed in as the owner: **Your projects, Publishing, Add a new
   pending publisher**. Enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `archi-kb` |
   | Owner | `archi-physics` |
   | Repository name | `archi` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

2. On GitHub, create the environment the workflow names. Add yourself as a
   required reviewer if you want a pause before each publish:

   ```bash
   gh api --method PUT repos/archi-physics/archi/environments/pypi
   gh api repos/archi-physics/archi/environments --jq '.environments[].name'
   ```

   (Or use Settings, Environments, New environment, `pypi`.)

3. **Install the wheel before the irreversible upload.** Build it exactly as
   `release.yml` does, install it into a clean virtual environment, and smoke
   test it. Outside the repository, in a scratch directory:

   ```bash
   W=$(mktemp -d)
   git clone --branch archi_v3 https://github.com/archi-physics/archi.git "$W/src"
   (cd "$W/src/python" && python3 -m venv "$W/build-venv" && "$W/build-venv/bin/pip" install build \
       && "$W/build-venv/bin/python" -m build --wheel --outdir "$W/dist")
   ls "$W/dist"                          # archi_kb-3.0.0-py3-none-any.whl
   python3 -m venv "$W/venv"
   "$W/venv/bin/pip" install \
       "okg @ https://github.com/mitdbg/akmon/releases/download/v0.2.0/okg-0.2.0-py3-none-any.whl#sha256=72bac3b67cf29cccd27e77e58ad309104f19fcce255a04a4a6f23b8318b88e12" \
       "$W"/dist/archi_kb-3.0.0-py3-none-any.whl
   "$W/venv/bin/python" -c "import archi; print(archi.__version__)"   # 3.0.0
   "$W/venv/bin/archi" bundles validate                              # exits 0
   "$W/venv/bin/archi" bundles list | head
   rm -rf "$W"
   ```

   TestPyPI is not an option: `release.yml` publishes to PyPI only and has no
   switch for another index, so this local install is the rehearsal.

**Check:** the environment list prints `pypi`; the wheel is named
`archi_kb-3.0.0-py3-none-any.whl`; the smoke commands succeed. The pending
publisher itself is first exercised by step 9.
**Rollback:** `gh api --method DELETE repos/archi-physics/archi/environments/pypi`;
delete the pending publisher on pypi.org. Nothing has been published.

## Step 7. Swap the branches

`JASON`. This is the cutover itself. It needs repository admin: GitHub allows
only an admin to rename the default branch or a protected branch.

### What GitHub does on a rename

From GitHub's page
[Renaming a branch](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/renaming-a-branch)
(read 2026-10-26):

- "Branch protection policies are also updated." The rule follows the branch to
  its new name.
- The base branch of open pull requests (including those from forks) and
  draft releases is changed to the new name. A pull request whose *head* is the
  renamed branch is closed.
- URLs with the old name redirect. "Raw file URLs are not redirected." "GitHub
  Actions workflows do not follow renames."
- A `git pull` for the old name is not redirected, and every local clone must
  be updated.
- "Some branches can only be renamed by a repository administrator: the
  repository's default branch, and any branch covered by a branch protection
  or a repository-level branch ruleset."

So the first rename moves `main`'s protection to `v2`, and the new `main`
(the old `archi_v3`) has **no protection** until you restore it. Do that
immediately, as the next command.

### 7a. Before the renames

Record the tips in the tracking issue, not only in your shell, and capture the
current protection rule:

```bash
git fetch origin
V2_TIP=$(git rev-parse origin/main);      echo "v2 tip  $V2_TIP"
V3_TIP=$(git rev-parse origin/archi_v3);  echo "v3 tip  $V3_TIP"
test "$V2_TIP" = "$FREEZE_TIP" || echo "STOP: main moved since the freeze"
gh issue comment <issue> --repo archi-physics/archi --body "before swap: v2 tip $V2_TIP, v3 tip $V3_TIP"

gh api repos/archi-physics/archi/branches/main/protection > main-protection.json
jq '{
  required_status_checks: (if .required_status_checks then {strict: .required_status_checks.strict, contexts: .required_status_checks.contexts} else null end),
  enforce_admins: .enforce_admins.enabled,
  required_pull_request_reviews: (if .required_pull_request_reviews then (.required_pull_request_reviews | {dismiss_stale_reviews, require_code_owner_reviews, require_last_push_approval, required_approving_review_count}) else null end),
  restrictions: (if .restrictions then {users: [.restrictions.users[].login], teams: [.restrictions.teams[].slug], apps: [.restrictions.apps[].slug]} else null end),
  required_linear_history: .required_linear_history.enabled,
  allow_force_pushes: .allow_force_pushes.enabled,
  allow_deletions: .allow_deletions.enabled,
  block_creations: .block_creations.enabled,
  required_conversation_resolution: .required_conversation_resolution.enabled,
  lock_branch: .lock_branch.enabled,
  allow_fork_syncing: .allow_fork_syncing.enabled
}' main-protection.json > main-protection-put.json
cat main-protection-put.json
jq '.required_signatures.enabled' main-protection.json     # false today; if true, see the note below
```

Read `main-protection-put.json`. On 2026-10-26 it said: one approving review,
no dismiss-stale, no code-owner review, no force pushes, no deletions, admins
not enforced, no required checks, no push restrictions. If the rule has changed,
the file is still right: it is built from the live rule. If signatures are
required, add them afterwards with
`gh api --method POST repos/archi-physics/archi/branches/main/protection/required_signatures`.

Attach both files to the tracking issue (`gh issue comment` with the text, or
upload them) so the rule can be rebuilt by hand.

### 7b. The renames, with nothing between them and the restore

Run these four commands as one block. Do not run anything else in between.

```bash
gh api --method POST repos/archi-physics/archi/branches/main/rename -f new_name=v2 \
&& gh api --method POST repos/archi-physics/archi/branches/archi_v3/rename -f new_name=main \
&& gh api --method PUT repos/archi-physics/archi/branches/main/protection --input main-protection-put.json \
&& gh repo edit archi-physics/archi --default-branch main
```

(`&&` stops at the first failure. If it stops, go to the rollback.)

### 7c. Stop point: verify, then stop if anything is off

```bash
# 1. The new main has the rule; v2 kept its rule (it followed the rename).
gh api repos/archi-physics/archi/branches/main/protection --jq '.required_pull_request_reviews.required_approving_review_count, .allow_force_pushes.enabled, .allow_deletions.enabled'
gh api repos/archi-physics/archi/branches/v2/protection   --jq '.required_pull_request_reviews.required_approving_review_count, .allow_force_pushes.enabled, .allow_deletions.enabled'
# 2. Default branch and tips.
gh api repos/archi-physics/archi --jq .default_branch                       # main
git ls-remote --heads origin main v2 archi_v3                               # main and v2 only
test "$(git ls-remote origin refs/heads/main | cut -f1)" = "$V3_TIP" && echo "main is the v3 line"
test "$(git ls-remote origin refs/heads/v2   | cut -f1)" = "$V2_TIP" && echo "v2 holds the v2 line"
```

Both protection reads must print `1`, `false`, `false`. Both `test` lines must
print. `archi_v3` must not be listed. Then open
<https://github.com/archi-physics/archi> in a browser and confirm that the
default branch is `main` and shows the 3.0 README, and
<https://github.com/archi-physics/archi/tree/v2> shows v2.

**If any check fails, stop.** Do not run step 8 or 9. Do the rollback below.
If only the new `main`'s protection is missing, re-run the `PUT` from 7b; the
swap itself is fine.

### 7d. Tell local clones

In every clone of the repository:

```bash
git fetch origin
git branch -m main v2 && git branch -u origin/v2 v2          # a clone that had main (v2)
git branch -m archi_v3 main && git branch -u origin/main main  # a clone that had archi_v3
git remote set-head origin -a
```

A clone that was on v2's `main` and only wants v2 runs `git fetch && git
switch v2` instead, and must never `git pull` on its old `main`.

### Rollback of step 7

Precondition: nobody has pushed to the new `main` since 7b. Check:
`test "$(git ls-remote origin refs/heads/main | cut -f1)" = "$V3_TIP"`. If it
fails, stop and talk first: a swap back moves their pull requests again.

```bash
gh api --method POST repos/archi-physics/archi/branches/main/rename -f new_name=archi_v3
gh api --method POST repos/archi-physics/archi/branches/v2/rename   -f new_name=main
gh repo edit archi-physics/archi --default-branch main
```

The rules follow the branches again: the new `archi_v3` carries the rule the
new `main` had, and `main` (v2) gets back the rule it had. Check with the
same commands as 7c with the names swapped, using `git ls-remote --heads
origin` and the web pages, and re-apply `main-protection-put.json` to `main`
if its rule is missing. Remove the rule from `archi_v3` in Settings, Branches,
if you want it as it was.

## Step 8. Fix the workflow triggers

A maintainer, in a pull request on the new `main`. These edits are in
`.github/workflows/`, which the release-readiness change does not touch:

- `ci.yml`: the push trigger lists `[main, archi_v3]`. Drop `archi_v3`.
- `nightly-e2e.yml`: the push trigger lists `archi_v3`. Change it to `main`.
- `deploy-docs.yml`: it deploys the site on a push to `main`. The rename is not
  a push, so run it once by hand:

  ```bash
  gh workflow run deploy-docs.yml --repo archi-physics/archi --ref main
  gh run list --workflow deploy-docs.yml --repo archi-physics/archi --limit 1
  ```

The workflows on the `v2` branch name `main`. They do not run on `v2` until
a v2 pull request changes them. Do that when the first v2 fix needs CI.

**Check:** the next push to `main` runs `ci`; the site at
<https://archi-physics.github.io/archi/> shows the 3.0 pages.
**Rollback:** revert the pull request.

## Step 9. Tag 3.0.0 and publish

`JASON` for the tag, because pushing it publishes to PyPI at once and a
published version cannot be reused. Only after step 7's checks pass.

Do not use `git pull` here: a clone's old local `main` may be the v2 line.
Make the local `main` the remote one, and prove it:

```bash
git fetch origin
git switch -C main origin/main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" && echo "HEAD is origin/main"
grep '^version' python/pyproject.toml          # version = "3.0.0"
git tag -a v3.0.0 -m "archi-kb 3.0.0"
git push origin v3.0.0
```

The tag starts `release.yml`. It refuses when the tag differs from the version
in `python/pyproject.toml` or in `archi.__version__`. It builds the wheel, then
publishes through the `pypi` environment. Watch it:

```bash
gh run list --workflow release.yml --repo archi-physics/archi --limit 1
gh run watch --repo archi-physics/archi "$(gh run list --workflow release.yml --repo archi-physics/archi --limit 1 --json databaseId --jq '.[0].databaseId')"
```

If you made yourself a required reviewer in step 6, approve the `publish` job
in the run.

Then create the GitHub release from the release notes:

```bash
gh release create v3.0.0 --repo archi-physics/archi --verify-tag \
    --title "archi 3.0.0" --notes-file docs/release-notes-3.0.0.md
```

**Check (this is the `r3-release` acceptance):** in an empty directory and a
new environment. The last line cleans up the environment.

```bash
V=$(mktemp -d)
python3 -m venv "$V/venv"
"$V/venv/bin/pip" install "okg @ https://github.com/mitdbg/akmon/releases/download/v0.2.0/okg-0.2.0-py3-none-any.whl#sha256=72bac3b67cf29cccd27e77e58ad309104f19fcce255a04a4a6f23b8318b88e12" archi-kb==3.0.0
"$V/venv/bin/python" -c "import archi; print(archi.__version__)"     # 3.0.0
"$V/venv/bin/archi" init cutover-check --dir "$V/instance"
"$V/venv/bin/archi" doctor --instance "$V/instance"
"$V/venv/bin/archi" down --purge --yes --instance "$V/instance"
rm -rf "$V"
```

`archi doctor` must exit 0. Record the output in the `r3-release` PACT evidence.

**If the workflow fails before anything is uploaded.** The `build` job (tag does
not match the version, build error) or the `publish` job before upload (the
pending publisher or the `pypi` environment is wrong) leaves nothing on PyPI.
Check `https://pypi.org/project/archi-kb/` shows no 3.0.0. Then fix the cause
(the settings in step 6, or a pull request for the version) and, if the tag is
still the right commit, re-run the failed jobs without moving the tag:

```bash
gh run rerun --failed --repo archi-physics/archi <run id>
```

If the fix needs a code change, delete the tag (`git push origin :refs/tags/v3.0.0`),
merge the fix, and tag again.

**Rollback once something was uploaded.** A published version stays on PyPI.
Do not reuse the number. `JASON` yanks the release on pypi.org (Manage,
Releases, Options, Yank), fixes the problem, and releases `3.0.1`. Delete the
GitHub release (`gh release delete v3.0.0 --yes`) only if you also delete the
tag, and only when nobody has installed it.

## Step 10. Announce

`JASON` sends it. Include:

- where to start ([quickstart](../quickstart.md)) and the
  [release notes](https://github.com/archi-physics/archi/blob/main/docs/release-notes-3.0.0.md);
- the supported AKMON version (okg 0.2.0) and what 3.0.0 does not have
  (step 2);
- **what changed for v2 users.** `main` is now 3.0 and `v2` is v2. A v2
  checkout must run `git fetch && git switch v2`. Do not `git pull` on `main`.
  Raw URLs that name `main`
  (`raw.githubusercontent.com/archi-physics/archi/main/...`) now return 3.0 files
  and are not redirected. Links in GitHub Actions `uses: ...@main` do not
  follow the rename either;
- the date the v2 branch stops receiving fixes. That date is set from the
  inventory in step 1, not before it.

Also: point the archi-okg README at this repository if it does not already,
and close the `3.0 · M7` milestone once the `r3-docs` and `r3-release` PACTs
are approved.

**Rollback:** send a correction.

## If something goes wrong after the cutover

| Problem | Do this |
|---|---|
| A v2 owner needs a fix. | Open the pull request against the `v2` branch. Tag `v2.<n>`. The owner pins the new tag as in step 4, route B. |
| A v2 deployment must be rebuilt. | `git checkout v2.6.0` (or its pinned tag), then the restore steps in [Moving from v2 to 3.0](../migration-from-v2.md#roll-back). |
| Someone's clone is confused by the renames. | Step 7d, or a fresh clone. |
| 3.0.0 has a bad bug. | Fix on `main`, bump to `3.0.1`, tag `v3.0.1`. Yank 3.0.0 only if it is harmful. |
| The branch swap itself was wrong. | The rollback of step 7. It is safe until people push to the new `main`. After that, talk first. |
