# Verification commands

All local proof below ran on consumer commit `378ff1c` using Python3.12.13 and the recovered framework wheel from source `528279f6abfd53ed2563ae72be0a3db3ac6e0e2a`. Hosted CI separately uses the repository's unchanged `34efbad1b5768fd3ca635992173694a9cc91ebd2` pin.

The actual test commands were:

```sh
python -m pytest python/tests/sources/test_cmssw.py -q --junitxml=<task-focused.xml>
python -m pytest python/tests -q --junitxml=<task-full.xml>
# Working directory for the build: python/
python -m hatchling build -t wheel
python -m black --check python/archi/sources/cmssw.py python/tests/sources/test_cmssw.py
python -m isort --check-only python/archi/sources/cmssw.py python/tests/sources/test_cmssw.py
```

Results:26 focused tests passed;357 full-suite tests passed;zero skipped,failed or errored rows in either JUnit file. Build and both targeted formatting checks exited0. Formatting is report-only in the repository CI, so these checks do not claim repository-wide formatting enforcement.

Each suite/build ran through a task-local execution file that acquired the existing OKG machine mutex and called its unchanged process guard before invoking the command. No database selector was configured because these commands use no database. The first inline heredoc was refused when the guard classified its own shell argv containing the pytest command as another test process; no test ran in that attempt. Moving the same orchestration into a normal file resolved that wrapper issue without changing process rows, selectors, or guard code.

The installed-artifact proof used a new virtual environment outside all authoring checkouts. Hash-required offline dependencies came from the already-verified Linux dependency manifest; the original framework wheel and new consumer wheel were installed with `uv pip install --no-index --no-deps`. It then ran `python -I <task-installed-check.py>` from outside the checkout. The following is that actual check with private roots replaced by placeholders:

```python
import hashlib,inspect,json,pathlib,urllib.request
import archi,yaml
from archi.sources.cmssw import CMSSWReleaseSource
from okg.deployment import NodeFact,EdgeFact
root=pathlib.Path('<isolated-installed-environment>').resolve();origin=pathlib.Path(inspect.getfile(CMSSWReleaseSource)).resolve();assert origin.is_relative_to(root)
pkg=pathlib.Path(archi.__file__).parent;example=pkg/'bundles/cern-team/source-defaults/cmssw_releases_frozen.yaml.example';assert example.is_file()
work=pathlib.Path('<task-fixture-input>');work.mkdir(exist_ok=True);path=work/'releases.map';body=b'architecture=el8_amd64_gcc12;label=CMSSW_14_0_1;type=Production;state=Announced;\narchitecture=el8_amd64_gcc12;label=CMSSW_14_0_2;type=Production;state=Announced;\n';path.write_bytes(body);pin='sha256:'+hashlib.sha256(body).hexdigest()
params=yaml.safe_load(example.read_text().replace('${cmssw_map_path}',str(path)).replace('${cmssw_map_digest}',pin))['cmssw_releases']['params']
def refuse(*a,**k):raise AssertionError('network called')
urllib.request.urlopen=refuse
s=CMSSWReleaseSource(**params);assert s.preflight().status=='ok';run=s.run('installed-fixture',mode='scope_complete');facts=list(run.facts);nodes={x.node_id for x in facts if isinstance(x,NodeFact)};edges={(x.src,x.dst,x.edge_type) for x in facts if isinstance(x,EdgeFact)};assert nodes=={'cmssw_release:CMSSW_14_0_X','cmssw_release:CMSSW_14_0_1','cmssw_release:CMSSW_14_0_2'};assert edges=={('cmssw_release:CMSSW_14_0_2','cmssw_release:CMSSW_14_0_1','supersedes')};assert all(x.source_revision['map_cache_digest']==pin for x in facts)
path.write_text('changed')
try:s.run('mismatch');raise AssertionError('mismatch accepted')
except ValueError as exc:assert 'digest mismatch' in str(exc)
path.unlink()
try:s.run('missing');raise AssertionError('missing accepted')
except FileNotFoundError:pass
print(json.dumps({'status':'pass','fixture_only_not_real_baseline':True,'consumer_origin':str(origin),'example_origin':str(example),'installed_origin_verified':True,'example_packaged':True,'source_sha256':hashlib.sha256(origin.read_bytes()).hexdigest(),'example_sha256':hashlib.sha256(example.read_bytes()).hexdigest(),'nodes':sorted(nodes),'edges':sorted(edges),'map_cache_digest':pin,'mismatch_refused':True,'missing_refused':True,'network_trap_active':True},indent=2))
```

This fixture only proves the installed consumer contract. It is not the unavailable historical catalog and does not prove graph publication. Exact wheel/member hashes, public test identities and raw JUnit digests are retained in artifact.json and tests.json. No synthetic data is labeled real deployment evidence.

## Applicable repository checks

This consumer repository has no OKG agent_preflight.py or managed PR-body checker. Its actual CI runs `python -m pytest python/tests -q` on same-repository pull requests targeting archi_v3. The applicable PACT gate is the operator environment's v5 `python -m pact.cli check frozen-cmssw-input --root <consumer-worktree> --base archi_v3 --strict --json`. Approval has not been recorded. No absent or skipped gate is reported as passed.
