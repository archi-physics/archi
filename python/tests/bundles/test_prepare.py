"""req.r3-bundle-cern-team.bundle: pack `prepare` steps and the bare cern-team install.

A pack declares `prepare:` (docs/docs/reference/pack-format.md); `archi init` and
`archi add` run it before okg's first ingest. These tests hold the mechanism
(substitution, order, failure) and what it gives the shipped cern-team bundle:
a bare install that has its inputs. The published bare install on a real
Postgres is the executed test recorded in the PACT
(acc.r3-bundle-cern-team.bare; python/tests/cli/golden_path_run.sh).
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from archi.bundles import discovery
from archi.bundles.lock import LockError, add, render_and_lock
from archi.bundles.manifest import validate_pack
from archi.bundles.prepare import PrepareError, Step, question_values, run_steps, steps_for
from archi.bundles.render import render
from archi.bundles.resolve import resolve
from archi.cli import init as init_mod
from archi.cli.init import InitError, init
from archi.stack import compose as stack_compose

from bundles.bundle_support import CATALOG, build_cern_team_catalog, edit_yaml, load_catalog
from bundles.test_lock import BASE_ANSWERS, _scaffold
from cli.test_init import FakeOkg
from packs.pack_support import shipped_catalog


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for key, value in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.org"),
                       ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.org")):
        monkeypatch.setenv(key, value)


@pytest.fixture(scope="module")
def shipped():
    return shipped_catalog()


def _alpha_with(tmp_path, prepare):
    """A copy of the fixture catalog whose alpha pack declares `prepare`."""
    import shutil

    root = tmp_path / "catalog"
    shutil.copytree(CATALOG, root)
    edit_yaml(root / "packs" / "alpha" / "pack.yaml", lambda d: d.update(prepare=prepare))
    return root


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_a_valid_prepare_declaration_passes(tmp_path):
    root = _alpha_with(tmp_path, [
        {"name": "fetch", "module": "archi.downloaders.static_docs",
         "args": ["--urls", "${site_url}", "--root", "${archi_data_root}"], "timeout": 60},
    ])
    assert validate_pack(root / "packs" / "alpha") == []
    pack = discovery.from_roots([root / "packs"], [root]).packs["alpha"]
    assert pack.prepare[0]["timeout"] == 60


def test_the_shipped_packs_with_prepare_validate(shipped):
    declared = {n: [s["name"] for s in p.prepare] for n, p in shipped.packs.items() if p.prepare}
    assert declared == {"docsite": ["records-cache"]}


# ---------------------------------------------------------------------------
# substitution
# ---------------------------------------------------------------------------


def test_answers_fill_the_args_by_rendered_id_and_defaults_apply(tmp_path):
    root = _alpha_with(tmp_path, [
        {"name": "fetch", "module": "a.b", "args": ["--url=${site_url}", "${archi_data_root}"]},
    ])
    pack = load_catalog(root).packs["alpha"]
    questions = [{"id": "alpha_site_url", "default": "https://example.org/docs"},
                 {"id": "archi_data_root", "default": "data"}]

    [step] = steps_for([pack], question_values(questions, {"archi_data_root": "/srv/d"}))
    assert step.command == (sys.executable, "-m", "a.b", "--url=https://example.org/docs", "/srv/d")
    assert step.label == "alpha:fetch" and step.timeout == 600

    [step] = steps_for([pack], question_values(questions, {"alpha_site_url": "https://x.org/"}))
    assert "--url=https://x.org/" in step.command


def test_an_unanswered_question_is_an_error_not_an_empty_argument(tmp_path):
    root = _alpha_with(tmp_path, [{"name": "fetch", "module": "a.b", "args": ["${site_url}"]}])
    pack = load_catalog(root).packs["alpha"]
    with pytest.raises(PrepareError, match="alpha_site_url"):
        steps_for([pack], question_values([{"id": "alpha_site_url"}], {}))


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def _step(name, *, timeout=5):
    return Step("alpha", name, ("python", "-m", name), timeout)


def test_steps_run_in_order_with_the_data_root(tmp_path):
    calls = []

    def runner(cmd, **kw):
        calls.append((cmd[-1], kw["env"]["ARCHI_DATA_ROOT"], kw["cwd"], kw["timeout"], kw.get("shell")))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    notes = []
    run_steps([_step("one"), _step("two")], tmp_path / "root", echo=notes.append, runner=runner)
    root = str(tmp_path / "root")
    assert calls == [("one", root, root, 5, None), ("two", root, root, 5, None)]
    assert notes == ["preparing alpha:one ...", "preparing alpha:two ..."]


def test_the_first_failure_stops_the_rest_and_carries_the_output(tmp_path):
    ran = []

    def runner(cmd, **kw):
        ran.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 3 if cmd[-1] == "one" else 0, "out", "boom: page 404")

    with pytest.raises(PrepareError, match=r"(?s)alpha:one: failed \(exit 3\).*boom: page 404"):
        run_steps([_step("one"), _step("two")], tmp_path, echo=lambda _: None, runner=runner)
    assert ran == ["one"]


def test_a_timeout_and_a_missing_interpreter_are_failures(tmp_path):
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    def gone(cmd, **kw):
        raise FileNotFoundError("no python")

    with pytest.raises(PrepareError, match="stopped after 5s"):
        run_steps([_step("one")], tmp_path, echo=lambda _: None, runner=slow)
    with pytest.raises(PrepareError, match="could not start"):
        run_steps([_step("one")], tmp_path, echo=lambda _: None, runner=gone)


def test_a_real_module_runs_with_the_environment(tmp_path, monkeypatch):
    """No fake runner: the step is `python -m`, with ARCHI_DATA_ROOT set."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "writer.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['ARCHI_DATA_ROOT'], 'marker').write_text('ran')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    root = tmp_path / "root"
    run_steps([Step("alpha", "w", (sys.executable, "-m", "pkg.writer"), 30)], root,
              echo=lambda _: None)
    assert (root / "marker").read_text() == "ran"


# ---------------------------------------------------------------------------
# archi init: prepare runs after render and before anything is started
# ---------------------------------------------------------------------------


def _init_args(tmp_path, shipped):
    return dict(bundles=["cern-team"], catalog=shipped, directory=tmp_path / "myteam",
                data_root=tmp_path / "data", echo=lambda _: None,
                wait_db=lambda dsn: None)


def test_init_prepares_before_the_database_and_the_first_ingest(tmp_path, shipped, monkeypatch):
    monkeypatch.setattr(stack_compose, "detect_runtime", lambda preferred, runner: "docker")
    monkeypatch.setattr(init_mod, "okg_version", lambda: "0.2.0")
    order = []
    okg = FakeOkg()

    def okg_runner(cmd, **kw):
        order.append("okg " + cmd[1])
        return okg(cmd, **kw)

    def compose(cmd, **kw):
        order.append("compose " + " ".join(cmd[-3:]))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def prepare(cmd, **kw):
        order.append("prepare " + cmd[2])
        assert kw["env"]["ARCHI_DATA_ROOT"] == str((tmp_path / "data").resolve())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    init("myteam", okg_runner=okg_runner, compose_runner=compose, prepare_runner=prepare,
         **_init_args(tmp_path, shipped))
    assert order[0] == "prepare archi.downloaders.static_docs"
    assert order.index("prepare archi.downloaders.static_docs") < order.index("compose up -d graph-db")
    assert order.index("compose up -d graph-db") < order.index("okg install") < order.index("okg provision")
    assert order.count("prepare archi.downloaders.static_docs") == 1


def test_init_passes_the_urls_answer_and_the_default(tmp_path, shipped, monkeypatch):
    monkeypatch.setattr(stack_compose, "detect_runtime", lambda preferred, runner: "docker")
    monkeypatch.setattr(init_mod, "okg_version", lambda: "0.2.0")
    seen = []

    def prepare(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def run(name, answers):
        args = _init_args(tmp_path / name, shipped)
        args["directory"] = tmp_path / name / "inst"
        args["data_root"] = tmp_path / name / "data"
        (tmp_path / name).mkdir()
        init("myteam", answers=answers, okg_runner=FakeOkg(), prepare_runner=prepare,
             compose_runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
             **args)
        return seen[-1]

    default = run("a", {})
    assert default[3] == "--skip-gitlab"
    assert default[default.index("--docsite-output") + 1] == str(
        (tmp_path / "a" / "data").resolve() / "docsite" / "records.json"
    )
    assert default[-2] == "--urls"
    assert default[-1] == (
        "https://fts3-docs.web.cern.ch/fts3-docs/,https://rucio.readthedocs.io/en/latest/"
    )
    assert run("b", {"docsite_urls": "https://example.org/docs/"})[-1] == "https://example.org/docs/"


def test_a_failed_step_stops_init_before_anything_starts(tmp_path, shipped, monkeypatch):
    monkeypatch.setattr(stack_compose, "detect_runtime", lambda preferred, runner: "docker")
    monkeypatch.setattr(init_mod, "okg_version", lambda: "0.2.0")
    started = []

    def prepare(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "error: 1 requested page(s) not cached")

    with pytest.raises(InitError, match="(?s)docsite:records-cache: failed.*not cached.*not installed"):
        init("myteam", okg_runner=lambda *a, **k: started.append("okg"),
             compose_runner=lambda cmd, **k: started.append("compose") or subprocess.CompletedProcess(cmd, 0, "", ""),
             prepare_runner=prepare, **_init_args(tmp_path, shipped))
    assert started == []


def test_no_okg_runs_no_prepare(tmp_path, shipped):
    def prepare(cmd, **kw):
        raise AssertionError("--no-okg starts and fetches nothing")

    init("myteam", run_okg=False, prepare_runner=prepare, **_init_args(tmp_path, shipped))


# ---------------------------------------------------------------------------
# archi add: prepare runs before the deployment changes
# ---------------------------------------------------------------------------


@pytest.fixture
def cern(tmp_path_factory):
    root = tmp_path_factory.mktemp("cern-add")
    build_cern_team_catalog(root)
    catalog = discovery.from_roots([root / "packs"], [root])
    assert catalog.problems == []
    return catalog


def test_add_prepares_the_new_pack_before_it_edits_the_deployment(cern, tmp_path, monkeypatch):
    instance = tmp_path / "instance"
    render_and_lock(cern, instance, ["cern-team"], without_packs=["docsite"])
    deployment = _scaffold(instance, "cern-team", BASE_ANSWERS)
    registry = deployment / "source_registry.yaml"
    before = registry.read_text()

    seen = []

    def failing(steps, data_root, **kw):
        seen.append(([s.label for s in steps], data_root))
        raise PrepareError("docsite:records-cache: failed (exit 1); nothing was ingested.")

    monkeypatch.setattr("archi.bundles.prepare.run_steps", failing)
    with pytest.raises(LockError, match="preparing the new pack failed.*records-cache"):
        add(instance, cern, "docsite", {}, deployment_dir=deployment, echo=lambda *_: None)
    assert seen == [(["docsite:records-cache"], Path("/srv/data"))]
    assert registry.read_text() == before, "a failed prepare must leave the deployment untouched"


# ---------------------------------------------------------------------------
# the shipped cern-team bundle installs bare
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bare(shipped, tmp_path_factory):
    return render(resolve(shipped, ["cern-team"]), tmp_path_factory.mktemp("bare"))


def test_a_bare_cern_team_has_docsite_indico_and_no_cmssw(bare):
    assert sorted(bare.composition.sources) == ["docsite.docsite", "indico.indico_live"]
    assert not any("cmssw" in sid for sid in bare.composition.sources)


def test_every_bare_default_source_has_its_input(shipped, bare):
    """Each default source either reads live with a non-empty scope or has a
    prepare step that writes its cache."""
    values = question_values(bare.composition.questions, {"archi_data_root": "/d"})
    steps = steps_for([shipped.packs[p] for p in ("docsite", "indico", "core-playbooks")], values)
    assert [s.label for s in steps] == ["docsite:records-cache"]
    urls = steps[0].command[-1].split(",")
    assert len(urls) == 2 and all(u.startswith("https://") for u in urls)

    # docsite reads the very file the step writes (not the downloader's own
    # default, <root>/data/docsite/records.json).
    docsite = yaml.safe_load(bare.composition.sources["docsite.docsite"])["docsite.docsite"]
    written = steps[0].command[steps[0].command.index("--docsite-output") + 1]
    assert docsite["params"]["records_path"].replace("${archi_data_root}", "/d") == written

    # indico: no answer needed; the source's own default event, a public
    # server, no category walk, no bound.
    assert values["indico_event_ids"] == "[]" and values["indico_category_ids"] == "[]"
    assert values["indico_base_url"] == "https://indico.cern.ch"
    indico = yaml.safe_load(bare.composition.sources["indico.indico_live"])["indico.indico_live"]
    assert indico["params"]["default_event_ids"] == ["1716863"]
    assert "max_events" not in indico["params"] and "cookie_file_env" not in indico["params"]


def test_cmssw_releases_stays_selectable(shipped, tmp_path):
    rendered = render(resolve(shipped, ["cern-team"], with_packs=["cmssw-releases"]), tmp_path)
    assert "cmssw-releases.cmssw_releases" in rendered.composition.sources


@pytest.mark.parametrize("events, categories, read_events, read_categories", [
    ("[]", "[]", ("1716863",), ()),                 # bare install: the default event
    ("[]", '["100"]', (), ("100",)),                # categories only: no default event
    ('["5"]', "[]", ("5",), ()),                    # events only: just those
    ('["5"]', '["100"]', ("5",), ("100",)),
])
def test_the_indico_default_event_yields_to_any_named_scope(
    bare, events, categories, read_events, read_categories
):
    from archi.sources.indico_live import IndicoLiveSource

    text = bare.composition.sources["indico.indico_live"]
    text = (text.replace("${indico_event_ids}", events)
                .replace("${indico_category_ids}", categories)
                .replace("${indico_base_url}", "https://indico.cern.ch"))
    params = yaml.safe_load(text)["indico.indico_live"]["params"]
    source = IndicoLiveSource(**{k: v for k, v in params.items() if k != "required"})
    assert (source.event_ids, source.category_ids) == (read_events, read_categories)


# ---------------------------------------------------------------------------
# a prepare step (a downloader) gets the instance's .env
# ---------------------------------------------------------------------------


def test_a_step_gets_the_instance_env_values_and_the_shell_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL_ONLY", "shell")
    monkeypatch.setenv("BOTH", "shell")
    monkeypatch.delenv("DOTENV_ONLY", raising=False)
    seen = {}

    def runner(cmd, **kw):
        seen.update(kw["env"])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    run_steps([_step("one")], tmp_path, echo=lambda _: None, runner=runner,
              env_values={"DOTENV_ONLY": "dotenv", "BOTH": "dotenv"})
    assert seen["DOTENV_ONLY"] == "dotenv"
    assert seen["BOTH"] == "shell"
    assert seen["SHELL_ONLY"] == "shell"
    assert seen["ARCHI_DATA_ROOT"] == str(tmp_path)


def test_add_passes_the_instance_env_to_the_prepare_steps(cern, tmp_path, monkeypatch):
    instance = tmp_path / "instance"
    render_and_lock(cern, instance, ["cern-team"], without_packs=["docsite"])
    deployment = _scaffold(instance, "cern-team", BASE_ANSWERS)
    (instance / ".env").write_text("# test\nCERN_JIRA_TOKEN=from-dotenv\n")
    seen = []

    def stop(steps, data_root, **kw):
        seen.append(kw.get("env_values"))
        raise PrepareError("stop here")

    monkeypatch.setattr("archi.bundles.prepare.run_steps", stop)
    with pytest.raises(LockError):
        add(instance, cern, "docsite", {}, deployment_dir=deployment, echo=lambda *_: None)
    assert seen == [{"CERN_JIRA_TOKEN": "from-dotenv"}]
