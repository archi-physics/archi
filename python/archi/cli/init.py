"""`archi init <name> --bundle B`: an empty directory to a published instance.

What the quickstart asked an operator to do by hand, in order, and what
this does instead:

1. git-init the deployments directory with a baseline commit
   -> the instance directory is a git repository from its first command;
2. copy the profile to ./profiles/ and run okg from beside it
   -> `archi render` into the instance; okg always runs from the instance;
3. pick passwords and keep them consistent across shells
   -> every secret is generated once into .env (mode 0600, git-ignored);
4. pass an absolute --archi-data-root outside the deployment, because a
   relative one resolves against the working directory and `--force`
   deletes the deployment
   -> every path answer is made absolute; the data root defaults to
      ~/.local/share/archi/<name> and must lie outside the instance;
5. run the Postgres container with the preload flags and shm size
   -> the compose file's graph-db service;
6. watch `okg install` fail at publish because its own files are not
   committed, commit them, and publish by hand
   -> `okg install --no-publish`, commit, `okg provision`;
7. set role passwords after the migration
   -> `okg db role-passwords --apply`, into the instance's .secrets/.
8. with an automation pack selected, `runtime.enabled` is in deployment.yaml from the start and
   provision stops at its ingest stage (the DBOS schema does not exist yet; upstream ask O12)
   -> `okg runtime bootstrap`, then provision again (`provision`, below);
   and okg's lint blocks a runtime-enabled deployment that names no `postgres.database`, which
   `okg install` never writes
   -> the instance's database is declared right after the install (`declare_runtime_database`).
9. run each source's cache downloader by hand before the first ingest
   -> the selected packs' `prepare` steps, before anything is started.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import yaml

from archi.bundles import discovery
from archi.stack import compose as stack_compose
from archi.stack.instance import (
    DEPLOYMENTS_DIR,
    GITIGNORE,
    Instance,
    InstanceError,
    Ports,
    default_data_root,
    free_ports,
    generate_secrets,
    graph_dsn,
    write_env,
)
from archi.stack.okg import Okg, OkgError, answer_flags, okg_version, role_password

NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
#: Question ids whose answers are filesystem paths.
PATH_SUFFIXES = ("_root", "_path", "_dir")
DATA_ROOT_QUESTION = "archi_data_root"


class InitError(Exception):
    """`archi init` cannot proceed; the message says what to change."""


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _git_identity_env() -> dict[str, str]:
    """An author for archi's own commits when git has none configured (CI runners)."""
    env = dict(os.environ)
    probe = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        env.setdefault("GIT_AUTHOR_NAME", "archi")
        env.setdefault("GIT_AUTHOR_EMAIL", "archi@localhost")
        env.setdefault("GIT_COMMITTER_NAME", "archi")
        env.setdefault("GIT_COMMITTER_EMAIL", "archi@localhost")
    return env


def git(directory: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True,
        env=_git_identity_env(),
    )
    if result.returncode != 0:
        raise InitError(f"`git {' '.join(args)}` failed: {result.stderr.strip()}")
    return result


def commit_all(directory: Path, message: str) -> bool:
    """Commit everything not ignored; False when there was nothing to commit."""
    git(directory, "add", "-A", ".")
    if subprocess.run(["git", "-C", str(directory), "diff", "--cached", "--quiet"]).returncode == 0:
        return False
    git(directory, "commit", "-q", "-m", message)
    return True


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------


def is_path_question(question: Mapping[str, Any]) -> bool:
    return question.get("type") == "path" or str(question.get("id", "")).endswith(PATH_SUFFIXES)


def absolute_answers(
    questions: Sequence[Mapping[str, Any]],
    given: Mapping[str, str],
    data_root: Path,
    cwd: Path,
) -> dict[str, str]:
    """The answers archi passes to okg install, with every path made absolute.

    - `archi_data_root` is the instance's data root.
    - A path answer the operator gave is resolved against the directory they
      ran archi from (what they meant), never left for okg to resolve against
      its own working directory.
    - A relative path *default* is anchored on the data root: the bundle
      writes `data/twiki-snapshot` meaning "under the data root", but okg
      would resolve it against the directory it runs from, and it does not
      follow `archi_data_root` (quickstart step 8).
    - Defaults that are empty or start with `${` (okg expands those) stay.
    """
    by_id = {q["id"]: q for q in questions}
    unknown = sorted(set(given) - set(by_id))
    if unknown:
        raise InitError(
            f"--answer names no question of this profile: {', '.join(unknown)} "
            f"(questions: {', '.join(sorted(by_id))})"
        )
    root_default = str((by_id.get(DATA_ROOT_QUESTION) or {}).get("default") or "data")
    answers: dict[str, str] = {}
    for qid, question in by_id.items():
        if qid in ("deployment_name", "postgres_dsn"):
            continue
        if qid == DATA_ROOT_QUESTION:
            if qid in given:
                raise InitError("set the data root with --data-root, not --answer archi_data_root")
            answers[qid] = str(data_root)
            continue
        if qid in given:
            value = given[qid]
            if is_path_question(question) and value and not value.startswith("${"):
                value = str((cwd / value).resolve()) if not Path(value).is_absolute() else value
            answers[qid] = value
            continue
        default = question.get("default")
        if not is_path_question(question) or default is None:
            continue
        default = str(default)
        if not default or default.startswith("${") or Path(default).is_absolute():
            continue
        rel = Path(default)
        if rel.parts and rel.parts[0] == root_default:
            rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(".")
        answers[qid] = str((data_root / rel).resolve())
    return answers


def check_data_root(data_root: Path, instance_dir: Path) -> None:
    if data_root == instance_dir or data_root.is_relative_to(instance_dir):
        raise InitError(
            f"the data root {data_root} is inside the instance {instance_dir}; keep it "
            "outside: the instance is a git repository of configuration, and "
            "`okg install --force` deletes everything inside a deployment"
        )


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def render_profile(
    catalog: discovery.Catalog,
    instance_dir: Path,
    name: str,
    bundles: Sequence[str],
    with_packs: Sequence[str],
    without_packs: Sequence[str],
) -> tuple[str, Path]:
    """Render the profile into the instance; returns (profile name, profile dir).

    A composable bundle is rendered with its archi.lock; a legacy okg profile
    (profile.yaml, no bundle.yaml) is copied as `archi render` does.
    """
    legacy = [b for b in bundles if b not in catalog.bundles and b in catalog.legacy_profiles]
    if legacy:
        if len(bundles) > 1 or with_packs or without_packs:
            raise InitError(
                f"{legacy[0]!r} is a legacy okg profile; it cannot be combined with other "
                "bundles, --with or --without"
            )
        from archi.bundles.render import render_legacy

        out = render_legacy(catalog.legacy_profiles[legacy[0]], instance_dir)
        return legacy[0], out
    missing = [b for b in bundles if b not in catalog.bundles]
    if missing:
        raise InitError(
            f"no bundle named {', '.join(missing)}; `archi bundles list` shows what is installed"
        )
    from archi.bundles.lock import render_and_lock
    from archi.bundles.resolve import CompositionError

    try:
        _, rendered, _ = render_and_lock(
            catalog, instance_dir, bundles, with_packs, without_packs,
            deployment=instance_dir / DEPLOYMENTS_DIR / name,
        )
    except CompositionError as exc:
        raise InitError(f"({exc.code}) {exc}") from exc
    return rendered.name, rendered.profile_dir


def profile_questions(profile_dir: Path) -> list[dict[str, Any]]:
    profile = yaml.safe_load((profile_dir / "profile.yaml").read_text(encoding="utf-8"))
    return list(profile.get("init_questions") or [])


def prepare_packs(
    instance_dir: Path,
    catalog: discovery.Catalog,
    questions: Sequence[Mapping[str, Any]],
    answers: Mapping[str, str],
    data_root: Path,
    echo: Callable[[str], None],
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Run the selected packs' `prepare` steps, before okg's first ingest.

    A legacy okg profile has no archi.lock and no packs, so nothing runs.
    """
    from archi.bundles.lock import LockError, read_lock
    from archi.bundles.prepare import PrepareError, question_values, run_steps, steps_for

    try:
        names = list(read_lock(instance_dir)["packs"])
    except LockError:
        return
    try:
        steps = steps_for(
            [catalog.packs[n] for n in names], question_values(questions, answers)
        )
        run_steps(steps, data_root, echo=echo, runner=runner)
    except PrepareError as exc:
        raise InitError(
            f"{exc}\nThe instance at {instance_dir} was rendered but not installed; remove "
            "it and run `archi init` again, or pass --answer to change what the step reads."
        ) from exc


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def has_automations(instance_dir: Path) -> bool:
    """Does the rendered instance select an automation pack? (archi.lock `composed.automations`.)"""
    from archi.bundles.lock import read_lock

    try:
        return bool(read_lock(Path(instance_dir)).get("composed", {}).get("automations"))
    except Exception:  # no lock yet: nothing was rendered, so there is no automation
        return False


def ensure_ops_tables(instance: Instance, secrets: Mapping[str, str]) -> None:
    """Create the `archi_ops` tables (signals, halts, mediations, demotions) in the graph database.
    Idempotent. Without them every approved L2 run would end as an error at its mediator step."""
    from archi.ops.okg_io import OpsIO

    OpsIO(graph_dsn(instance, secrets), instance.name).ensure_tables()


def declare_runtime_database(deployment_dir: Path, database: str) -> bool:
    """Name the instance's database in deployment.yaml when the ops runtime is enabled.

    okg 0.2.0's lint blocks a runtime-enabled deployment that declares no `postgres.database`
    (`deployment.runtime.database_undeclared`): its worker must not run against whatever `${OKG_DSN}`
    happens to name. `okg install` writes `postgres: {dsn: ...}` itself, replacing any `postgres:` block
    a profile carries, so no profile can declare it; archi does, after the install. The name is the one
    `OKG_DSN` already points at (`Instance.database`), so okg's DSN/target agreement check holds.
    Returns whether the file changed. No deployment.yaml (nothing installed) changes nothing.
    """
    from archi.bundles.lock import _dump_yaml, _load_yaml

    path = Path(deployment_dir) / "deployment.yaml"
    if not path.is_file():
        return False
    manifest = _load_yaml(path) or {}
    if not (manifest.get("runtime") or {}).get("enabled"):
        return False
    postgres = manifest.get("postgres") or {}
    manifest["postgres"] = postgres
    declared = postgres.get("database")
    if declared == database:
        return False
    if isinstance(declared, str) and declared.strip():
        raise InitError(
            f"{path} declares postgres.database {declared!r}, but this instance's database is "
            f"{database!r}; fix one of them"
        )
    postgres["database"] = database
    _dump_yaml(path, manifest)
    return True


def provision(okg: Okg, name: str, *, runtime: bool, echo: Callable[[str], None]):
    """`okg provision`, and with an automation pack (runtime enabled from the start) the one hand step.

    With `runtime.enabled` in deployment.yaml, provision's ingest stage enqueues a DBOS workflow, which
    needs the DBOS schema (it does not exist yet: relation dbos.workflow_status) and a running worker
    (there is none during `archi init`); upstream ask O12. Every stage before ingest has finished, so:
    `okg runtime bootstrap`, then the stages that remain, with the ingest run in-process (`--inline`):
    `okg ingest --inline`, `okg run --once --apply` (the publish). Any other failure is raised as it is.
    """
    result = okg.run("provision", "--deployment", name, ok_codes=range(256) if runtime else (0, 8))
    if runtime and result.returncode not in (0, 8):
        text = (result.stderr or "") + (result.stdout or "")
        if "dbos" not in text.lower():
            raise OkgError(f"`okg provision --deployment {name}` failed ({result.returncode}):\n{text[-2500:]}", result)
        echo("provision stopped before the DBOS schema existed (the ops runtime is enabled); bootstrapping the "
             "runtime, then ingesting in-process and publishing")
        okg.run("runtime", "bootstrap", "--deployment", name)
        okg.run("ingest", "--deployment", name, "--inline")
        result = okg.run("run", "--once", "--deployment", name, "--apply", ok_codes=(0, 8))
    elif runtime:
        # provision got through (okg 0.2.0 does not stop on the missing schema the way `okg install` does);
        # the worker still needs the bootstrap, and it is idempotent.
        okg.run("runtime", "bootstrap", "--deployment", name)
    return result


def install_and_publish(
    instance: Instance, secrets: Mapping[str, str], okg: Okg, profile: str, resolved: Mapping[str, str],
    echo: Callable[[str], None] = print,
) -> dict[str, str]:
    """okg install, commit, provision (with the runtime bootstrap when an automation is selected), role
    passwords. Returns the secrets with the MCP role password filled in (also written to .env)."""
    name, instance_dir = instance.name, instance.dir
    okg.run(
        "install", "--profile", profile, "--deployment-name", name,
        # Stored verbatim; okg interpolates it from the environment at command
        # time, so no password is written into the deployment.
        "--postgres-dsn", "${OKG_DSN}",
        "--deployments-root", str(instance_dir / DEPLOYMENTS_DIR),
        "--non-interactive", "--no-publish",
        *answer_flags(resolved),
    )
    declare_runtime_database(instance_dir / DEPLOYMENTS_DIR / name, instance.database)
    # okg publishes only from committed configuration; the install just wrote it.
    stack_compose.write_stack_files(instance)
    commit_all(instance_dir, f"archi init: okg deployment {name}")
    runtime = has_automations(instance_dir)
    result = provision(okg, name, runtime=runtime, echo=echo)
    if runtime:
        ensure_ops_tables(instance, secrets)
    if result.returncode == 8:
        echo("note: published, but the publish held the publisher lock past okg's capacity "
             "bar (okg provision exit 8, capacity_warning); the generation is published")
    okg.run("db", "role-passwords", "--deployment", name, "--apply")
    secrets = dict(secrets)
    secrets["ARCHI_OKG_MCP_PASSWORD"] = role_password(instance.role_secrets_path, "okg_mcp")
    write_env(instance.env_path, secrets)
    return secrets


def init(
    name: str,
    *,
    bundles: Sequence[str],
    catalog: discovery.Catalog,
    directory: Optional[Path] = None,
    with_packs: Sequence[str] = (),
    without_packs: Sequence[str] = (),
    answers: Optional[Mapping[str, str]] = None,
    data_root: Optional[Path] = None,
    runtime: Optional[str] = None,
    admin_email: Optional[str] = None,
    run_okg: bool = True,
    echo: Callable[[str], None] = print,
    okg_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    compose_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    wait_db: Optional[Callable[[str], None]] = None,
    prepare_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Instance:
    if not NAME_RE.match(name):
        raise InitError(
            f"{name!r} is not a valid instance name: lowercase letters, digits, - and _, "
            "starting with a letter"
        )
    cwd = Path.cwd()
    instance_dir = (Path(directory) if directory else cwd / name).resolve()
    if instance_dir.exists() and any(instance_dir.iterdir()):
        raise InitError(
            f"{instance_dir} exists and is not empty; `archi init` starts from an empty "
            "directory (to start over: `archi down --purge --yes` there, then remove it)"
        )
    root = (Path(data_root).expanduser() if data_root else default_data_root(name)).resolve()
    check_data_root(root, instance_dir)
    version = okg_version()
    if version is None:
        raise InitError("okg is not installed in this Python environment; install the AKMON wheel")
    if run_okg:
        runtime = stack_compose.detect_runtime(runtime, runner=compose_runner)
    runtime = runtime or "docker"

    instance_dir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    echo(f"instance {instance_dir}  (data root {root})")
    git(instance_dir, "init", "-q", "-b", "main")
    (instance_dir / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    commit_all(instance_dir, "archi init: baseline")

    profile, profile_dir = render_profile(
        catalog, instance_dir, name, bundles, with_packs, without_packs
    )
    resolved = absolute_answers(profile_questions(profile_dir), dict(answers or {}), root, cwd)
    graph, mcp, webui = free_ports(3)
    instance = Instance(
        dir=instance_dir, name=name, bundles=list(bundles), data_root=root, runtime=runtime,
        ports=Ports(graph_db=graph, mcp=mcp, webui=webui),
        with_packs=list(with_packs), without_packs=list(without_packs), profile=profile,
        okg_version=version, admin_email=admin_email or f"admin@{name}.archi.invalid",
    )
    secrets = generate_secrets()
    # Filled in after the first publish, when okg has set the role passwords.
    secrets["ARCHI_OKG_MCP_PASSWORD"] = ""
    write_env(instance.env_path, secrets)
    instance.save()
    stack_compose.write_stack_files(instance)
    commit_all(instance_dir, f"archi init: render {profile} and the stack")
    echo(f"rendered profile {profile!r}; secrets in {instance.env_path} (git-ignored)")
    if not run_okg:
        return instance

    prepare_packs(instance_dir, catalog, profile_questions(profile_dir), resolved, root, echo,
                  prepare_runner)

    compose = stack_compose.Compose(instance, runner=compose_runner)
    echo("starting the graph database ...")
    compose.up("graph-db")
    dsn = graph_dsn(instance, secrets)
    if wait_db is None:
        stack_compose.wait_for("the graph database", stack_compose.graph_db_ready(dsn), 300, echo)
    else:
        wait_db(dsn)

    okg = Okg(instance, secrets, runner=okg_runner, echo=echo)
    install_and_publish(instance, secrets, okg, profile, resolved, echo)
    echo(f"published {name}. Next: `archi up` in {instance_dir}")
    return instance


def _packs_arg(values: Sequence[str]) -> list[str]:
    return [name for value in values for name in value.split(",") if name]


def cmd_init(args: argparse.Namespace) -> int:
    answers: dict[str, str] = {}
    for item in args.answer:
        key, sep, value = item.partition("=")
        if not sep:
            print(f"error: --answer {item!r} is not ID=VALUE", file=sys.stderr)
            return 1
        answers[key.replace("-", "_")] = value
    try:
        catalog = discovery.discover(args.pack_root, args.bundle_root)
        init(
            args.name, bundles=args.bundle, catalog=catalog, directory=args.dir,
            with_packs=_packs_arg(args.with_packs), without_packs=_packs_arg(args.without_packs),
            answers=answers, data_root=args.data_root, runtime=args.runtime,
            admin_email=args.admin_email, run_okg=not args.no_okg,
        )
    except (InitError, InstanceError, OkgError, stack_compose.ComposeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def add_parser(sub, roots) -> None:
    p = sub.add_parser("init", help="create an instance: git repo, secrets, profile, published graph")
    p.add_argument("name", help="instance and deployment name (lowercase slug)")
    p.add_argument("--bundle", action="append", default=None,
                   help="bundle to install (repeatable; default: cern-team)")
    p.add_argument("--with", dest="with_packs", action="append", default=[],
                   help="add packs (comma-separated or repeated)")
    p.add_argument("--without", dest="without_packs", action="append", default=[],
                   help="remove packs (comma-separated or repeated)")
    p.add_argument("--dir", type=Path, help="instance directory (default: ./<name>)")
    p.add_argument("--data-root", type=Path,
                   help="connector caches and checkouts, outside the instance "
                        "(default: ~/.local/share/archi/<name>)")
    p.add_argument("--answer", action="append", default=[], metavar="ID=VALUE",
                   help="answer a profile question (repeatable); relative paths are made absolute")
    p.add_argument("--runtime", choices=("docker", "podman"),
                   help="container runtime (default: the first that works)")
    p.add_argument("--admin-email", help="Open WebUI admin account (default: admin@<name>.archi.invalid)")
    p.add_argument("--no-okg", action="store_true",
                   help="write and commit the instance, but start nothing and run no okg command")
    roots(p)
    p.set_defaults(func=cmd_init, _default_bundle=True)


def finalize_args(args: argparse.Namespace) -> None:
    if getattr(args, "_default_bundle", False) and not args.bundle:
        args.bundle = ["cern-team"]
