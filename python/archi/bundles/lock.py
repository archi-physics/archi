"""archi.lock, `archi add` and `archi diff` (ADR 0002 section 3.3).

`archi render` writes `<instance>/archi.lock` beside `profiles/`. It records
what was selected and exactly what was rendered from it:

- the selection (bundles, --with, --without) and the okg range;
- each bundle and pack: version, provider and a digest of its files;
- a digest of every file in the rendered profile;
- what the renderer composed that okg does not read: freshness SLOs, live
  tools, action classes, and which pack owns each source;
- the installed deployment, once one is recorded.

`archi diff` compares three things against it: the rendered profile on disk,
the packs and bundles installed now, and the deployment's source registry
against what the rendered profile produces for the deployment's own answers.
`archi add <pack>` re-renders with one more pack, adds that pack's entries
to the installed deployment, and runs the okg steps the quickstart used to
list as manual ones: commit, ownership claim, catalog load, ingest, publish.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from archi.bundles.compose import digest
from archi.bundles.discovery import Catalog
from archi.bundles.manifest import load_unique_yaml
from archi.bundles.render import PROFILES_DIR, Rendered
from archi.bundles.resolve import Resolution, resolve
from archi.bundles.versions import canonical

LOCK_FILE = "archi.lock"
LOCK_VERSION = 1
HEADER = (
    "# archi.lock: what `archi render` selected and wrote. Commit it with the\n"
    "# instance. `archi diff` reports drift from it; do not edit it by hand.\n"
)


class LockError(Exception):
    """The instance has no usable lock, or an add cannot proceed."""


def tree_digest(directory: Path) -> str:
    """One digest over every file under `directory` (symlinks followed)."""
    h = hashlib.sha256()
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        h.update(path.relative_to(directory).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + h.hexdigest()


def _okg_version() -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("okg")
    except PackageNotFoundError:
        return None


def _archi_version() -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version

    for name in ("archi-kb", "archi"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


def build_lock(
    resolution: Resolution,
    rendered: Rendered,
    instance_dir: Path,
    catalog: Catalog,
    deployment: Optional[Path] = None,
) -> dict[str, Any]:
    instance_dir = Path(instance_dir)
    comp = rendered.composition
    profile_rel = rendered.profile_dir.relative_to(instance_dir).as_posix()
    lock: dict[str, Any] = {
        "lock_version": LOCK_VERSION,
        "archi": _archi_version(),
        "selection": {
            "bundles": list(resolution.given),
            "with": list(resolution.with_packs),
            "without": list(resolution.without_packs),
        },
        "okg": {"range": canonical(resolution.okg_range), "rendered_with": _okg_version()},
        "bundles": {
            b.name: {
                "version": b.version,
                "provider": catalog.providers.get(f"bundle:{b.name}", ""),
                "digest": tree_digest(b.dir),
            }
            for b in resolution.bundles
        },
        "packs": {
            p.name: {
                "version": p.version,
                "kind": p.kind,
                "provider": catalog.providers.get(f"pack:{p.name}", ""),
                "digest": tree_digest(p.dir),
            }
            for p in resolution.packs
        },
        "profile": {
            "name": rendered.name,
            "dir": profile_rel,
            "files": {
                p.relative_to(rendered.profile_dir).as_posix(): digest(p.read_bytes())
                for p in sorted(rendered.profile_dir.rglob("*"))
                if p.is_file()
            },
        },
        "composed": {
            "sources": {
                key.split(":", 1)[1]: owners[0]
                for key, owners in comp.owners.items()
                if key.startswith("source:")
            },
            "freshness": comp.freshness,
            "live_tools": comp.live_tools,
            "action_classes": comp.action_classes,
            "automations": comp.automations,
        },
        "deployment": str(deployment) if deployment is not None else None,
    }
    return lock


def write_lock(instance_dir: Path, lock: Mapping[str, Any]) -> Path:
    path = Path(instance_dir) / LOCK_FILE
    path.write_text(HEADER + yaml.safe_dump(dict(lock), sort_keys=False), encoding="utf-8")
    return path


def read_lock(instance_dir: Path) -> dict[str, Any]:
    path = Path(instance_dir) / LOCK_FILE
    if not path.is_file():
        raise LockError(f"{path} does not exist; run `archi render` in this instance first")
    try:
        lock = load_unique_yaml(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LockError(f"{path}: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("lock_version") != LOCK_VERSION:
        raise LockError(f"{path}: not an archi.lock version {LOCK_VERSION}")
    return lock


def render_and_lock(
    catalog: Catalog,
    instance_dir: Path,
    bundles: Iterable[str],
    with_packs: Iterable[str] = (),
    without_packs: Iterable[str] = (),
    *,
    name: Optional[str] = None,
    deployment: Optional[Path] = None,
) -> tuple[Resolution, Rendered, dict[str, Any]]:
    from archi.bundles.render import render

    resolution = resolve(catalog, bundles, with_packs, without_packs)
    rendered = render(resolution, instance_dir, name=name)
    lock = build_lock(resolution, rendered, instance_dir, catalog, deployment)
    write_lock(instance_dir, lock)
    return resolution, rendered, lock


# ---------------------------------------------------------------------------
# The deployment side
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    return load_unique_yaml(path.read_text(encoding="utf-8"))


def _dump_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def deployment_answers(deployment_dir: Path) -> dict[str, Any]:
    """The init answers okg recorded in deployment.yaml at install time."""
    manifest = _load_yaml(Path(deployment_dir) / "deployment.yaml")
    answers = dict((manifest.get("profile") or {}).get("init_answers") or {})
    answers.pop("HOME", None)  # okg's own environment answer, not a profile question
    return answers


def expected_scaffold(
    instance_dir: Path, profile_name: str, deployment_dir: Path, answers: Mapping[str, Any]
):
    """okg's plan for the rendered profile with this deployment's answers."""
    from okg.substrate.deployment_bootstrap.profile_init import build_plan, discover_profile

    deployment_dir = Path(deployment_dir)
    # build_plan writes nothing; the root only has to be a place no deployment is.
    with tempfile.TemporaryDirectory(prefix="archi-expected-") as scratch:
        return build_plan(
            profile=discover_profile(profile_name, profiles_root=Path(instance_dir) / PROFILES_DIR),
            deployment_name=deployment_dir.name,
            deployments_root=Path(scratch),
            raw_answers=dict(answers),
        )


def _registry_entries(plan) -> dict[str, Any]:
    from okg.substrate.deployment_bootstrap.profile_init import source_default_entries

    return source_default_entries(plan.profile, plan.source_defaults)


# ---------------------------------------------------------------------------
# archi diff
# ---------------------------------------------------------------------------


@dataclass
class Drift:
    """Every difference `archi diff` found, by area."""

    profile: list[str] = field(default_factory=list)
    upstream: list[str] = field(default_factory=list)
    deployment: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.profile or self.upstream or self.deployment)

    def as_dict(self) -> dict[str, list[str]]:
        return {"profile": self.profile, "upstream": self.upstream, "deployment": self.deployment}


def diff(
    instance_dir: Path, catalog: Catalog, deployment_dir: Optional[Path] = None
) -> Drift:
    instance_dir = Path(instance_dir)
    lock = read_lock(instance_dir)
    drift = Drift()

    profile_dir = instance_dir / lock["profile"]["dir"]
    recorded = lock["profile"]["files"]
    on_disk = {
        p.relative_to(profile_dir).as_posix(): digest(p.read_bytes())
        for p in sorted(profile_dir.rglob("*"))
        if p.is_file()
    } if profile_dir.is_dir() else {}
    for rel in sorted(set(recorded) | set(on_disk)):
        if rel not in on_disk:
            drift.profile.append(f"{lock['profile']['dir']}/{rel}: missing")
        elif rel not in recorded:
            drift.profile.append(f"{lock['profile']['dir']}/{rel}: not rendered by archi")
        elif on_disk[rel] != recorded[rel]:
            drift.profile.append(f"{lock['profile']['dir']}/{rel}: edited since render")

    for kind, items, current in (
        ("pack", lock["packs"], catalog.packs),
        ("bundle", lock["bundles"], catalog.bundles),
    ):
        for name, entry in items.items():
            now = current.get(name)
            if now is None:
                drift.upstream.append(f"{kind} {name}: no longer installed")
                continue
            if now.version != entry["version"]:
                drift.upstream.append(
                    f"{kind} {name}: version {entry['version']} locked, {now.version} installed"
                )
            elif tree_digest(now.dir) != entry["digest"]:
                drift.upstream.append(f"{kind} {name}: content changed at version {now.version}")

    deployment_dir = deployment_dir or (Path(lock["deployment"]) if lock.get("deployment") else None)
    if deployment_dir is not None:
        if not (Path(deployment_dir) / "source_registry.yaml").is_file():
            drift.deployment.append(f"{deployment_dir}: no source_registry.yaml")
            return drift
        plan = expected_scaffold(
            instance_dir, lock["profile"]["name"], deployment_dir, deployment_answers(deployment_dir)
        )
        expected = _registry_entries(plan)
        live = (_load_yaml(Path(deployment_dir) / "source_registry.yaml") or {}).get("sources") or {}
        for sid in sorted(set(expected) | set(live)):
            if sid not in live:
                drift.deployment.append(f"source {sid}: in the profile, not in the registry")
            elif sid not in expected:
                drift.deployment.append(f"source {sid}: in the registry, not in the profile")
            elif live[sid] != expected[sid]:
                keys = sorted(
                    k for k in set(live[sid]) | set(expected[sid])
                    if live[sid].get(k) != expected[sid].get(k)
                )
                drift.deployment.append(f"source {sid}: registry differs from the profile in {keys}")
    return drift


# ---------------------------------------------------------------------------
# archi add
# ---------------------------------------------------------------------------


def _okg_executable() -> str:
    beside = Path(sys.executable).with_name("okg")
    if beside.is_file():
        return str(beside)
    found = shutil.which("okg")
    if found is None:
        raise LockError("okg is not installed in this environment")
    return found


def _instance_env_values(instance_dir: Path) -> dict[str, str]:
    """The values in the instance's `.env`, or none when it has no `.env` (a bare profile)."""
    from archi.stack.instance import ENV_FILE, read_env

    path = Path(instance_dir) / ENV_FILE
    return read_env(path) if path.is_file() else {}


@dataclass
class AddResult:
    pack: str
    added_sources: list[str]
    commands: list[list[str]]
    #: Every pack this add installed: the pack, what it requires, its companions.
    added_packs: list[str] = field(default_factory=list)


def _merge_into_deployment(plan, deployment_dir: Path, new_sources: list[str],
                           new_answers: Mapping[str, Any], echo=print) -> list[str]:
    """Add the new packs' parts to an installed deployment; return changed paths.

    Only additions: entries, files and list items the deployment lacks. A path
    the deployment already has with different content is refused, because the
    instance changed it after install and archi will not overwrite that.
    """
    from okg.substrate.deployment_bootstrap.profile_init import planned_writes

    changed: list[str] = []
    fresh = {w.path.relative_to(plan.deployment_dir).as_posix(): w for w in planned_writes(plan)}

    registry_path = deployment_dir / "source_registry.yaml"
    registry = _load_yaml(registry_path) or {}
    sources = registry.setdefault("sources", {})
    expected = _registry_entries(plan)
    for sid in new_sources:
        if sid in sources:
            raise LockError(f"source {sid} is already in {registry_path}")
        sources[sid] = expected[sid]
    _dump_yaml(registry_path, registry)
    changed.append("source_registry.yaml")

    for rel, write in fresh.items():
        if rel in ("deployment.yaml", "source_registry.yaml", "invariants.yaml",
                   "skills/skill-triggers.yaml") or rel.endswith("_base.yaml"):
            continue
        target = deployment_dir / rel
        data = write.data if write.data is not None else write.text.encode("utf-8")
        if target.exists():
            if target.read_bytes() != data and rel == "skills/chat-system-prompt.md":
                echo(f"note: {target} left as is; the new selection's prompt differs "
                     f"(see {plan.profile.profile_dir}/skills/chat-system-prompt.md)")
            elif target.read_bytes() != data:
                raise LockError(
                    f"{target} differs from what the new selection renders; archi add only "
                    "adds files, so reconcile it by hand first"
                )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        changed.append(rel)

    if plan.invariants_text:
        inv_path = deployment_dir / "invariants.yaml"
        live = (_load_yaml(inv_path) if inv_path.exists() else None) or {"invariants": []}
        names = {i["name"] for i in live.get("invariants", [])}
        added = [i for i in yaml.safe_load(plan.invariants_text)["invariants"] if i["name"] not in names]
        if added:
            live.setdefault("invariants", []).extend(added)
            _dump_yaml(inv_path, live)
            changed.append("invariants.yaml")

    if "skills/skill-triggers.yaml" in fresh:
        trig_path = deployment_dir / "skills" / "skill-triggers.yaml"
        new = yaml.safe_load(fresh["skills/skill-triggers.yaml"].text or fresh["skills/skill-triggers.yaml"].data)
        live = (_load_yaml(trig_path) if trig_path.exists() else None) or {"version": new["version"], "intents": {}}
        before = dict(live.get("intents") or {})
        for intent, value in new["intents"].items():
            live.setdefault("intents", {}).setdefault(intent, value)
        if live["intents"] != before:
            trig_path.parent.mkdir(parents=True, exist_ok=True)
            _dump_yaml(trig_path, live)
            changed.append("skills/skill-triggers.yaml")

    manifest_path = deployment_dir / "deployment.yaml"
    manifest = _load_yaml(manifest_path)
    for module in plan.modules:
        if module not in manifest.setdefault("modules", []):
            manifest["modules"].append(module)
    subtypes = (
        plan.deployment_defaults.get("search", {}).get("profiles", {}).get("content", {}).get("subtypes", [])
    )
    live_subtypes = (
        manifest.setdefault("search", {}).setdefault("profiles", {})
        .setdefault("content", {}).setdefault("subtypes", [])
    )
    for subtype in subtypes:
        if subtype not in live_subtypes:
            live_subtypes.append(subtype)
    answers = manifest.setdefault("profile", {}).setdefault("init_answers", {})
    answers.update(new_answers)
    _dump_yaml(manifest_path, manifest)
    changed.append("deployment.yaml")
    return changed


def _git_commit(deployment_dir: Path, message: str, instance_dir: Optional[Path] = None) -> Optional[list[str]]:
    """Commit archi's own paths: the deployment directory and, when it sits in the same repository
    as the instance, archi.lock and profiles/ (okg refuses to publish while they are uncommitted:
    "repository provenance inputs are not committed"). The commit is limited to those paths, so
    other staged work stays staged and uncommitted. Returns None when there is no repository or
    nothing changed."""
    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(deployment_dir), *args], capture_output=True, text=True,
                              check=check)

    if git("rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        return None
    top = Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
    candidates = [Path(deployment_dir).resolve()]
    if instance_dir is not None:
        instance = Path(instance_dir).resolve()
        candidates += [instance / name for name in ("archi.lock", "profiles")]
    # pathspecs relative to the repository's top level, so a nested instance is handled the same way
    paths = [str(p.relative_to(top)) for p in candidates
             if p.exists() and (p == top or top in p.parents)]
    if not paths:
        return None
    pathspec = ["--", *paths]
    git("-C", str(top), "add", "-A", *pathspec)
    if git("-C", str(top), "diff", "--cached", "--quiet", *pathspec, check=False).returncode == 0:
        return None  # nothing changed in archi's paths
    from archi.cli.init import _git_identity_env  # lazy: archi.cli imports this module

    cmd = ["git", "-C", str(top), "commit", "-m", message, *pathspec]
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=_git_identity_env())
    return cmd


def add(
    instance_dir: Path,
    catalog: Catalog,
    pack: str,
    answers: Mapping[str, Any],
    *,
    deployment_dir: Optional[Path] = None,
    run_okg: bool = True,
    echo=print,
    without_extra: Iterable[str] = (),
) -> AddResult:
    """Add `pack` (and whatever it requires, and its bundle companions) to an
    installed instance. `without` leaves out companions `pack` would bring; it
    is recorded in the lock's selection like init's --without."""
    instance_dir = Path(instance_dir)
    lock = read_lock(instance_dir)
    deployment_dir = Path(deployment_dir or lock.get("deployment") or "")
    if not (deployment_dir / "deployment.yaml").is_file():
        raise LockError(
            "no installed deployment recorded in archi.lock; pass --deployment DIR "
            "(the okg deployment directory holding deployment.yaml)"
        )
    deployment_dir = deployment_dir.resolve()
    if pack in lock["packs"]:
        raise LockError(f"pack {pack!r} is already in this instance")

    sel = lock["selection"]
    # The pack's requirements come along when the instance lacks them;
    # resolve() refuses a --with pack whose requirement is not selected.
    needed: list[str] = []

    def visit(name: str) -> None:
        if name not in catalog.packs:
            return  # resolve() reports the unknown pack by name
        for dep in catalog.packs[name].requires_packs:
            visit(dep)
        if name not in needed and name not in lock["packs"]:
            needed.append(name)

    visit(pack)
    with_packs = [p for p in sel["with"] if p not in needed] + needed
    without = [p for p in sel["without"] if p != pack]
    without += [p for p in without_extra if p not in without]
    resolution, rendered, new_lock = render_and_lock(
        catalog, instance_dir, sel["bundles"], with_packs, without,
        name=lock["profile"]["name"], deployment=deployment_dir,
    )
    added_packs = [p for p in resolution.pack_names if p not in lock["packs"]]
    for note in resolution.notes:
        if "comes with" in note or note.startswith("--without"):
            echo(f"note: {note}")
    echo(f"adding packs: {', '.join(added_packs)}")
    new_sources = [
        sid for sid, owner in new_lock["composed"]["sources"].items() if owner in added_packs
    ]
    all_answers = deployment_answers(deployment_dir)
    all_answers.update(answers)
    plan = expected_scaffold(instance_dir, rendered.name, deployment_dir, all_answers)
    new_answers = {k: v for k, v in plan.answers.items() if k not in deployment_answers(deployment_dir)}
    new_answers.pop("HOME", None)
    # The instance's `.env` (generated secrets and source credentials) is added
    # to the environment of every step that ingests; a value exported in the
    # shell wins.
    env_values = _instance_env_values(instance_dir)
    if run_okg:
        # Before the deployment changes: a failed step leaves it as it was.
        from archi.bundles.prepare import PrepareError, run_steps, steps_for

        try:
            run_steps(
                steps_for([catalog.packs[p] for p in added_packs], plan.answers),
                Path(plan.answers["archi_data_root"]), echo=echo, env_values=env_values,
            )
        except PrepareError as exc:
            raise LockError(f"preparing the new pack failed: {exc}") from exc
    changed = _merge_into_deployment(plan, deployment_dir, new_sources, new_answers, echo)
    echo(f"added to {deployment_dir}: {', '.join(changed)}")

    commands: list[list[str]] = []
    extra = [p for p in added_packs if p != pack]
    message = f"archi add {pack}" + (f" (with {', '.join(extra)})" if extra else "")
    commit = _git_commit(deployment_dir, message, instance_dir)
    if commit:
        commands.append(commit)
        echo("committed the deployment change")
    if run_okg:
        okg = _okg_executable()
        name = deployment_dir.name
        steps = [
            [okg, "catalog", "ownership", "claim", "--deployment", str(deployment_dir), "--json"],
            [okg, "catalog", "load", "--deployment", str(deployment_dir), "--apply", "--json"],
            [okg, "ingest", "--deployment", name, "--include", ",".join(new_sources)],
            [okg, "run", "--once", "--deployment", name, "--apply"],
        ]
        from archi.stack.instance import instance_env

        env = instance_env(env_values)
        env.setdefault("OKG_DEPLOYMENTS_DIR", str(deployment_dir.parent))
        for cmd in steps:
            echo("$ " + " ".join(cmd[1:] if cmd[0] == okg else cmd))
            result = subprocess.run(cmd, cwd=deployment_dir, env=env, capture_output=True, text=True)
            commands.append(cmd)
            if result.returncode != 0:
                raise LockError(
                    f"`okg {' '.join(cmd[1:])}` failed ({result.returncode}):\n"
                    + (result.stderr or result.stdout)[-3000:]
                )
    return AddResult(pack, new_sources, commands, added_packs)
