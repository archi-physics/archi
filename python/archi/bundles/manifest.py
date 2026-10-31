"""Load and validate pack.yaml and bundle.yaml (ADR 0002 sections 3.1-3.2).

Validation has two layers. The published JSON schemas in `schema/` fix the
shape: required keys, types, and no unknown keys. The checks here cover what
a schema cannot see: that every path a manifest names exists, that source
templates reference only questions that exist, that nothing carries a
credential value, and that a bundle can publish with nothing configured.

Every check reports a `Problem` instead of stopping at the first one, so
`archi bundles validate` prints the whole list in one run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from archi.bundles.duration import parse_duration

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
BASE_DIR = Path(__file__).resolve().parent / "base"

PACK_FILE = "pack.yaml"
BUNDLE_FILE = "bundle.yaml"

#: okg's init-question types (`okg.substrate.catalog.profile.INIT_QUESTION_TYPES`).
#: Spelled out so validation runs without okg importable; a test pins the two.
QUESTION_TYPES = ("slug", "path", "path_list", "orcid", "string", "integer")
QUESTION_KEYS = frozenset({"id", "prompt", "type", "required", "default", "description"})

#: Manifest keys that name a directory, and those that name a file.
PACK_DIR_KEYS = ("schemas", "sources", "playbooks")
PACK_FILE_KEYS = (
    "init_questions",
    "invariants",
    "skill_triggers",
    "prompt_fragment",
    "action_classes",
)

ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
NAME = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
SOURCE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
QUESTION_ID = re.compile(r"^[a-z][a-z0-9_]*$")
TEMPLATE_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
#: Keys whose value would be a secret if it were written literally.
CREDENTIAL_KEY = re.compile(r"(token|password|passwd|secret|api_?key|credential)", re.I)


@dataclass(frozen=True)
class Problem:
    """One validation failure, tied to the file it is about."""

    path: Path
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ManifestError(Exception):
    """A manifest failed validation. `problems` lists every failure."""

    def __init__(self, problems: Iterable[Problem]):
        self.problems = list(problems)
        super().__init__("\n".join(str(p) for p in self.problems))


@lru_cache(maxsize=None)
def schema(kind: str) -> dict[str, Any]:
    """The published JSON schema for `pack` or `bundle`."""
    return json.loads((SCHEMA_DIR / f"{kind}.schema.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def base_questions() -> tuple[dict[str, Any], ...]:
    """Init questions every rendered profile carries (base/questions.yaml)."""
    data = yaml.safe_load((BASE_DIR / "questions.yaml").read_text(encoding="utf-8"))
    return tuple(data["init_questions"])


def base_answer_ids() -> frozenset[str]:
    return frozenset(q["id"] for q in base_questions())


def _schema_problems(kind: str, data: Any, path: Path) -> list[Problem]:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema(kind))
    out = []
    for error in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        where = ".".join(str(p) for p in error.absolute_path) or "(top level)"
        out.append(Problem(path, f"{where}: {error.message}"))
    return out


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated mapping key instead of keeping the last."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_unique_yaml(text: str) -> Any:
    """yaml.safe_load that raises yaml.YAMLError on a duplicate mapping key."""
    return yaml.load(text, Loader=_UniqueKeyLoader)


def _read_yaml(path: Path, problems: list[Problem]) -> Any:
    try:
        return load_unique_yaml(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(Problem(path, "file is missing"))
    except yaml.YAMLError as exc:
        problems.append(Problem(path, f"YAML parse error: {exc}"))
    return None


@dataclass(frozen=True)
class Pack:
    """A loaded, validated pack. `dir` holds pack.yaml."""

    dir: Path
    manifest: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def kind(self) -> str:
        return self.manifest["kind"]

    @property
    def version(self) -> str:
        return self.manifest["version"]

    @property
    def needs_credentials(self) -> bool:
        return bool(self.manifest["needs_credentials"])

    @property
    def credentials(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("credentials", ()))

    @property
    def requires_modules(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("requires", {}).get("modules", ()))

    @property
    def requires_packs(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("requires", {}).get("packs", ()))

    @property
    def search_subtypes(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("search_subtypes", ()))

    @property
    def live_tools(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("live_tools", ()))

    @property
    def idempotent_tools(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("idempotent_tools", ()))

    @property
    def freshness(self) -> dict[str, dict[str, str]]:
        return dict(self.manifest.get("freshness", {}))

    def max_staleness_seconds(self) -> dict[str, int]:
        """Rendered source id -> declared `max_staleness`, in seconds."""
        return {sid: parse_duration(slo["max_staleness"]) for sid, slo in self.freshness.items()}

    @property
    def link_coverage(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(t) for t in self.manifest.get("link_coverage", ()))

    @property
    def prepare(self) -> tuple[dict[str, Any], ...]:
        """The pack's declared preparation steps (see archi.bundles.prepare)."""
        return tuple(dict(step) for step in self.manifest.get("prepare", ()))

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("files", ()))

    def path(self, key: str) -> Optional[Path]:
        """The path a manifest key names, or None when the key is absent."""
        value = self.manifest.get(key)
        return None if value is None else self.dir / value

    def sources(self) -> dict[str, Path]:
        """Source name -> template file, sorted by name."""
        directory = self.path("sources")
        if directory is None:
            return {}
        return {p.stem: p for p in sorted(directory.glob("*.yaml"))}

    def questions(self) -> list[dict[str, Any]]:
        path = self.path("init_questions")
        if path is None:
            return []
        return list((load_unique_yaml(path.read_text(encoding="utf-8")) or {}).get(
            "init_questions", []
        ))


@dataclass(frozen=True)
class Bundle:
    """A loaded bundle manifest. Pack references are checked by `validate_bundle`."""

    dir: Path
    manifest: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def version(self) -> str:
        return self.manifest["version"]

    @property
    def extends(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("extends", ()))

    @property
    def default_packs(self) -> tuple[str, ...]:
        return tuple(self.manifest["packs"].get("default", ()))

    @property
    def optional_packs(self) -> tuple[str, ...]:
        return tuple(self.manifest["packs"].get("optional", ()))

    @property
    def companions(self) -> dict[str, tuple[str, ...]]:
        """Pack -> packs that come on whenever it is selected (ADR 0002 section 3.3)."""
        return {k: tuple(v) for k, v in self.manifest["packs"].get("companions", {}).items()}

    @property
    def answers(self) -> dict[str, Any]:
        return dict(self.manifest.get("answers", {}))

    @property
    def chat(self) -> dict[str, Any]:
        return dict(self.manifest.get("chat", {}))

    @property
    def okg(self) -> str:
        return self.manifest["okg"]

    @property
    def automation_bounds(self) -> dict[str, str]:
        """Automation name -> the highest level this bundle allows it."""
        return {n: b["max_level"] for n, b in self.manifest.get("automations", {}).items()}


# ---------------------------------------------------------------------------
# Packs
# ---------------------------------------------------------------------------


def _credential_value_problems(value: Any, where: str, path: Path) -> list[Problem]:
    """A credential-named key must hold an env-var reference, never a value."""
    out: list[Problem] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_where = f"{where}.{key}" if where else str(key)
            if (
                isinstance(child, str)
                and CREDENTIAL_KEY.search(str(key))
                and child.strip()
                and not TEMPLATE_REF.fullmatch(child.strip())
                and not ENV_NAME.fullmatch(child.strip())
            ):
                out.append(
                    Problem(
                        path,
                        f"{child_where}: looks like a credential value; name an "
                        "environment variable instead (e.g. `${INDICO_TOKEN}`)",
                    )
                )
            else:
                out.extend(_credential_value_problems(child, child_where, path))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            out.extend(_credential_value_problems(child, f"{where}[{i}]", path))
    return out


def _output_edges(entry: Any) -> set[tuple[str, str, str]]:
    """(src_subtype, edge_type, dst_subtype) triples a source template declares."""
    try:
        edges = entry["admission_policy"]["output_signature"].get("edges") or []
    except (KeyError, TypeError, AttributeError):
        return set()
    out = set()
    for edge in edges:
        if isinstance(edge, Mapping) and {"src_subtype", "edge_type", "dst_subtype"} <= set(edge):
            out.add((edge["src_subtype"], edge["edge_type"], edge["dst_subtype"]))
    return out


def _template_refs(text: str) -> set[str]:
    return set(TEMPLATE_REF.findall(text))


def _question_problems(path: Path, data: Any) -> tuple[list[Problem], set[str]]:
    problems: list[Problem] = []
    ids: set[str] = set()
    if not isinstance(data, Mapping) or not isinstance(data.get("init_questions"), list):
        return [Problem(path, "must hold a top-level `init_questions:` list")], ids
    for i, q in enumerate(data["init_questions"]):
        where = f"init_questions[{i}]"
        if not isinstance(q, Mapping):
            problems.append(Problem(path, f"{where}: must be a mapping"))
            continue
        unknown = sorted(set(q) - QUESTION_KEYS)
        if unknown:
            problems.append(Problem(path, f"{where}: unknown keys {unknown}"))
        qid = q.get("id")
        if not isinstance(qid, str) or not QUESTION_ID.fullmatch(qid):
            problems.append(Problem(path, f"{where}.id: must match {QUESTION_ID.pattern}"))
            continue
        if qid in ids:
            problems.append(Problem(path, f"{where}.id: duplicate id {qid!r}"))
        ids.add(qid)
        if q.get("type", "string") not in QUESTION_TYPES:
            problems.append(
                Problem(path, f"{where}.type: {q.get('type')!r} is not one of {QUESTION_TYPES}")
            )
        if "required" in q and not isinstance(q["required"], bool):
            problems.append(Problem(path, f"{where}.required: must be true or false"))
        default = q.get("default")
        if (
            CREDENTIAL_KEY.search(qid)
            and isinstance(default, str)
            and default.strip()
            and not TEMPLATE_REF.fullmatch(default.strip())
        ):
            problems.append(
                Problem(path, f"{where}.default: a credential question may not ship a value")
            )
    return problems, ids


def validate_pack(
    directory: Path,
    *,
    packs: Optional[Mapping[str, "Pack"]] = None,
    defer_requires: bool = False,
) -> list[Problem]:
    """Every problem with the pack in `directory` (empty list when valid).

    An automation pack's mediators resolve against the live tools of the packs
    it requires. `packs` is the catalog to look them up in (any root); without
    it a mediator that needs a required pack's tool reports that the pack is
    not resolvable without a catalog. `defer_requires`
    skips that lookup, for the catalog to repeat once every pack is loaded.
    """
    directory = Path(directory)
    manifest_path = directory / PACK_FILE
    problems: list[Problem] = []
    data = _read_yaml(manifest_path, problems)
    if problems:
        return problems
    problems.extend(_schema_problems("pack", data, manifest_path))
    if problems:
        return problems

    if data["name"] != directory.name:
        problems.append(
            Problem(manifest_path, f"name {data['name']!r} must equal the directory name {directory.name!r}")
        )
    for key in PACK_DIR_KEYS:
        if key in data and not (directory / data[key]).is_dir():
            problems.append(Problem(manifest_path, f"{key}: directory {data[key]!r} is missing"))
    for key in PACK_FILE_KEYS:
        if key in data and not (directory / data[key]).is_file():
            problems.append(Problem(manifest_path, f"{key}: file {data[key]!r} is missing"))
    for rel in data.get("files", ()):
        if not (directory / rel).is_file():
            problems.append(Problem(manifest_path, f"files: {rel!r} is missing"))
    for tool in data.get("idempotent_tools", ()):
        if tool not in data.get("live_tools", ()):
            problems.append(Problem(manifest_path, f"idempotent_tools: {tool!r} is not one of this pack's live_tools"))
    # A context pack brings sources, or only a schema slice that other
    # producers write into through admission (ops-record).
    if data["kind"] == "context" and "sources" not in data and "schemas" not in data:
        problems.append(Problem(manifest_path, "a context pack must declare `sources:` or `schemas:`"))
    if problems:
        return problems

    pack = Pack(directory, data)
    question_ids: set[str] = set()
    if "init_questions" in data:
        qpath = pack.path("init_questions")
        qproblems, question_ids = _question_problems(qpath, _read_yaml(qpath, problems))
        problems.extend(qproblems)
    known_refs = base_answer_ids() | question_ids
    # A template may also name a question of a pack this one requires, by its
    # rendered id (`${git_code_repo_name}`); compose checks it exists.
    required_prefixes = tuple(
        dep.replace("-", "_") + "_" for dep in data.get("requires", {}).get("packs", ())
    )

    sources = pack.sources()
    declared_edges: set[tuple[str, str, str]] = set()
    if "sources" in data and data["kind"] == "context" and not sources:
        problems.append(Problem(manifest_path, f"sources: {data['sources']!r} holds no *.yaml template"))
    for source, template in sources.items():
        if not SOURCE_NAME.fullmatch(source):
            problems.append(Problem(template, f"source name {source!r} must match {SOURCE_NAME.pattern}"))
        text = template.read_text(encoding="utf-8")
        body = _read_yaml(template, problems)
        if not isinstance(body, Mapping) or list(body) != [source]:
            problems.append(
                Problem(template, f"must hold exactly one top-level key, {source!r} (the file name)")
            )
            continue
        for ref in sorted(_template_refs(text) - known_refs):
            if ref.startswith(required_prefixes):
                continue
            problems.append(
                Problem(template, f"references ${{{ref}}}, which is neither a base question, one of this pack's questions, nor a question of a pack it requires")
            )
        problems.extend(_credential_value_problems(body, "", template))
        declared_edges |= _output_edges(body[source])

    seen_steps: set[str] = set()
    for i, step in enumerate(data.get("prepare", ())):
        where = f"prepare[{i}]"
        if step["name"] in seen_steps:
            problems.append(Problem(manifest_path, f"{where}.name: duplicate step {step['name']!r}"))
        seen_steps.add(step["name"])
        for j, arg in enumerate(step.get("args", ())):
            for ref in sorted(_template_refs(arg) - known_refs):
                problems.append(
                    Problem(manifest_path, f"{where}.args[{j}]: references ${{{ref}}}, which is neither a base question nor one of this pack's questions")
                )

    for source_id, slo in data.get("freshness", {}).items():
        pack_name, _, source = source_id.partition(".")
        if pack_name != pack.name or source not in sources:
            problems.append(
                Problem(manifest_path, f"freshness: {source_id!r} is not one of this pack's sources ({pack.name}.<source>)")
            )
        try:
            parse_duration(slo["max_staleness"])
        except ValueError as exc:
            problems.append(Problem(manifest_path, f"freshness.{source_id}.max_staleness: {exc}"))
    # ADR 0002 section 6.1: every source of a context pack declares how stale
    # it may get, so `archi doctor` and context gates have a number to hold it to.
    if data["kind"] == "context":
        for source in sources:
            if f"{pack.name}.{source}" not in data.get("freshness", {}):
                problems.append(
                    Problem(manifest_path, f"freshness: source {pack.name}.{source} declares no max_staleness")
                )

    link_names: set[str] = set()
    for i, target in enumerate(data.get("link_coverage", ())):
        where = f"link_coverage[{i}]"
        if target["name"] in link_names:
            problems.append(Problem(manifest_path, f"{where}.name: duplicate target {target['name']!r}"))
        link_names.add(target["name"])
        if target.get("direction", "out") == "in":
            edge = (target["to"], target["edge"], target["from"])
        else:
            edge = (target["from"], target["edge"], target["to"])
        if edge not in declared_edges:
            problems.append(
                Problem(
                    manifest_path,
                    f"{where}: no source of this pack declares the edge "
                    f"{edge[0]} -[{edge[1]}]-> {edge[2]} in its output_signature",
                )
            )

    if "invariants" in data:
        ipath = pack.path("invariants")
        body = _read_yaml(ipath, problems)
        if not isinstance(body, Mapping) or not isinstance(body.get("invariants"), list):
            problems.append(Problem(ipath, "must hold a top-level `invariants:` list"))
        else:
            names = [i.get("name") for i in body["invariants"] if isinstance(i, Mapping)]
            if len(names) != len(body["invariants"]) or not all(isinstance(n, str) for n in names):
                problems.append(Problem(ipath, "every invariant must be a mapping with a `name`"))
            elif len(set(names)) != len(names):
                problems.append(Problem(ipath, "invariant names must be unique"))

    if "skill_triggers" in data:
        tpath = pack.path("skill_triggers")
        body = _read_yaml(tpath, problems)
        if not isinstance(body, Mapping) or not isinstance(body.get("intents"), Mapping):
            problems.append(Problem(tpath, "must hold `version:` and an `intents:` mapping"))

    if "action_classes" in data:
        apath = pack.path("action_classes")
        before = len(problems)
        body = _read_yaml(apath, problems)
        if len(problems) == before and (
            not isinstance(body, Mapping) or not isinstance(body.get("action_classes"), Mapping)
        ):
            problems.append(Problem(apath, "must hold an `action_classes:` mapping keyed by id"))

    from archi.bundles.automation import automation_problems

    problems.extend(automation_problems(directory, data, packs=packs, check_requires=not defer_requires))
    return problems


def load_pack(directory: Path, *, defer_requires: bool = False) -> Pack:
    """Load the pack in `directory`, raising ManifestError if it is invalid."""
    problems = validate_pack(directory, defer_requires=defer_requires)
    if problems:
        raise ManifestError(problems)
    return Pack(Path(directory), load_unique_yaml((Path(directory) / PACK_FILE).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def read_bundle(directory: Path) -> tuple[Optional[Bundle], list[Problem]]:
    """Shape-check one bundle.yaml. Pack references are not checked here."""
    directory = Path(directory)
    manifest_path = directory / BUNDLE_FILE
    problems: list[Problem] = []
    data = _read_yaml(manifest_path, problems)
    if problems:
        return None, problems
    problems.extend(_schema_problems("bundle", data, manifest_path))
    if problems:
        return None, problems
    if data["name"] != directory.name:
        problems.append(
            Problem(manifest_path, f"name {data['name']!r} must equal the directory name {directory.name!r}")
        )
    if "eval" in data and not (directory / data["eval"]).is_file():
        problems.append(Problem(manifest_path, f"eval: file {data['eval']!r} is missing"))
    overlap = sorted(set(data["packs"].get("default", ())) & set(data["packs"].get("optional", ())))
    if overlap:
        problems.append(Problem(manifest_path, f"packs listed as both default and optional: {overlap}"))
    from archi.ops.levels import settings_problems

    for message in settings_problems(data.get("automations"), "max_level", "automations"):
        problems.append(Problem(manifest_path, message))
    from archi.bundles.versions import OkgRangeError, parse_range

    try:
        parse_range(data["okg"])
    except OkgRangeError as exc:
        problems.append(Problem(manifest_path, f"okg: {exc}"))
    return Bundle(directory, data), problems


def extends_closure(
    name: str, bundles: Mapping[str, Bundle], *, _stack: tuple[str, ...] = ()
) -> list[str]:
    """`name` and every bundle it extends, parents first, each once.

    Raises KeyError for an unknown bundle and ValueError for a cycle.
    """
    if name in _stack:
        raise ValueError(" -> ".join(_stack + (name,)))
    bundle = bundles[name]
    out: list[str] = []
    for parent in bundle.extends:
        for item in extends_closure(parent, bundles, _stack=_stack + (name,)):
            if item not in out:
                out.append(item)
    out.append(name)
    return out


def validate_bundle(
    directory: Path,
    packs: Mapping[str, Pack],
    bundles: Mapping[str, Bundle],
) -> list[Problem]:
    """Every problem with the bundle in `directory`, checked against a catalog."""
    bundle, problems = read_bundle(directory)
    if bundle is None:
        return problems
    manifest_path = Path(directory) / BUNDLE_FILE
    catalog = dict(bundles)
    catalog[bundle.name] = bundle

    for parent in bundle.extends:
        if parent not in catalog:
            problems.append(Problem(manifest_path, f"extends: unknown bundle {parent!r}"))
    if problems:
        return problems
    try:
        chain = extends_closure(bundle.name, catalog)
    except ValueError as exc:
        return problems + [Problem(manifest_path, f"extends: cycle {exc}")]

    for group in ("default", "optional"):
        for name in bundle.manifest["packs"].get(group, ()):
            if name not in packs:
                problems.append(Problem(manifest_path, f"packs.{group}: unknown pack {name!r}"))
    for trigger, companions in bundle.companions.items():
        if trigger not in packs:
            problems.append(Problem(manifest_path, f"packs.companions: unknown pack {trigger!r}"))
        for name in companions:
            if name not in packs:
                problems.append(Problem(
                    manifest_path,
                    f"packs.companions.{trigger}: unknown companion pack {name!r}",
                ))
            elif name == trigger:
                problems.append(Problem(manifest_path, f"packs.companions.{trigger}: a pack cannot be its own companion"))
    for name in bundle.automation_bounds:
        if name in packs and packs[name].kind != "automation":
            problems.append(Problem(manifest_path, f"automations.{name}: pack {name!r} is not an automation pack"))
        elif name not in packs:
            problems.append(Problem(manifest_path, f"automations.{name}: unknown automation pack {name!r}"))
    if problems:
        return problems

    defaults = [p for b in chain for p in catalog[b].default_packs]
    # Only a pack with sources publishes anything on its own.
    credential_free = [
        p for p in defaults
        if packs[p].kind == "context" and not packs[p].needs_credentials and packs[p].sources()
    ]
    if not credential_free:
        problems.append(
            Problem(
                manifest_path,
                "no credential-free default context pack: a bundle with nothing "
                "configured must still publish (ADR 0002 section 3.3); add a "
                "default context pack with `needs_credentials: false`",
            )
        )
    return problems
