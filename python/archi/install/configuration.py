"""Compose the consumer's complete frozen configuration before packaging."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

from archi.paths import bundle_dir

from .snapshots import digest, prepare_caches, strict_json


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


SOURCE_MODES = {
    "cmssw_releases": "release_new",
    "docsite_snapshot": "scope_complete",
    "jira_snapshot": "reconcile",
}


#: Fields the framework installer's source-policy audit requires, plus the two
#: privacy fields its prepared-configuration check reads. A frozen private
#: instance must declare every one explicitly.
POLICY_FIELDS = frozenset(
    {
        "sensitivity",
        "data_classification",
        "credential_ref_policy",
        "live_call_allowed",
        "exportability",
        "retention",
        "provenance",
        "privacy_obligations",
        "store_raw",
        "pii_classes",
    }
)
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute(value, answers):
    if isinstance(value, str):

        def replacement(match):
            name = match.group(1)
            if name not in answers:
                raise ValueError(f"template placeholder {name!r} has no value")
            return answers[name]

        return _PLACEHOLDER.sub(replacement, value)
    if isinstance(value, list):
        return [_substitute(item, answers) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, answers) for key, item in value.items()}
    return value


def prepare_configuration(
    *,
    instance_name: str,
    instance_root: Path,
    snapshot: dict[str, bytes],
    source_policy: dict,
) -> tuple[dict[str, bytes], list[str]]:
    """Author bytes only; the generic installer owns validation and execution.

    All five registry policies must be explicit. Only the three frozen readers
    are selected for install; optional repository sources keep their normal
    disabled-by-selection status rather than acquiring implicit live authority.
    """
    bundle = bundle_dir("cern-team")
    cache = prepare_caches(snapshot)
    configuration_root = instance_root / "configuration"
    pins = {name: digest(body) for name, body in sorted(cache.items())}
    answers = dict(
        deployment_name=instance_name,
        archi_data_root=str(instance_root / "data"),
        cmssw_map_path=str(configuration_root / "snapshots/cmssw/releases.map"),
        cmssw_map_digest=pins["snapshots/cmssw/releases.map"],
        github_repo_name="github-repo",
        github_repo_url="",
        gitlab_repo_name="gitlab-repo",
        gitlab_repo_url="",
        chat_site_name="Team knowledge chat",
        chat_model="llama3.1:8b",
    )
    templates = (
        (
            "cmssw_releases_frozen.yaml.example",
            "cmssw_releases",
            "cmssw_releases",
            "FrozenCMSSWAdapter",
        ),
        (
            "docsite.yaml.example",
            "docsite",
            "docsite_snapshot",
            "FrozenDocumentationAdapter",
        ),
        ("jira.yaml.example", "jira", "jira_snapshot", "FrozenJiraAdapter"),
        ("github_repo.yaml", "github_repo", "github_repo", None),
        ("gitlab_repo.yaml", "gitlab_repo", "gitlab_repo", None),
    )
    sources = {}
    for filename, original, name, adapter in templates:
        source = _substitute(
            yaml.safe_load((bundle / "source-defaults" / filename).read_bytes())[
                original
            ],
            answers,
        )
        if adapter:
            ownership = (
                instance_name
                + "."
                + ("cmssw-releases" if name == "cmssw_releases" else name)
            )
            params = dict(
                configuration_root=str(configuration_root), snapshot_digests=pins
            )
            authority = dict(
                source_family="cern-team-bounded-snapshot",
                source_name=name,
                configuration_root=str(configuration_root),
                snapshot_digests=pins,
            )
            if name == "jira_snapshot":
                authority["record_allowlist"] = [
                    strict_json(snapshot["records.json"])[0]["key"]
                ]
            elif name == "docsite_snapshot":
                authority["record_allowlist"] = [
                    "https://fts3-docs.web.cern.ch/fts3-docs/"
                ]
            if "record_allowlist" in authority:
                # The adapter enforces the same allowlist before every run.
                params["record_allowlist"] = list(authority["record_allowlist"])
            source.update(
                module="archi.install.adapters",
                **{"class": adapter},
                ownership_id=ownership,
                required_for_baseline=True,
                params=params,
            )
            source.pop("credential_refs", None)
            source.pop("credential_aliases", None)
            source["admission_policy"].update(
                producer_id=ownership, authority_scope=authority
            )
        sources[name] = source
    if (
        not isinstance(source_policy, dict)
        or set(source_policy) != {"sources"}
        or not isinstance(source_policy["sources"], dict)
        or set(source_policy["sources"]) != set(sources)
    ):
        raise ValueError("explicit policy must cover exactly the five registry sources")
    for name, entry in source_policy["sources"].items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"source_policy"}
            or not isinstance(entry["source_policy"], dict)
        ):
            raise ValueError(
                "each explicit policy entry must contain only source_policy"
            )
        policy = entry["source_policy"]
        missing = sorted(POLICY_FIELDS - set(policy))
        if missing:
            raise ValueError(f"source policy for {name!r} lacks {', '.join(missing)}")
        obligations = policy["privacy_obligations"]
        if (
            policy["store_raw"] is not False
            or policy["live_call_allowed"] is not False
            or not isinstance(policy["pii_classes"], dict)
            or not isinstance(obligations, list)
            or "redaction" not in obligations
        ):
            raise ValueError(
                "frozen private configuration requires redaction with store_raw false "
                "and no live calls"
            )
        sources[name]["source_policy"] = copy.deepcopy(policy)
    modules = [
        "okg:" + name
        for name in yaml.safe_load((bundle / "modules.yaml").read_bytes())["modules"]
    ]
    deployment = _substitute(
        yaml.safe_load((bundle / "deployment-defaults.yaml").read_bytes()), answers
    )
    deployment.update(
        yaml.safe_load((bundle / "prepared-deployment.yaml").read_bytes())
    )
    deployment.update(
        name=instance_name,
        description="Consumer-owned immutable CERN knowledge instance",
        modules=modules,
        source_registry="source_registry.yaml",
        schema_dir="schemas",
        postgres={"dsn": "${OKG_INSTALL_DSN}"},
    )
    files = dict(cache)
    for name, value in (
        ("deployment.yaml", deployment),
        ("source_registry.yaml", {"sources": sources}),
        ("source-policy.yaml", source_policy),
    ):
        files[name] = yaml.dump(value, Dumper=_NoAliases, sort_keys=True).encode()
    files["invariants.yaml"] = (bundle / "invariants.yaml").read_bytes()
    schemas = Path(__file__).resolve().parents[1] / "schemas"
    for path in schemas.rglob("*.yaml"):
        files["schemas/" + path.relative_to(schemas).as_posix()] = path.read_bytes()
    # The bundle's skills link resolves to consumer-owned packaged assets.
    for path in (bundle / "skills").rglob("*"):
        if path.is_file():
            files["skills/" + path.relative_to(bundle / "skills").as_posix()] = (
                path.read_bytes()
            )
    return files, modules
