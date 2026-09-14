"""Compose the consumer's complete frozen configuration before packaging."""

from __future__ import annotations

import copy
from pathlib import Path
from string import Template

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


def _substitute(value, answers):
    if isinstance(value, str):
        return Template(value).substitute(answers)
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
            source.update(
                module="archi.install.adapters",
                **{"class": adapter},
                ownership_id=ownership,
                required_for_baseline=True,
                params=dict(
                    configuration_root=str(configuration_root), snapshot_digests=pins
                ),
            )
            source.pop("credential_refs", None)
            source.pop("credential_aliases", None)
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
        if (
            policy.get("store_raw") is not False
            or policy.get("live_call_allowed") is not False
            or not isinstance(policy.get("pii_classes"), dict)
        ):
            raise ValueError(
                "frozen private configuration requires explicit redaction and no live calls"
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
