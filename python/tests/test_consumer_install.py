"""Consumer package producer tests; wheel fixtures are synthetic unit inputs."""

import json
import zipfile
from pathlib import Path

import pytest
import yaml
from okg.distributions import verify_distribution_package

from archi.install.configuration import SOURCE_MODES, prepare_configuration
from archi.install.distribution import build_prepared_package, verify_release_receipt
from archi.install.snapshots import digest

EXAMPLE_POLICY = (
    Path(__file__).resolve().parents[2]
    / "docs/cern-team-private-source-policy.example.yaml"
)


def release(tmp_path):
    path = tmp_path / "archi-3.0.0a1-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "archi-3.0.0a1.dist-info/METADATA", "Name: archi\nVersion: 3.0.0a1\n"
        )
        archive.writestr("archi/install/adapters.py", "# synthetic unit fixture\n")
    receipt = dict(
        schema="archi.install-release/v1",
        wheel_digest=digest(path.read_bytes()),
        package="archi",
        version="3.0.0a1",
        repository_url="https://github.com/archi-physics/archi",
        revision="a" * 40,
    )
    return path, receipt


def prepared(root, name="example"):
    from test_frozen_adapters import snapshot

    policy = yaml.safe_load(EXAMPLE_POLICY.read_text())
    return prepare_configuration(
        instance_name=name,
        instance_root=root,
        snapshot=snapshot(),
        source_policy=policy,
    )


def package(tmp_path, *, output="package", files=None, modules=None, **overrides):
    wheel, receipt = release(tmp_path)
    root = tmp_path / "instance"
    if files is None:
        files, prepared_modules = prepared(root)
        modules = prepared_modules if modules is None else modules
    arguments = dict(
        wheel=wheel,
        release_receipt=json.dumps(receipt).encode(),
        framework_digest="sha256:" + "b" * 64,
        instance_root=root,
        instance_name="example",
        modules=modules,
        configuration=files,
        source_modes=dict(SOURCE_MODES),
        output=tmp_path / output,
    )
    arguments.update(overrides)
    return build_prepared_package(**arguments)


def test_public_package_build_verifies_selected_configuration(tmp_path):
    files, modules = prepared(tmp_path / "instance")
    archive = package(tmp_path, files=files, modules=modules)
    verified = verify_distribution_package(archive)
    try:
        assert verified.manifest.provenance.revision == "a" * 40
        ids = {item.asset.id for item in verified.manifest.assets}
        assert "prepared-configuration" in ids
        assert len(ids) == len(files) + 1
    finally:
        verified.close_runtime_authority()


def test_package_build_is_deterministic(tmp_path):
    first = package(tmp_path, output="first")
    second = package(tmp_path, output="second")
    assert first.read_bytes() == second.read_bytes()


def test_package_refuses_configuration_prepared_for_another_target(tmp_path):
    files, modules = prepared(tmp_path / "other")
    with pytest.raises(ValueError, match="different target"):
        package(tmp_path, files=files, modules=modules)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"source_modes": {**SOURCE_MODES, "github_repo": "scope_complete"}},
            "three frozen readers",
        ),
        ({"source_modes": {"cmssw_releases": "release_new"}}, "three frozen readers"),
        ({"instance_name": "renamed"}, "another instance"),
        ({"modules": ["okg:person"]}, "another instance"),
    ],
)
def test_package_refuses_other_selection_name_or_modules(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        package(tmp_path, **overrides)


def test_package_refuses_cache_bytes_that_differ_from_pins(tmp_path):
    files, modules = prepared(tmp_path / "instance")
    files = dict(files, **{"snapshots/jira/meta.json": b"{}"})
    with pytest.raises(ValueError, match="pinned digests"):
        package(tmp_path, files=files, modules=modules)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wheel_digest", "sha256:" + "f" * 64),
        ("version", "9.0"),
        ("revision", "master"),
        ("repository_url", "https://example.org/other"),
    ],
)
def test_release_receipt_refuses_mismatch(tmp_path, field, value):
    wheel, receipt = release(tmp_path)
    receipt[field] = value
    with pytest.raises(ValueError):
        verify_release_receipt(wheel, json.dumps(receipt).encode())


def test_manifest_rejects_runtime_receipt_collision(tmp_path):
    files, modules = prepared(tmp_path / "instance")
    files = dict(files, **{"install.json": b"{}"})
    with pytest.raises(ValueError, match="runtime state destination"):
        package(tmp_path, files=files, modules=modules)


def test_oversized_compressed_metadata_refused(tmp_path):
    wheel, receipt = release(tmp_path)
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("archi-3.0.0a1.dist-info/METADATA", b"x" * (1024 * 1024 + 1))
        archive.writestr("archi/install/adapters.py", b"")
    receipt["wheel_digest"] = digest(wheel.read_bytes())
    with pytest.raises(ValueError, match="oversized"):
        verify_release_receipt(wheel, json.dumps(receipt).encode())


def test_consumer_configuration_contains_complete_frozen_bindings(tmp_path):
    import yaml
    from test_frozen_adapters import snapshot

    from archi.install.configuration import SOURCE_MODES, prepare_configuration

    names = {
        "cmssw_releases",
        "docsite_snapshot",
        "jira_snapshot",
        "github_repo",
        "gitlab_repo",
    }
    policy = yaml.safe_load(EXAMPLE_POLICY.read_text())
    files, modules = prepare_configuration(
        instance_name="example",
        instance_root=tmp_path / "instance",
        snapshot=snapshot(),
        source_policy=policy,
    )
    registry = yaml.safe_load(files["source_registry.yaml"])["sources"]
    assert set(registry) == names
    assert modules == [
        "okg:document_starter",
        "okg:person",
        "okg:extraction",
        "okg:git_graph",
    ]
    assert "invariants.yaml" in files and "skills/chat-system-prompt.md" in files
    assert "schemas/sources.yaml" in files
    for name in SOURCE_MODES:
        entry = registry[name]
        assert entry["module"] == "archi.install.adapters"
        assert entry["params"]["configuration_root"] == str(
            tmp_path / "instance/configuration"
        )
        for path, pin in entry["params"]["snapshot_digests"].items():
            assert digest(files[path]) == pin
        scope = entry["admission_policy"]["authority_scope"]
        assert entry["params"]["snapshot_digests"] == scope["snapshot_digests"]
        assert entry["params"].get("record_allowlist") == scope.get("record_allowlist")
        assert "credential_refs" not in entry
    assert registry["github_repo"]["params"]["url"] == ""
    assert registry["github_repo"]["required_for_baseline"] is False


def test_documented_private_policy_example_prepares_configuration(tmp_path):
    """The operator example stays usable with the producer it documents."""
    from pathlib import Path

    import yaml
    from test_frozen_adapters import snapshot

    from archi.install.configuration import prepare_configuration

    example = (
        Path(__file__).resolve().parents[2]
        / "docs/cern-team-private-source-policy.example.yaml"
    )
    policy = yaml.safe_load(example.read_text())
    files, _ = prepare_configuration(
        instance_name="example",
        instance_root=tmp_path / "instance",
        snapshot=snapshot(),
        source_policy=policy,
    )
    registry = yaml.safe_load(files["source_registry.yaml"])["sources"]
    for name, entry in registry.items():
        declared = entry["source_policy"]
        assert declared == policy["sources"][name]["source_policy"]
        assert "redaction" in declared["privacy_obligations"]
        assert declared["store_raw"] is False and declared["live_call_allowed"] is False


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda policy: policy.pop("retention"), "lacks retention"),
        (
            lambda policy: policy.update(privacy_obligations=["audit", "no_export"]),
            "redaction",
        ),
        (lambda policy: policy.update(store_raw=True), "redaction"),
        (lambda policy: policy.update(live_call_allowed=True), "live calls"),
    ],
)
def test_source_policy_requires_every_installer_field(tmp_path, change, match):
    from test_frozen_adapters import snapshot

    policy = yaml.safe_load(EXAMPLE_POLICY.read_text())
    change(policy["sources"]["jira_snapshot"]["source_policy"])
    with pytest.raises(ValueError, match=match):
        prepare_configuration(
            instance_name="example",
            instance_root=tmp_path / "instance",
            snapshot=snapshot(),
            source_policy=policy,
        )
