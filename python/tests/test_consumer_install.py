"""Consumer package producer tests; wheel fixtures are synthetic unit inputs."""

import json
import zipfile

import pytest
from okg.distributions import verify_distribution_package

from archi.install.distribution import build_prepared_package, verify_release_receipt
from archi.install.snapshots import digest


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


def test_public_package_build_verifies_selected_configuration(tmp_path):
    wheel, receipt = release(tmp_path)
    configuration = {
        "deployment.yaml": b"name: example\nmodules: []\nrelease:\n  kind: packaged\n",
        "source_registry.yaml": b"sources:\n  frozen:\n    enabled: true\n",
    }
    archive = build_prepared_package(
        wheel=wheel,
        release_receipt=json.dumps(receipt).encode(),
        framework_digest="sha256:" + "b" * 64,
        instance_root=tmp_path / "instance",
        instance_name="example",
        modules=[],
        configuration=configuration,
        source_modes={"frozen": "release_new"},
        output=tmp_path / "package",
    )
    package = verify_distribution_package(archive)
    try:
        assert package.manifest.provenance.revision == "a" * 40
        assert {item.asset.id for item in package.manifest.assets} == {
            "file-0000",
            "file-0001",
            "prepared-configuration",
        }
    finally:
        package.close_runtime_authority()


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
    wheel, receipt = release(tmp_path)
    with pytest.raises(ValueError):
        build_prepared_package(
            wheel=wheel,
            release_receipt=json.dumps(receipt).encode(),
            framework_digest="sha256:" + "b" * 64,
            instance_root=tmp_path / "instance",
            instance_name="example",
            modules=[],
            configuration={"install.json": b"{}"},
            source_modes={"frozen": "release_new"},
            output=tmp_path / "package",
        )


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
    policy = {
        "sources": {
            name: {
                "source_policy": {
                    "store_raw": False,
                    "live_call_allowed": False,
                    "pii_classes": {},
                }
            }
            for name in names
        }
    }
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
        assert "credential_refs" not in entry
    assert registry["github_repo"]["params"]["url"] == ""
    assert registry["github_repo"]["required_for_baseline"] is False
