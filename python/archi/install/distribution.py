"""Build the consumer-owned immutable installation package from prepared bytes.

The caller authenticates the external build receipt. Wheel bytes and METADATA
are checked here; a Git revision is provenance supplied by that receipt, not a
claim inferred from arbitrary wheel metadata.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from okg.distributions import (
    AssetLocator,
    CompatibilityMetadata,
    DistributionObject,
    DistributionPackageManifest,
    DistributionPackageProvenance,
    DistributionPackageSource,
    DistributionRelease,
    OwnerQualifiedReference,
    PackagedAsset,
    PreparedConfigurationManifest,
    ProductBundle,
    TypedAsset,
    build_distribution_package,
    canonical_bundle_digest,
    canonical_release_digest,
)

from .snapshots import digest, read_regular, strict_json

TRANSLATION = "archi-consumer-frozen/v1"
OWNER = "archi"
REPOSITORY = "https://github.com/archi-physics/archi"


def verify_release_receipt(wheel: Path, receipt: bytes) -> dict:
    """Verify bytes against an externally authenticated, detached build receipt."""
    data = strict_json(receipt)
    fields = {
        "schema",
        "wheel_digest",
        "package",
        "version",
        "repository_url",
        "revision",
    }
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("release receipt fields differ")
    if (
        data["schema"] != "archi.install-release/v1"
        or data["package"] != "archi"
        or data["repository_url"] != REPOSITORY
        or not isinstance(data["revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", data["revision"])
    ):
        raise ValueError("unsupported consumer release receipt")
    body = read_regular(wheel, data["wheel_digest"], 128 * 1024 * 1024)
    import io

    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = archive.namelist()
        metadata = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(set(members)) != len(members) or len(metadata) != 1:
            raise ValueError("ambiguous wheel inventory")
        info = archive.getinfo(metadata[0])
        if (
            info.file_size > 1024 * 1024
            or info.is_dir()
            or len(Path(metadata[0]).parts) != 2
        ):
            raise ValueError("invalid or oversized wheel metadata")
        with archive.open(info) as stream:
            metadata_bytes = stream.read(1024 * 1024 + 1)
        if len(metadata_bytes) > 1024 * 1024:
            raise ValueError("oversized wheel metadata")
        parsed = BytesParser().parsebytes(metadata_bytes)
        if parsed.get_all("Name") != ["archi"] or parsed.get_all("Version") != [
            data["version"]
        ]:
            raise ValueError("wheel package/version differs from release receipt")
        if "archi/install/adapters.py" not in members:
            raise ValueError("selected release lacks consumer-owned frozen adapters")
    return data


def build_prepared_package(
    *,
    wheel: Path,
    release_receipt: bytes,
    framework_digest: str,
    instance_root: Path,
    instance_name: str,
    modules: list[str],
    configuration: dict[str, bytes],
    source_modes: dict[str, str],
    output: Path,
) -> Path:
    """Package complete consumer-authored configuration without host callbacks.

    No receipt or key is written into an existing installation. The root is an
    exact identity binding; generic installation independently validates all
    configuration paths, selections, plan/lock and installed wheel identities.
    """
    release = verify_release_receipt(wheel, release_receipt)
    file_refs = {
        name: f"{OWNER}:file-{index:04d}"
        for index, name in enumerate(sorted(configuration))
    }
    declaration = PreparedConfigurationManifest(
        json.dumps(
            {
                "schema": "okg.prepared_configuration/v1",
                "bundle_id": "cern-team",
                "instance_name": instance_name,
                "instance_root": str(instance_root),
                "framework_wheel": framework_digest,
                "consumer_wheel": release["wheel_digest"],
                "translation": TRANSLATION,
                "modules": modules,
                "source_modes": source_modes,
                "files": file_refs,
            },
            sort_keys=True,
        ).encode()
    )
    payloads = [
        (ref.split(":", 1)[1], "source-template", configuration[name])
        for name, ref in file_refs.items()
    ]
    payloads.append(("prepared-configuration", "release-policy", declaration.data))
    compatibility = CompatibilityMetadata(
        contract_versions=("v1",), capabilities=("declarative",)
    )
    assets, objects, sources = [], {}, {}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archi-package-") as temporary:
        root = Path(temporary)
        for name, kind, body in payloads:
            if not isinstance(body, bytes):
                raise ValueError("prepared configuration must contain bytes")
            pin = digest(body)
            path = root / name
            path.write_bytes(body)
            asset = TypedAsset(
                owner=OWNER,
                asset_id=name,
                kind=kind,
                plane=(
                    "operations-governance"
                    if kind == "release-policy"
                    else "ingestion-projection"
                ),
                version="v1",
                digest=pin,
                schema=f"okg.deployment_product.asset.{kind}/v1",
                merge_identity=name,
                merge_rule="refuse",
                compatibility={
                    "contract_versions": ["v1"],
                    "capabilities": ["declarative"],
                },
                provenance={
                    "source": "distribution",
                    "repository_url": REPOSITORY,
                    "revision": release["revision"],
                    "artifact_ref": f"{OWNER}:{name}",
                },
            )
            assets.append(
                PackagedAsset(
                    asset=asset,
                    locator=AssetLocator(object_digest=pin),
                    uncompressed_size=len(body),
                )
            )
            objects[pin] = DistributionObject(
                digest=pin, size=len(body), media_type="application/octet-stream"
            )
            sources[pin] = DistributionPackageSource(root=root, path=path)
        refs = tuple(
            OwnerQualifiedReference(
                owner=OWNER,
                id=item.asset.id,
                kind=item.asset.kind,
                version="v1",
                digest=item.asset.digest,
            )
            for item in assets
        )
        zero = "sha256:" + "0" * 64
        bundle = ProductBundle(
            bundle_id="cern-team",
            version=release["version"],
            distribution_id=OWNER,
            distribution_version=release["version"],
            bundle_digest=zero,
            assets=refs,
            compatibility=compatibility,
        )
        bundle = bundle.model_copy(
            update={"bundle_digest": canonical_bundle_digest(bundle)}
        )
        identity = DistributionRelease(
            distribution_id=OWNER,
            version=release["version"],
            origin="external",
            release_digest=zero,
            assets=(),
            bundle_ids=("cern-team",),
            compatibility=compatibility,
        )
        manifest = DistributionPackageManifest(
            distribution_release=identity,
            bundles=(bundle,),
            assets=tuple(assets),
            objects=tuple(objects.values()),
            provenance=DistributionPackageProvenance(
                repository_url=REPOSITORY,
                revision=release["revision"],
                builder="archi.install",
            ),
        )
        manifest = manifest.model_copy(
            update={
                "distribution_release": identity.model_copy(
                    update={"release_digest": canonical_release_digest(manifest)}
                )
            }
        )
        return build_distribution_package(manifest, sources, output)
