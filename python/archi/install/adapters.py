"""Consumer-owned SDK facades for bounded immutable cache readers."""

from __future__ import annotations

from pathlib import Path

from okg.deployment import ConnectorAdapter, ConnectorRun

from .snapshots import verify_cache, verify_records


class _FrozenConnector:
    def __init__(self, reader, verify):
        self.reader = reader
        self.verify = verify
        self.name = reader.name
        self.profile = reader.profile

    def run(self, ctx):
        if ctx.sync_scope:
            raise ValueError(
                "bounded Archi readers do not support narrowed sync scopes"
            )
        self.verify()
        result = self.reader.run(ctx.run_id, mode=ctx.mode)
        if not isinstance(result, ConnectorRun):
            raise ValueError("Archi reader returned an unsupported run result")
        return result


class _FrozenAdapter(ConnectorAdapter):
    def _bind(self, reader, verify):
        self._reader = reader
        self._verify = verify
        super().__init__(_FrozenConnector(reader, verify))
        self.change_probe = reader.change_probe
        self.change_probe_kind = reader.change_probe_kind
        # No probe-short-circuit or completed-scope permission is added here.

    def preflight(self, *args, **kwargs):
        self._verify()
        return self._reader.preflight(*args, **kwargs)

    @property
    def cache_paths(self):
        return self._reader.cache_paths


def _verifier(configuration_root, snapshot_digests, records=None):
    """Check cache bytes, then record identifiers when an allowlist applies."""
    digests = dict(snapshot_digests)

    def verify():
        verify_cache(configuration_root, digests)
        if records is not None:
            verify_records(configuration_root, digests, *records)

    return verify


class FrozenCMSSWAdapter(_FrozenAdapter):
    profile = "reference_catalog"

    def __init__(self, *, configuration_root: str, snapshot_digests: dict[str, str]):
        verify = _verifier(configuration_root, snapshot_digests)
        verify()
        from archi.sources.cmssw import CMSSWReleaseSource

        path = "snapshots/cmssw/releases.map"
        self._bind(
            CMSSWReleaseSource(
                map_cache_path=str(Path(configuration_root) / path),
                map_cache_digest=snapshot_digests[path],
                fetch=False,
            ),
            verify,
        )


class FrozenDocumentationAdapter(_FrozenAdapter):
    profile = "discovery_crawl"

    def __init__(
        self,
        *,
        configuration_root: str,
        snapshot_digests: dict[str, str],
        record_allowlist: list[str],
    ):
        verify = _verifier(
            configuration_root,
            snapshot_digests,
            ("snapshots/docsite/records.json", "url", list(record_allowlist)),
        )
        verify()
        from archi.sources.docs import DocumentationSource

        root = Path(configuration_root)
        self._bind(
            DocumentationSource(
                source_name="docsite_snapshot",
                records_path=str(root / "snapshots/docsite/records.json"),
                jira_records_path=str(root / "snapshots/jira/records.json"),
            ),
            verify,
        )


class FrozenJiraAdapter(_FrozenAdapter):
    profile = "mutable_api"
    requires_live_call_authorization = False

    def __init__(
        self,
        *,
        configuration_root: str,
        snapshot_digests: dict[str, str],
        record_allowlist: list[str],
    ):
        verify = _verifier(
            configuration_root,
            snapshot_digests,
            ("snapshots/jira/records.json", "key", list(record_allowlist)),
        )
        verify()
        from archi.sources.jira import JiraIssueSource

        root = Path(configuration_root)
        self._bind(
            JiraIssueSource(
                records_path=str(root / "snapshots/jira/records.json"),
                meta_path=str(root / "snapshots/jira/meta.json"),
                project_keys=["CMSPROD"],
            ),
            verify,
        )
