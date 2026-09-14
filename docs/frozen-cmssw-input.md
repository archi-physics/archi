# Use a frozen CMSSW release catalog

A frozen binding reads a selected `releases.map` file and verifies its raw SHA256 before parsing. It never downloads replacement bytes. Use it when a deployment must consume a reviewed catalog snapshot instead of the current public catalog.

Configure the existing CMSSWReleaseSource with:

```yaml
params:
  map_cache_path: /owned/catalog/releases.map
  map_cache_digest: sha256:<64 lowercase hexadecimal digits>
  fetch: false
```

Replace the path and digest with the actual authorized snapshot and independently verified hash. Relative paths resolve against the source's existing `base` or `ARCHI_DATA_ROOT`; the existing `cache_path` alias also works. The packaged example is `archi/bundles/cern-team/source-defaults/cmssw_releases_frozen.yaml.example`. It is separate from the live default and needs `cmssw_map_path` and `cmssw_map_digest` values when rendered.

A digest without a map path, a malformed digest, or `fetch: true` is rejected. A missing file yields `cache_missing` during preflight and refuses ingestion. A digest mismatch refuses both preflight and ingestion; neither path contacts the network. Frozen files must be valid UTF-8. Unpinned live/cache behavior retains its existing decoding and fetching rules.

Each preflight and ingestion reads and verifies independently. Verification, parsing and revision hashing use the same buffer, so a later filesystem write cannot change the bytes represented by already-emitted facts. Source revision exposes `map_cache_digest` (the raw file pin) separately from `content_hash` (the existing path-aware hash). The change probe includes the frozen binding; it selects whether ingestion should run and does not itself certify the digest. Unchanged verified inputs can still use the existing probe optimization.

Malformed catalog lines and truncated limits retain the existing incomplete-scope behavior. A valid digest proves byte identity, not that every row parses or that a limited selection covers the whole catalog. Keep the real input's origin, capture time, hash and permitted scope with the deployment evidence.

## Packaged installer handoff

This consumer contract is not a claim that the existing OKG packaged entry accepts a frozen CMSSW snapshot. Its retained Archi translation pins the old consumer wheel, fixes the live CMSSW parameters, accepts only four documentation/Jira snapshot files, and excludes other `.example` files from its translated package. The installer owner must review the new wheel identity, frozen parameters and asset inventory through its existing workflow. Do not alter installed framework files or intercept networking to make an old release appear to support this contract.

The historical CMSSW map/cache was not present in the recovered handoff and could not be found in the available files. The doc/Jira snapshot is available, but its four-file manifest does not identify the CMSSW catalog. Real historical replay remains blocked on that artifact and supported installer binding; fixture-based consumer tests do not satisfy the real baseline. These dependencies remain tracked under `mitdbg/okg#1795` and the baseline issue `mitdbg/okg#1872`.
