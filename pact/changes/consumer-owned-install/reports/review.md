# Independent adversarial review (consumer slice of mitdbg/okg#1906)

An independent reviewer read `origin/archi_v3..5a7c153c` without editing. Fixes
landed in `3d85d1f0`; each fixed defect has a regression test that fails on the
reviewed code. The framework slice had its own review, recorded in
mitdbg/okg `pact/changes/consumer-separation/reports/reviews.md`.

| # | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| 1 | Should-fix | `build_prepared_package` accepted configuration prepared for another target and any source selection. | Fixed: refuses a different target, name, module set, selection, authority mismatch or unpinned cache bytes. |
| 2 | Should-fix | The framework import contract test failed when run alone. | Fixed: 36 passed alone. |
| 3 | Should-fix | The pipe-bridge benchmark report check could not fail in either repository. | Resolved: the report and transcripts now belong here, where the moved test validates them; recorded in `docs/consumer-install.md` and the framework inventory. No report exists yet. |
| 4 | Should-fix | The producer accepted policies the installer refuses. | Fixed: every audited field, redaction with `store_raw: false` and no live calls are required; tests use the example policy. |
| 5 | Should-fix | The record allowlist the old framework hook enforced was no longer checked at run time. | Fixed: Jira and documentation adapters take the allowlist and refuse mismatches at construction, preflight and every run. |
| 6 | Should-fix | Jira and documentation readers reopen files after digest verification. | Disclosed in code; reader algorithms are unchanged and the framework code behaved the same way. |
| 7 | Should-fix | The reserved-name test passed for the wrong reason. | Fixed with an exact refusal match. |
| 8 | Should-fix | The design report claimed a consumer CLI that does not exist. | Corrected. |
| 9 | Nit | `read_snapshot` accepted schemas that preparation always refuses. | Fixed: only `archi.frozen-snapshot/v1`. |
| 10 | Nit | Derived caches over the reader size limit could be packaged. | Fixed: refused at preparation. |
| 11 | Nit | `string.Template` also substitutes `$name` and `$$`; answers duplicate profile defaults; numbered asset ids shift with the file set. | Substitution now handles only `${name}`; the other two are disclosed, unchanged. |
| 12 | Nit | No determinism, recovery-construction or declared-mode tests. | Determinism and declared-mode round trips added; recovery construction is not exercised by consumer tests. |

Reviewer uncertainty carried forward: unselected repository sources did not run
in the installed proof (its inventory matched only the three frozen readers);
readers embed the run id in record revisions, so unchanged repeats re-emit,
which is the known replay defect and a framework dependency.
