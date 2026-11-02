"""Archi wrappers for substrate declarative linkers.

Provenance: ported from ``cms/enrichers/declarative.py`` (26 LOC,
okg-deployments ``main@f33a9c4``). Changes: class de-CMS-ified
(``CMSGlobalTagReleaseLinker`` -> :class:`GlobalTagReleaseLinker`), and
the deployment-tree-relative rule file path is replaced by the packaged
default (:func:`archi.enrichment.defaults.cross_link_rules_file`),
overridable through the standard ``rules``/``rules_files``/``rules_dir``
kwargs. The default enricher ``name`` keeps its historical value for
cutover parity with the comp-ops instance.

Deployment wiring (enrichers block)::

    enrichers:
      - class: archi.enrichment.declarative.GlobalTagReleaseLinker
"""
from __future__ import annotations

from okg.substrate.library.linkers.declarative import DeclarativeLinker

from archi.enrichment.defaults import cross_link_rules_file


class GlobalTagReleaseLinker(DeclarativeLinker):
    """Load the packaged global-tag release rule.

    The substrate ``DeclarativeLinker`` owns the linking behavior. This
    wrapper only resolves the packaged rule file without relying on the
    caller's current working directory or a deployment checkout.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "cms_declarative_global_tag_release")
        # None/empty rule sources mean "not provided". A bare key-presence
        # check treated rules_files=None/[] as a caller override, which
        # suppressed the packaged default and let DeclarativeLinker load
        # its entire substrate rule set under this linker's name.
        if not any(kwargs.get(k) for k in ("rules", "rules_files", "rules_dir")):
            for key in ("rules", "rules_files", "rules_dir"):
                kwargs.pop(key, None)
            kwargs["rules_files"] = [
                str(cross_link_rules_file("global_tag_release.yaml"))
            ]
        super().__init__(**kwargs)
