"""Cache downloaders for the cache-backed archi readers.

Moved from okg-deployments ``cms/scripts/`` (main ``32f2b4e3c2``):
``jira`` (``download_jira.py``), ``static_docs`` (``download_static_docs.py``,
with its ``source-download-manifest.yaml``) and ``sso_login``
(``sso-login.py``). ``cmssst`` (CMS Site Support Team site status) is new
here and needs no credentials. Each runs as
``python -m archi.downloaders.<name>``. They need network access, the first
three need CERN credentials, and none of them imports anything from okg.

The cms-catalogs downloaders (issue #903) share :mod:`._catalog`: one HTTPS
client per run (optional X.509 proxy, bounded retries), and a group write
that stages the cache files, runs the real reader over the staged copy and
moves them into place only when the reader accepts them whole. ``cric``
(CMS CRIC topology, needs an X.509 proxy) and ``cric_core`` (WLCG CRIC core
services and federations, anonymous) are the first. ``_catalog`` reuses
:mod:`archi.auth` (the readers' path resolution and the preflight's
proxy-expiry decoder), so these do import okg through it.
"""
