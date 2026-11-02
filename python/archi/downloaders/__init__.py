"""Cache downloaders for the cache-backed archi readers.

Moved from okg-deployments ``cms/scripts/`` (main ``32f2b4e3c2``):
``jira`` (``download_jira.py``), ``static_docs`` (``download_static_docs.py``,
with its ``source-download-manifest.yaml``) and ``sso_login``
(``sso-login.py``). ``cmssst`` (CMS Site Support Team site status) is new
here and needs no credentials. Each runs as
``python -m archi.downloaders.<name>``. They need network access, the first
three need CERN credentials, and none imports anything from okg.
"""
