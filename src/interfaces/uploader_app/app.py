from __future__ import annotations

import os
import shlex
import time
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
from functools import wraps
import re

from flask import Flask, jsonify, redirect, render_template, request, url_for, session, flash
from flask_cors import CORS

from src.data_manager.collectors.persistence import PersistenceService
from src.data_manager.collectors.localfile_manager import LocalFileManager
from src.data_manager.collectors.scrapers.scraper_manager import ScraperManager
from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService
from src.data_manager.collectors.tickets.ticket_manager import TicketManager
from src.data_manager.vectorstore.loader_utils import load_text_from_path
from src.interfaces.chat_app.document_utils import check_credentials
from src.utils.env import read_secret, read_or_create_persistent_secret
from src.utils.logging import get_logger
from src.data_manager.collectors.utils.catalog_postgres import _METADATA_COLUMN_MAP
from src.utils.config_access import get_full_config

logger = get_logger(__name__)


class FlaskAppWrapper:
    """Uploader UI + API wrapper for the data manager service."""

    def __init__(
        self,
        app: Flask,
        *,
        post_update_hook: Optional[Callable[[], None]] = None,
        status_file: Optional[Path] = None,
    ) -> None:
        self.app = app
        self.config = get_full_config()
        self.global_config = self.config["global"]
        self.services_config = self.config["services"]

        self.data_path = self.global_config["DATA_PATH"]
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.persistence = PersistenceService(self.data_path, pg_config=self.pg_config)
        self.catalog = PostgresCatalogService(self.data_path, pg_config=self.pg_config)
        self.status_file = status_file or (Path(self.data_path) / "ingestion_status.json")

        # Persist an auto-generated key under DATA_PATH when none is configured,
        # so uploader sessions survive a restart (same policy as the chat app).
        # Distinct filename: the chat app persists under this secret name's
        # default file on the SAME shared data volume — the two independently-
        # authenticated apps must not end up signing with one key.
        self.app.secret_key = read_or_create_persistent_secret(
            "FLASK_UPLOADER_APP_SECRET_KEY", self.data_path,
            filename=".uploader_app_secret_key",
        )
        self.app.config["SESSION_COOKIE_NAME"] = "uploader_session"
        self.app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

        self.auth_config = (self.services_config or {}).get("data_manager", {}).get("auth", {}) or {}
        self.auth_enabled = bool(self.auth_config.get("enabled", False))
        self.api_token = read_secret("DM_API_TOKEN") or None
        self.admin_users = {
            user.strip().lower()
            for user in (self.auth_config.get("admins") or [])
            if user and user.strip()
        }
        self.default_admin_user = (self.auth_config.get("default_admin_user") or "admin").strip()
        self.default_admin_password = read_secret("DM_ADMIN_PASSWD")
        self.salt = read_secret("UPLOADER_SALT")
        self.accounts_path = self.global_config.get("ACCOUNTS_PATH")
        if self.auth_enabled:
            if not self.accounts_path:
                logger.warning("ACCOUNTS_PATH not configured; only default auth account avilable. Set is as DM_ADMIN_PASSWD in your secrets file.")
                self.auth_enabled = True
            else:
                os.makedirs(self.accounts_path, exist_ok=True)
                if not self.salt:
                    logger.warning("UPLOADER_SALT not set; account checks may fail.")

        self.scraper_manager = ScraperManager(dm_config=self.config.get("data_manager"))
        self.ticket_manager = TicketManager(dm_config=self.config.get("data_manager"))
        self.localfile_manager = LocalFileManager(dm_config=self.config.get("data_manager"))
        self.post_update_hook = post_update_hook

        CORS(self.app)

        protected = self.require_admin
        self.add_endpoint("/api/health", "health", self.health, methods=["GET"])
        self.add_endpoint("/document_index/upload", "upload", protected(self.upload), methods=["POST"])
        self.add_endpoint("/document_index/delete/<file_hash>", "delete", protected(self.delete))
        self.add_endpoint(
            "/document_index/delete_source/<source_type>",
            "delete_source",
            protected(self.delete_source),
        )
        self.add_endpoint("/document_index/upload_url", "upload_url", protected(self.upload_url), methods=["POST"])
        self.add_endpoint("/document_index/ingest_local_path", "ingest_local_path", protected(self.ingest_local_path), methods=["POST"])
        self.add_endpoint("/document_index/add_git_repo", "add_git_repo", protected(self.add_git_repo), methods=["POST"])
        self.add_endpoint("/document_index/remove_git_repo", "remove_git_repo", protected(self.remove_git_repo), methods=["POST"])
        self.add_endpoint("/document_index/add_jira_project", "add_jira_project", protected(self.add_jira_project), methods=["POST"])
        self.add_endpoint("/document_index/update_schedule", "update_schedule", protected(self.update_schedule), methods=["POST"])
        self.add_endpoint("/document_index/load_document/<path:file_hash>", "load_document", protected(self.load_document))
        # API endpoints for remote catalog access
        self.add_endpoint("/api/catalog/search", "api_catalog_search", protected(self.api_catalog_search), methods=["GET"])
        self.add_endpoint("/api/catalog/document/<path:resource_hash>", "api_catalog_document", protected(self.api_catalog_document), methods=["GET"])
        self.add_endpoint("/api/catalog/schema", "api_catalog_schema", protected(self.api_catalog_schema), methods=["GET"])
        if self.auth_enabled:
            self.add_endpoint("/login", "login", self.login, methods=["GET", "POST"])
            self.add_endpoint("/logout", "logout", self.logout)

    def add_endpoint(self, endpoint, endpoint_name, handler, methods=None):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods=methods or ["GET"])

    def run(self, **kwargs):
        self.app.run(**kwargs)

    def require_admin(self, handler):
        @wraps(handler)
        def wrapped(*args, **kwargs):
            if not self.auth_enabled:
                return handler(*args, **kwargs)
            if session.get("admin_logged_in"):
                return handler(*args, **kwargs)
            # Allow service-to-service calls authenticated via API token
            if self.api_token:
                auth_header = request.headers.get("Authorization", "")
                if auth_header == f"Bearer {self.api_token}":
                    return handler(*args, **kwargs)
            return redirect(url_for("login"))

        return wrapped

    def _is_admin_user(self, username: str) -> bool:
        if not username:
            return False
        normalized = username.strip().lower()
        if self.default_admin_user and normalized == self.default_admin_user.strip().lower():
            return True
        if not self.admin_users:
            return True
        return normalized in self.admin_users

    def login(self):
        if not self.auth_enabled:
            return redirect(url_for("document_index"))
        if session.get("admin_logged_in"):
            return redirect(url_for("document_index"))
        if request.method == "POST":
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""
            if username and password and self._is_admin_user(username):
                is_default_admin = (
                    self.default_admin_password
                    and username == self.default_admin_user
                    and password == self.default_admin_password
                )
                if is_default_admin or check_credentials(username, password, self.salt, self.accounts_path):
                    session["admin_logged_in"] = True
                    session["admin_user"] = username
                    return redirect(url_for("document_index"))
            flash("Invalid credentials")
        return render_template("login.html", sso_enabled=False, basic_auth_enabled=True)

    def logout(self):
        session.pop("admin_logged_in", None)
        session.pop("admin_user", None)
        return redirect(url_for("login"))

    def health(self):
        return jsonify({"status": "OK"}), 200

    def add_git_repo(self):
        repo_url = request.form.get("repo_url") or ""
        if not repo_url.strip():
            return jsonify({"error": "missing_repo_url"}), 400

        try:
            self.scraper_manager.collect_git(self.persistence, [repo_url.strip()])
            self._update_source_status("git", state="idle", last_run=self._now_iso())
            self._notify_update()
            return jsonify({"status": "ok"})
        except Exception as exc:
            logger.error("Failed to add git repo %s: %s", repo_url, exc)
            return jsonify({"error": "ingest_failed", "detail": str(exc)}), 500

    def remove_git_repo(self):
        repo_value = request.form.get("repo") or request.form.get("repo_url") or request.form.get("repo_name") or ""
        repo_name = self._extract_git_repo_name(repo_value)
        if not repo_name:
            return jsonify({"error": "missing_repo_name"}), 400

        try:
            self.catalog.refresh()
            to_remove = []
            for resource_hash in self.catalog.metadata_index.keys():
                metadata = self.catalog.get_metadata_for_hash(resource_hash) or {}
                if metadata.get("source_type") != "git":
                    continue
                if metadata.get("parent") == repo_name:
                    to_remove.append(resource_hash)

            if not to_remove:
                return jsonify({"error": "repo_not_found", "repo": repo_name, "deleted": 0}), 404

            for resource_hash in to_remove:
                self.persistence.delete_resource(resource_hash, flush=False)
            self.persistence.flush_index()
            self._update_source_status("git", state="idle", last_run=self._now_iso())
            self._notify_update()
            return jsonify({"status": "ok", "repo": repo_name, "deleted": len(to_remove)})
        except Exception as exc:
            logger.error("Failed to remove git repo %s: %s", repo_name, exc)
            return jsonify({"error": "delete_failed", "detail": str(exc)}), 500

    def add_jira_project(self):
        project_key = request.form.get("project_key") or ""
        if not project_key.strip():
            return jsonify({"error": "missing_project_key"}), 400

        if not self.ticket_manager or not self.ticket_manager.jira_client:
            return jsonify({"error": "jira_not_configured"}), 400

        try:
            self.ticket_manager.collect_jira(self.persistence, [project_key.strip()])
            self.persistence.flush_index()
            self._update_source_status("jira", state="idle", last_run=self._now_iso())
            self._notify_update()
            return jsonify({"status": "ok"})
        except Exception as exc:
            logger.error("Failed to add JIRA project %s: %s", project_key, exc)
            return jsonify({"error": "ingest_failed", "detail": str(exc)}), 500

    def upload(self):
        """Handle file uploads from the UI and persist them via the local files manager."""
        upload = request.files.get("file")
        if not upload:
            return jsonify({"error": "missing_file"}), 400

        filename = upload.filename or ""
        if not filename.strip():
            return jsonify({"error": "empty_filename"}), 400

        accepted = [ext.lower() for ext in self.global_config.get("ACCEPTED_FILES", [])]
        file_extension = os.path.splitext(filename)[1].lower()
        if accepted and file_extension not in accepted:
            return jsonify({"error": "unsupported_extension", "allowed": accepted}), 400

        try:
            stored_path = self.localfile_manager.ingest_uploaded_file(upload, self.persistence)
            self.persistence.flush_index()
            self._update_source_status("local_files", state="idle", last_run=self._now_iso())
            self._notify_update()
            return jsonify({"status": "ok", "path": str(stored_path)})
        except Exception as exc:
            logger.error("Failed to ingest uploaded file %s: %s", filename, exc)
            return jsonify({"error": "upload_failed", "detail": str(exc)}), 500

    def delete(self, file_hash):
        self.persistence.delete_resource(file_hash)
        self._notify_update()
        return redirect(url_for("document_index"))

    def delete_source(self, source_type):
        self.persistence.delete_by_metadata_filter("source_type", source_type)
        self._notify_update()
        return redirect(url_for("document_index"))

    def upload_url(self):
        """
        Use the ScraperManager to collect and persist a single URL provided via form data.

        Optional form field ``allow_sso_fallback=true`` enables a retry via the
        Selenium ``CERNSSOScraper`` when the anonymous LinkScraper hits a
        Keycloak login page. The agent decides per-request whether to opt in
        (gated by its ``services.chat_app.tools.ingest_url`` config). The
        retry only fires if the SSO source is enabled in data-manager config
        (``data_manager.sources.sso.enabled``) and the SSO_* secrets exist.
        """
        url = request.form.get("url")
        depth_raw = request.form.get("depth")
        depth: Optional[int] = None
        if depth_raw not in (None, ""):
            try:
                depth = int(depth_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "invalid_depth"}), 400
            if depth < 0:
                return jsonify({"error": "invalid_depth"}), 400
            # LinkScraper currently uses max_depth >= 1 for the initial URL fetch.
            if depth == 0:
                depth = 1
        allow_sso_fallback = (request.form.get("allow_sso_fallback") or "").strip().lower() in {"1", "true", "yes"}
        if url:
            logger.info("Uploading the following URL: %s (allow_sso_fallback=%s)", url, allow_sso_fallback)
            scraper_used: Optional[str] = None
            try:
                if self.scraper_manager._is_indico_url(url):
                    scraped_count = self.scraper_manager.collect_indico(self.persistence, indico_urls=[url])
                    scraper_used = "indico"
                else:
                    scraped_count = self.scraper_manager.collect_links(self.persistence, link_urls=[url], max_depth=depth)
                    scraper_used = "link"
                # Guard: if LinkScraper followed an SSO redirect and ingested a Keycloak login
                # page, the count above is misleading. Remove any just-ingested resources whose
                # stored content matches the Keycloak login signature.
                removed = self._purge_keycloak_login_pages_for_url(url)
                if removed:
                    scraped_count = max(0, scraped_count - removed)
                    if scraped_count == 0 and allow_sso_fallback:
                        # Retry through the Selenium CERNSSOScraper. The agent opted
                        # in by setting allow_sso_fallback=true; the data-manager
                        # still requires the SSO source to be enabled. Pass the
                        # request's depth (defaults to 1) — the Selenium crawl is
                        # otherwise governed by base_source_depth=5, which can
                        # spider hundreds of pages from a single seed and blow
                        # past the agent's client timeout.
                        sso_count = self._sso_retry(url, max_depth=(depth or 1))
                        if sso_count is not None and sso_count > 0:
                            scraped_count = sso_count
                            scraper_used = "sso"
                            logger.info(
                                "ingest_url SSO retry succeeded for %s (%d resource(s))",
                                url, sso_count,
                            )
                        else:
                            self._notify_update()
                            return (
                                jsonify(
                                    {
                                        "error": "auth_required",
                                        "detail": (
                                            "URL redirected to a CERN SSO login page; "
                                            "anonymous LinkScraper cannot fetch it. SSO "
                                            "fallback was attempted but did not yield "
                                            "any resources (see data-manager logs)."
                                        ),
                                        "url": url,
                                    }
                                ),
                                502,
                            )
                    elif scraped_count == 0:
                        self._notify_update()
                        return (
                            jsonify(
                                {
                                    "error": "auth_required",
                                    "detail": (
                                        "URL redirected to a CERN SSO login page; "
                                        "anonymous LinkScraper cannot fetch it. "
                                        "Set allow_sso_fallback=true to retry via the "
                                        "Selenium SSO scraper, or use a tool that "
                                        "authenticates (e.g. ingest_indico_event)."
                                    ),
                                    "url": url,
                                }
                            ),
                            502,
                        )
                self.persistence.flush_index()
                self._update_source_status("web", state="idle", last_run=self._now_iso())
                added_to_urls = True
            except Exception as exc:
                logger.exception("Failed to upload URL: %s", exc)
                added_to_urls = False
                upload_error = str(exc)

            if added_to_urls:
                logger.info("URL uploaded successfully (scraper=%s)", scraper_used)
                self._notify_update()
                return jsonify(
                    {
                        "status": "ok",
                        "resources_scraped": scraped_count,
                        "scraper": scraper_used,
                    }
                )
            else:
                return jsonify({"error": "upload_failed", "detail": upload_error}), 500
        else:
            return jsonify({"error": "missing_url"}), 400

    def _sso_retry(self, url: str, max_depth: int = 1) -> Optional[int]:
        """Run the configured CERNSSOScraper against *url* and return the number
        of new resources persisted. ``None`` if SSO is not enabled or the retry
        could not run (the caller treats that as failure).

        ``max_depth`` defaults to 1 (just the seed URL). For one-shot agent
        ingests, anything higher means crawling every internal link and
        typically blowing past the client timeout.
        """
        if not getattr(self.scraper_manager, "sso_enabled", False):
            logger.info(
                "ingest_url SSO retry requested for %s but data_manager.sources.sso.enabled=false",
                url,
            )
            return None
        before = set(self.persistence.catalog.get_resource_hashes_by_metadata_filter("url", url))
        logger.info("SSO retry starting for %s (max_depth=%d)", url, max_depth)
        try:
            self.scraper_manager.collect_sso(self.persistence, sso_urls=[url], max_depth=max_depth)
        except Exception as exc:
            logger.exception("SSO retry raised for %s: %s", url, exc)
            return None
        after = set(self.persistence.catalog.get_resource_hashes_by_metadata_filter("url", url))
        new_hashes = after - before
        logger.info("SSO retry done for %s: %d new resource(s)", url, len(new_hashes))
        return len(new_hashes)

    def ingest_local_path(self):
        """Ingest every file under a directory living on a path the caller already
        has access to (typically a shared volume populated by a sibling container
        such as the Indico MCP server).

        Form fields:
          - path (str, required): absolute path to a directory whose contents
            should be ingested. Must be under one of the configured allowed roots.
          - source_type (str, default "local_files"): metadata.source_type tag.
          - target_subdir (str, optional): subdirectory under DATA_PATH to write
            persisted files into. Defaults to ``source_type``.
          - any other form field: copied verbatim into each resource's metadata
            (e.g. ``event_id``, ``url``, ``contribution_id``).
        """
        raw_path = (request.form.get("path") or "").strip()
        if not raw_path:
            return jsonify({"error": "missing_path"}), 400

        # Reject `..` segments pre-resolution so traversal attempts can't even
        # reach the symlink layer. Defense-in-depth against a misconfigured caller.
        if ".." in Path(raw_path).parts:
            return jsonify({"error": "invalid_path", "detail": "path contains '..'"}), 400

        try:
            requested = Path(raw_path).resolve(strict=False)
        except Exception:
            return jsonify({"error": "invalid_path"}), 400

        allowed_roots = self._allowed_ingest_roots()
        # commonpath is symlink-resilient: if a symlink under the root points
        # outside, the resolved path won't share commonpath with the root and
        # we'll reject. `requested.parents` membership wouldn't catch that.
        def _under_root(p: Path, root: Path) -> bool:
            try:
                return os.path.commonpath([str(p), str(root)]) == str(root)
            except ValueError:
                # Different drives on Windows (irrelevant here, but be safe).
                return False

        if not any(_under_root(requested, root) for root in allowed_roots):
            return (
                jsonify(
                    {
                        "error": "path_not_allowed",
                        "detail": f"Path must be under one of: {[str(r) for r in allowed_roots]}",
                        "path": str(requested),
                    }
                ),
                403,
            )

        if not requested.exists() or not requested.is_dir():
            return jsonify({"error": "path_not_found", "path": str(requested)}), 404

        source_type = (request.form.get("source_type") or "local_files").strip()
        target_subdir = (request.form.get("target_subdir") or "").strip() or None

        reserved = {"path", "source_type", "target_subdir"}
        extra_metadata = {
            k: v for k, v in request.form.items() if k not in reserved and v not in (None, "")
        }

        try:
            records = self.localfile_manager.ingest_directory(
                requested,
                self.persistence,
                source_type=source_type,
                extra_metadata=extra_metadata,
                target_subdir=target_subdir,
            )
        except ValueError as exc:
            return jsonify({"error": "invalid_request", "detail": str(exc)}), 400
        except Exception as exc:
            logger.exception("ingest_local_path failed for %s: %s", requested, exc)
            return jsonify({"error": "ingest_failed", "detail": str(exc)}), 500

        self.persistence.flush_index()
        self._notify_update()
        return jsonify(
            {
                "status": "ok",
                "resources_ingested": len(records),
                "resources": records,
            }
        )

    def _allowed_ingest_roots(self) -> List[Path]:
        """Roots under which ``ingest_local_path`` is allowed to read.

        Read from the env var ``INGEST_ALLOWED_ROOTS`` (colon-separated paths) or
        fall back to ``/shared/indico-downloads`` — the canonical cross-container
        share for Indico MCP downloads.
        """
        env_roots = os.environ.get("INGEST_ALLOWED_ROOTS", "")
        candidates = [p for p in env_roots.split(":") if p.strip()] or ["/shared/indico-downloads"]
        resolved: List[Path] = []
        for c in candidates:
            try:
                resolved.append(Path(c).resolve())
            except Exception:
                continue
        return resolved

    def _purge_keycloak_login_pages_for_url(self, url: str) -> int:
        """Delete any resource that was just ingested for *url* but whose stored
        content is a Keycloak login page. Returns the number of rows removed.

        The LinkScraper happily follows the CERN SSO redirect and stores the
        login HTML under the original URL — a false positive that previously
        looped the agent. This walks the catalog by ``url:<url>`` and removes
        rows whose persisted file matches the Keycloak signature.
        """
        try:
            matches = self.persistence.catalog.get_resource_hashes_by_metadata_filter("url", url)
        except Exception as exc:
            logger.debug("Could not query catalog for url=%s: %s", url, exc)
            return 0

        removed = 0
        for resource_hash in matches:
            try:
                stored_file = self.persistence.catalog.file_index.get(resource_hash)
                if not stored_file:
                    continue
                file_path = Path(stored_file)
                if not file_path.is_absolute():
                    file_path = (Path(self.data_path) / file_path).resolve()
                if not file_path.exists():
                    continue
                head = file_path.read_bytes()[:2048].decode("utf-8", errors="ignore")
                if (
                    "login-pf" in head
                    or 'id="kc-header"' in head
                    or 'class="login-pf"' in head
                ):
                    self.persistence.delete_resource(resource_hash, flush=False)
                    removed += 1
                    logger.info("Purged Keycloak login page ingested under %s (hash=%s)", url, resource_hash)
            except Exception as exc:
                logger.debug("Failed to inspect/remove resource %s: %s", resource_hash, exc)
        return removed

    def update_schedule(self):
        source = (request.form.get("source") or "").strip().lower()
        schedule = (request.form.get("schedule") or "").strip()
        if not source:
            return jsonify({"error": "missing_source"}), 400

        sources_cfg = (self.config.get("data_manager", {}) or {}).get("sources", {}) or {}
        if source not in sources_cfg:
            return jsonify({"error": "unknown_source", "source": source}), 404

        if schedule:
            try:
                from croniter import croniter
                logger.debug(f"Updating source {source} schedule to {schedule}")

                croniter(schedule)
            except Exception as exc:
                return jsonify({"error": "invalid_schedule", "detail": str(exc)}), 400

        try:
            self._update_source_status(source, schedule=schedule)
            return jsonify({"status": "ok", "source": source, "schedule": schedule})
        except Exception as exc:
            logger.error("Failed to update schedule for %s: %s", source, exc)
            return jsonify({"error": "schedule_update_failed", "detail": str(exc)}), 500

    def load_document(self, file_hash):
        index = self.catalog.file_index
        if file_hash in index.keys():
            path = self.catalog.get_filepath_for_hash(file_hash)
            metadata = self.catalog.get_metadata_for_hash(file_hash) or {}

            document = ""
            suffix = metadata.get("suffix") or (path.suffix if path else "")

            try:
                if suffix.lower() in {".html", ".htm"} and path and path.exists():
                    # For HTML, return the raw document so the preview can render fully.
                    document = path.read_text(encoding="utf-8", errors="ignore")
                elif suffix.lower() == ".pdf" and path and path.exists():
                    document = f"__PDF_INLINE__::{path.as_posix()}"
                else:
                    document_obj = self.catalog.get_document_for_hash(file_hash)
                    if hasattr(document_obj, "page_content"):
                        document = document_obj.page_content or ""
                    elif isinstance(document_obj, str):
                        document = document_obj
                    else:
                        document = load_text_from_path(path) if path else ""
            except Exception as exc:
                logger.warning("Failed to load document content for %s: %s", file_hash, exc)

            display_name = metadata.get("display_name") or metadata.get("file_name") or ""
            title = metadata.get("title") or display_name
            return jsonify(
                {
                    "document": document or "",
                    "display_name": display_name,
                    "source_type": metadata.get("source_type") or "",
                    "original_url": metadata.get("url") or "",
                    "title": title or "",
                }
            )

        return jsonify(
            {
                "document": "Document not found",
                "display_name": "Error",
                "source_type": "null",
                "original_url": "no_url",
                "title": "Not found",
            }
        )

    def _notify_update(self) -> None:
        if not self.post_update_hook:
            return
        try:
            self.post_update_hook()
        except Exception as exc:
            logger.warning("Post-update hook failed: %s", exc)

    def _load_source_status(self) -> Dict[str, Dict[str, str]]:
        if not self.status_file.exists():
            return {}
        try:
            import json

            return json.loads(self.status_file.read_text())
        except Exception as exc:
            logger.warning("Failed to read source status file: %s", exc)
            return {}

    def _update_source_status(
        self,
        source: str,
        *,
        state: Optional[str] = None,
        last_run: Optional[str] = None,
        schedule: Optional[str] = None,
    ) -> None:
        try:
            import json
            data = self._load_source_status()
            entry = data.get(source, {})
            if state is not None:
                entry["state"] = state
            if last_run is not None:
                entry["last_run"] = last_run
            if schedule is not None:
                if schedule:
                    entry["schedule"] = schedule
                else:
                    entry.pop("schedule", None)
            data[source] = entry
            logger.debug(f"Updated source status with state {state}, last_run: {last_run}, schedule: {schedule}")
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            self.status_file.write_text(json.dumps(data))
        except Exception as exc:
            logger.warning("Failed to update source status: %s", exc)

    def _now_iso(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def _extract_git_repo_name(self, value: str) -> str:
        if not value:
            return ""
        raw = value.strip()
        if not raw:
            return ""
        pattern = r"(?:github|gitlab)\.[\w.]+[:/][^/]+/([\w.-]+)(?:\.git|/|$)"
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1)
        candidate = raw.rstrip("/").split("/")[-1]
        if candidate.endswith(".git"):
            candidate = candidate[:-4]
        return candidate

    # -------------------------
    # API endpoints
    # -------------------------
    def api_catalog_search(self):
        start_time = time.monotonic()
        logger.debug("Received catalog search request: %s", request.args)
        query = request.args.get("q") or request.args.get("query") or ""
        if not query.strip():
            return jsonify({"hits": [], "total_duration": 0.0})
        limit = request.args.get("limit", default=5, type=int)
        window = request.args.get("window", default=-1, type=int)
        search_content = request.args.get("search_content", default="true").lower() != "false"
        mode = (request.args.get("mode") or "").strip().lower()
        regex = _parse_bool(request.args.get("regex"), default=False)
        case_sensitive = _parse_bool(request.args.get("case_sensitive"), default=False)
        max_matches_per_file = request.args.get("max_matches_per_file", default=3, type=int)
        before = request.args.get("before", default=0, type=int)
        after = request.args.get("after", default=0, type=int)

        filters, free_query = _parse_metadata_query(query)
        q_lower = free_query.lower()
        hits = []
        self.catalog.refresh()
        if not search_content:
            results = self.catalog.search_metadata(free_query, limit=limit, filters=filters)
            for item in results:
                metadata = item.get("metadata") or {}
                snippet = (
                    metadata.get("display_name")
                    or metadata.get("file_name")
                    or metadata.get("title")
                    or metadata.get("url")
                    or ""
                )
                hits.append(
                    {
                        "hash": item["hash"],
                        "path": str(item["path"]),
                        "metadata": metadata,
                        "snippet": snippet,
                    }
                )
        else:
            if mode == "grep":
                if not free_query.strip():
                    return jsonify({"hits": [], "total_duration": 0.0})
                try:
                    pattern = _compile_query_pattern(
                        free_query, regex=regex, case_sensitive=case_sensitive
                    )
                except re.error as exc:
                    return jsonify({"error": f"invalid_regex: {exc}"}), 400

                candidate_hashes = None
                candidate_metadata: Dict[str, Dict[str, object]] = {}
                if filters:
                    candidates = self.catalog.search_metadata("", limit=None, filters=filters)
                    candidate_hashes = {item["hash"] for item in candidates}
                    candidate_metadata = {
                        item["hash"]: item.get("metadata") or {}
                        for item in candidates
                    }

                if candidate_hashes is None:
                    iterable = self.catalog.iter_files()
                else:
                    iterable = []
                    for resource_hash in candidate_hashes:
                        path = self.catalog.get_filepath_for_hash(resource_hash)
                        if path:
                            iterable.append((resource_hash, path))

                for resource_hash, path in iterable:
                    metadata = candidate_metadata.get(resource_hash) or self.catalog.get_metadata_for_hash(resource_hash) or {}
                    text = load_text_from_path(path) or ""
                    if not text:
                        continue
                    matches = _grep_text_lines(
                        text,
                        pattern,
                        before=before,
                        after=after,
                        max_matches=max_matches_per_file,
                    )
                    if not matches:
                        continue
                    hits.append(
                        {
                            "hash": resource_hash,
                            "path": str(path),
                            "metadata": metadata,
                            "matches": matches,
                            "snippet": matches[0].get("text", ""),
                        }
                    )
                    if len(hits) >= limit:
                        break

                total_duration = time.monotonic() - start_time
                logger.debug(
                    "Catalog grep search completed in %.3f seconds with %d hits",
                    total_duration,
                    len(hits),
                )
                return jsonify({"hits": hits, "total_duration": total_duration})

            candidate_hashes = None
            candidate_metadata: Dict[str, Dict[str, object]] = {}
            if filters:
                candidates = self.catalog.search_metadata("", limit=None, filters=filters)
                candidate_hashes = {item["hash"] for item in candidates}
                candidate_metadata = {
                    item["hash"]: item.get("metadata") or {}
                    for item in candidates
                }

            if candidate_hashes is None:
                iterable = self.catalog.iter_files()
            else:
                iterable = []
                for resource_hash in candidate_hashes:
                    path = self.catalog.get_filepath_for_hash(resource_hash)
                    if path:
                        iterable.append((resource_hash, path))

            for resource_hash, path in iterable:
                metadata = candidate_metadata.get(resource_hash) or self.catalog.get_metadata_for_hash(resource_hash) or {}
                flattened_meta = _flatten_metadata(metadata)
                if q_lower:
                    meta_match = any(q_lower in k.lower() or q_lower in v.lower() for k, v in flattened_meta.items())
                else:
                    meta_match = True

                snippet = ""
                content_match = False
                text = ""
                if q_lower:
                    text = load_text_from_path(path) or ""
                    if text:
                        idx = text.lower().find(q_lower)
                        if idx != -1:
                            content_match = True
                            snippet = _collect_snippet(text, idx, len(free_query), window=window)
                    else:
                        logger.error("No text content loaded from %s for search", path)

                if meta_match and not content_match:
                    if q_lower:
                        if not text:
                            text = load_text_from_path(path) or ""
                            if not text:
                                logger.error("No text content loaded from %s for metadata match", path)
                        snippet = text
                    else:
                        snippet = (
                            metadata.get("display_name")
                            or metadata.get("file_name")
                            or metadata.get("url")
                            or ""
                        )

                if meta_match or content_match:
                    hits.append(
                        {
                            "hash": resource_hash,
                            "path": str(path),
                            "metadata": metadata,
                            "snippet": snippet,
                        }
                    )
                if len(hits) >= limit:
                    break

        total_duration = time.monotonic() - start_time
        logger.debug("Catalog search completed in %.3f seconds with %d hits", total_duration, len(hits))
        return jsonify({"hits": hits, "total_duration": total_duration})

    def api_catalog_document(self, resource_hash: str):
        max_chars = request.args.get("max_chars", default=4000, type=int)
        self.catalog.refresh()
        path = self.catalog.get_filepath_for_hash(resource_hash)
        if not path:
            return jsonify({"error": "not_found"}), 404
        metadata = self.catalog.get_metadata_for_hash(resource_hash) or {}
        text = load_text_from_path(path) or ""
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        return jsonify({"hash": resource_hash, "path": str(path), "metadata": metadata, "text": text})

    def api_catalog_schema(self):
        """
        Return metadata schema hints for agents: supported keys and distinct values for source_type/suffix.
        """
        keys = sorted(_METADATA_COLUMN_MAP.keys())
        distinct = self.catalog.get_distinct_metadata(["source_type", "suffix"])
        return jsonify({
            "keys": keys,
            "source_types": distinct.get("source_type", []),
            "suffixes": distinct.get("suffix", []),
        })


def _flatten_metadata(data: Dict[str, object], prefix: str = "") -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_metadata(value, prefix=full_key))
        else:
            flattened[full_key] = "" if value is None else str(value)
    return flattened


_METADATA_ALIAS_MAP = {
    "resource_type": "source_type",
    "resource_id": "ticket_id",
}


def _parse_metadata_query(query: str) -> Tuple[Dict[str, str] | List[Dict[str, str]], str]:
    filters_groups: List[Dict[str, str]] = []
    current_group: Dict[str, str] = {}
    free_tokens = []
    try:
        tokens = shlex.split(query)
    except ValueError as exc:
        # Avoid 500s on malformed quoted input generated by LLM tools.
        logger.warning("Invalid metadata query syntax; using fallback tokenization: %s", exc)
        tokens = query.split()

    for token in tokens:
        if token.upper() == "OR":
            if current_group:
                filters_groups.append(current_group)
                current_group = {}
            continue
        if ":" in token:
            key, value = token.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                # Normalize legacy keys to canonical column names
                key = _METADATA_ALIAS_MAP.get(key, key)
                current_group[key] = value
                continue
        free_tokens.append(token)

    if current_group:
        filters_groups.append(current_group)

    if not filters_groups:
        filters: Dict[str, str] | List[Dict[str, str]] = {}
    elif len(filters_groups) == 1:
        filters = filters_groups[0]
    else:
        filters = filters_groups

    return filters, " ".join(free_tokens)


def _parse_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _compile_query_pattern(query: str, *, regex: bool, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = query if regex else re.escape(query)
    return re.compile(pattern, flags)


def _grep_text_lines(
    text: str,
    pattern: re.Pattern[str],
    *,
    before: int = 0,
    after: int = 0,
    max_matches: int = 3,
) -> list[Dict[str, object]]:
    if max_matches <= 0:
        return []
    lines = text.splitlines()
    matches: list[Dict[str, object]] = []
    for idx, line in enumerate(lines):
        if not pattern.search(line):
            continue
        match = {
            "line": idx + 1,
            "text": line,
            "before": lines[max(0, idx - before) : idx] if before else [],
            "after": lines[idx + 1 : idx + 1 + after] if after else [],
        }
        matches.append(match)
        if len(matches) >= max_matches:
            break
    return matches


def _collect_snippet(text: str, start_idx: int, query_len: int, window: int = -1) -> str:
    start = max(start_idx - window, 0) if window >= 0 else 0
    end = min(start_idx + query_len + window, len(text)) if window >= 0 else len(text)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    excerpt = text[start:end].replace("\n", " ")
    return f"{prefix}{excerpt}{suffix}"
