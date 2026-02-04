from __future__ import annotations

import os
import shlex
import time
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
from functools import wraps
import secrets
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
from src.utils.env import read_secret
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

        secret_key = read_secret("FLASK_UPLOADER_APP_SECRET_KEY") or secrets.token_hex(32)
        self.app.secret_key = secret_key
        self.app.config["SESSION_COOKIE_NAME"] = "uploader_session"

        self.auth_config = (self.services_config or {}).get("data_manager", {}).get("auth", {}) or {}
        self.auth_enabled = bool(self.auth_config.get("enabled", True))
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
        self.add_endpoint("/", "index", protected(self.index))
        self.add_endpoint("/api/health", "health", self.health, methods=["GET"])
        self.add_endpoint("/document_index/", "document_index", protected(self.document_index))
        self.add_endpoint("/document_index/index", "document_index_alt", protected(self.document_index))
        self.add_endpoint("/document_index/upload", "upload", protected(self.upload), methods=["POST"])
        self.add_endpoint("/document_index/delete/<file_hash>", "delete", protected(self.delete))
        self.add_endpoint(
            "/document_index/delete_source/<source_type>",
            "delete_source",
            protected(self.delete_source),
        )
        self.add_endpoint("/document_index/upload_url", "upload_url", protected(self.upload_url), methods=["POST"])
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
            if request.path.startswith("/api/"):
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
        return jsonify({"status": "OK"}, 200)

    def index(self):
        return redirect(url_for("document_index"))

    def document_index(self):
        # Ensure the catalog reflects the latest ingested resources
        try:
            self.catalog.refresh()
        except Exception as exc:
            logger.warning("Failed to refresh catalog before rendering index: %s", exc)

        # Seed all configured source types so empty sources still render a card.
        configured_sources = list((self.config.get("data_manager", {}) or {}).get("sources", {}).keys())
        sources_index = {name: [] for name in configured_sources}
        source_status = self._load_source_status()

        for source_hash in self.catalog.metadata_index.keys():
            metadata_source = self.catalog.get_metadata_for_hash(source_hash)
            if not isinstance(metadata_source, dict):
                logger.info("Metadata for hash %s missing or invalid; skipping", source_hash)
                continue

            source_type = metadata_source.get("source_type")
            if not source_type:
                logger.info("Metadata for hash %s missing source_type; skipping", source_hash)
                continue

            title = metadata_source.get("ticket_id") or metadata_source.get("url")
            if not title:
                title = (
                    metadata_source.get("display_name")
                    or metadata_source.get("file_name")
                    or source_hash
                )

            ts = metadata_source.get("modified_at") or metadata_source.get("created_at") or metadata_source.get("ingested_at") or ""
            sources_index.setdefault(source_type, []).append(
                {"hash": source_hash, "title": title, "ts": ts}
            )

        # sort each source list by timestamp (desc) then title
        def _sort_key(entry: dict):
            ts = entry.get("ts") or ""
            return (ts, entry.get("title") or "")

        for key, entries in sources_index.items():
            entries.sort(key=_sort_key, reverse=True)

        sorted_sources = sorted(sources_index.items(), key=lambda x: x[0])
        return render_template("document_index.html", sources_index=sorted_sources, source_status=source_status)

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
        """
        url = request.form.get("url")
        if url:
            logger.info("Uploading the following URL: %s", url)
            try:
                self.scraper_manager.collect_links(self.persistence, link_urls=[url])
                self.persistence.flush_index()
                self._update_source_status("links", state="idle", last_run=self._now_iso())
                added_to_urls = True
            except Exception as exc:
                logger.exception("Failed to upload URL: %s", exc)
                added_to_urls = False

            if added_to_urls:
                logger.info("URL uploaded successfully")
                self._notify_update()
                return jsonify({"status": "ok"})
            else:
                return jsonify({"error": "upload_failed", "detail": str(exc)}), 500
        else:
            return jsonify({"error": "missing_url"}), 400

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
                    iterable = list(self.catalog.iter_files())
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
                iterable = list(self.catalog.iter_files())
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
    for token in shlex.split(query):
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
