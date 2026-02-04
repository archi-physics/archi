#!/usr/bin/env python3
"""Preflight checks for Postgres, data-manager, and Ollama."""
import os
import sys
import time
from typing import Dict, Optional

import requests
import yaml


def _fail(message: str) -> None:
    print(f"[preflight] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _info(message: str) -> None:
    print(f"[preflight] {message}")


def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        _fail(f"Missing required env var: {name}")
    return value


def _check_postgres() -> None:
    host = _get_env("PGHOST")
    port = int(os.getenv("PGPORT", "5432"))
    user = _get_env("PGUSER")
    password = _get_env("PGPASSWORD")
    database = _get_env("PGDATABASE")

    _info(f"Checking Postgres at {host}:{port}/{database} ...")
    try:
        import psycopg2  # type: ignore
    except Exception:
        psycopg2 = None

    if psycopg2 is None:
        import socket

        try:
            with socket.create_connection((host, port), timeout=5):
                pass
        except Exception as exc:
            _fail(f"Postgres TCP connection failed: {exc}")
        _info("Postgres reachable (TCP check only)")
        return

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=database,
            connect_timeout=5,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            _ = cur.fetchone()
        conn.close()
    except Exception as exc:
        _fail(f"Postgres connection failed: {exc}")
    _info("Postgres OK")


def _wait_for_ingestion() -> None:
    dm_base_url = os.getenv("DM_BASE_URL", "http://localhost:7871").rstrip("/")
    timeout = int(os.getenv("DM_INGESTION_TIMEOUT", "300"))
    interval = int(os.getenv("DM_INGESTION_INTERVAL", "5"))
    status_url = f"{dm_base_url}/api/ingestion/status"
    _info(f"Waiting for ingestion to complete at {status_url} (timeout {timeout}s) ...")
    deadline = time.time() + timeout
    while True:
        try:
            resp = requests.get(status_url, timeout=10)
            if resp.status_code != 200:
                _fail(f"Ingestion status check failed: HTTP {resp.status_code}")
            payload = resp.json()
            state = payload.get("state")
            if state == "completed":
                _info("Ingestion completed")
                return
            if state == "error":
                _fail(f"Ingestion failed: {payload.get('error')}")
        except Exception as exc:
            _info(f"Ingestion status check error: {exc}; retrying...")

        if time.time() >= deadline:
            _fail("Ingestion did not complete before timeout")
        time.sleep(interval)


def _check_data_manager_catalog() -> None:
    dm_base_url = os.getenv("DM_BASE_URL", "http://localhost:7871").rstrip("/")
    query = os.getenv("DM_CATALOG_QUERY", "file_name:seed.txt")
    timeout = int(os.getenv("DM_CATALOG_TIMEOUT", "120"))
    interval = int(os.getenv("DM_CATALOG_INTERVAL", "5"))
    seed_file = os.getenv("DM_CATALOG_SEED_FILE")
    search_url = f"{dm_base_url}/api/catalog/search"
    _info(f"Checking data-manager catalog at {search_url} ...")
    deadline = time.time() + timeout
    seeded = False
    while True:
        try:
            resp = requests.get(
                search_url,
                params={"q": query, "limit": 1, "search_content": "false"},
                timeout=10,
            )
            if resp.status_code != 200:
                _fail(f"Data-manager catalog search failed: HTTP {resp.status_code}")
            payload = resp.json()
            hits = payload.get("hits", [])
            if hits:
                _info("Data-manager catalog OK")
                return
            if seed_file:
                filename_query = os.path.basename(seed_file)
                resp = requests.get(
                    search_url,
                    params={"q": filename_query, "limit": 1, "search_content": "false"},
                    timeout=10,
                )
                if resp.status_code == 200 and (resp.json().get("hits") or []):
                    _info("Data-manager catalog OK (metadata search)")
                    return
        except Exception as exc:
            _info(f"Data-manager catalog check error: {exc}; retrying...")

        if seed_file and not seeded:
            _info(f"Catalog empty; seeding with {seed_file} ...")
            try:
                with open(seed_file, "rb") as handle:
                    files = {"file": (os.path.basename(seed_file), handle)}
                    resp = requests.post(
                        f"{dm_base_url}/document_index/upload",
                        files=files,
                        timeout=30,
                    )
                if resp.status_code != 200:
                    _fail(f"Data-manager seed upload failed: HTTP {resp.status_code}")
                seeded = True
            except Exception as exc:
                _fail(f"Data-manager seed upload failed: {exc}")

        if time.time() >= deadline:
            _fail("Data-manager catalog search returned no hits before timeout")
        _info("Catalog empty, waiting for ingestion...")
        time.sleep(interval)


def _check_ollama_model() -> None:
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    ollama_model = _get_env("OLLAMA_MODEL")
    tags_url = f"{ollama_url}/api/tags"
    _info(f"Checking Ollama model '{ollama_model}' at {tags_url} ...")
    try:
        resp = requests.get(tags_url, timeout=5)
        if resp.status_code != 200:
            _fail(f"Ollama tags request failed: HTTP {resp.status_code}")
        payload = resp.json()
    except Exception as exc:
        _fail(f"Ollama tags request failed: {exc}")

    models = payload.get("models", [])
    model_names = {item.get("name") or item.get("model") for item in models}
    if ollama_model not in model_names:
        _fail(f"Ollama model '{ollama_model}' not found in {sorted(model_names)}")
    _info("Ollama model OK")


def _check_config_ollama(config_path: str, pipeline_name: str, ollama_model: str) -> None:
    _info(f"Validating config at {config_path} uses Ollama for {pipeline_name} ...")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception as exc:
        _fail(f"Failed to read config at {config_path}: {exc}")

    pipeline_cfg = ((config.get("archi") or {}).get("pipeline_map") or {}).get(pipeline_name) or {}
    required_models = (pipeline_cfg.get("models") or {}).get("required") or {}
    agent_model = required_models.get("agent_model")
    if agent_model != "OllamaInterface":
        _fail(f"Pipeline {pipeline_name} agent_model is '{agent_model}', expected 'OllamaInterface'")

    model_map = (config.get("archi") or {}).get("model_class_map") or {}
    ollama_cfg = (model_map.get("OllamaInterface") or {}).get("kwargs") or {}
    base_model = ollama_cfg.get("base_model")
    if base_model and base_model != ollama_model:
        _fail(
            f"Ollama base_model '{base_model}' does not match OLLAMA_MODEL '{ollama_model}'"
        )
    _info("Config Ollama settings OK")


def main() -> None:
    _wait_for_ingestion()
    _check_postgres()
    # ChromaDB removed - PostgreSQL with pgvector is the only supported backend
    _check_data_manager_catalog()
    _check_ollama_model()

    config_path = os.getenv("ARCHI_CONFIG_PATH")
    pipeline_name = os.getenv("ARCHI_PIPELINE_NAME", "CMSCompOpsAgent")
    ollama_model = os.getenv("OLLAMA_MODEL", "")
    if config_path:
        _check_config_ollama(config_path, pipeline_name, ollama_model)
    else:
        _info("ARCHI_CONFIG_PATH not set; skipping config Ollama validation")

    _info("Preflight checks passed")


if __name__ == "__main__":
    main()
