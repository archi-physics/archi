#!/bin/python
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

from flask import Flask, jsonify

from src.data_manager.data_manager import DataManager
from src.data_manager.scheduler import CronScheduler
from src.interfaces.uploader_app.app import FlaskAppWrapper
from src.utils.runtime_config_loader import load_runtime_config
from src.utils.env import read_secret
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    # set API keys in env for downstream clients
    os.environ["ANTHROPIC_API_KEY"] = read_secret("ANTHROPIC_API_KEY")
    os.environ["OPENAI_API_KEY"] = read_secret("OPENAI_API_KEY")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = read_secret("HUGGING_FACE_HUB_TOKEN")

    runtime_config = load_runtime_config()
    services_config = runtime_config["services"]
    data_manager_cfg = services_config.get("data_manager", {})
    status_file = Path(runtime_config["global"]["DATA_PATH"]) / "ingestion_status.json"

    data_manager = DataManager(run_ingestion=False)
    lock = threading.RLock()

    def load_status() -> Dict[str, Dict[str, str]]:
        if not status_file.exists():
            return {}
        try:
            return json.loads(status_file.read_text())
        except Exception:
            return {}

    def save_status(data: Dict[str, Dict[str, str]]) -> None:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(json.dumps(data))

    def set_source_status(source: str, *, state: str | None = None, last_run: str | None = None) -> None:
        data = load_status()
        entry = data.get(source, {})
        if state is not None:
            entry["state"] = state
        if last_run is not None:
            entry["last_run"] = last_run
        data[source] = entry
        save_status(data)

    def run_locked(name: str, func: Callable[[], None]) -> None:
        with lock:
            logger.info("Running ingestion task: %s", name)
            set_source_status(name, state="running")
            func()
            set_source_status(name, state="idle", last_run=datetime.now(timezone.utc).isoformat())

    def trigger_update() -> None:
        with lock:
            data_manager.update_vectorstore(force=True)

    schedule_map: Dict[str, Callable[[Optional[str]], None]] = {
        "local_files": lambda last_run=None: data_manager.localfile_manager.schedule_collect_local_files(data_manager.persistence, last_run=last_run),
        "links": lambda last_run=None: data_manager.scraper_manager.schedule_collect_links(data_manager.persistence, last_run=last_run),
        "git": lambda last_run=None: data_manager.scraper_manager.schedule_collect_git(data_manager.persistence, last_run=last_run),
        "sso": lambda last_run=None: data_manager.scraper_manager.schedule_collect_sso(data_manager.persistence, last_run=last_run),
        "jira": lambda last_run=None: data_manager.ticket_manager.schedule_collect_jira(data_manager.persistence, last_run=last_run),
        "redmine": lambda last_run=None: data_manager.ticket_manager.schedule_collect_redmine(data_manager.persistence, last_run=last_run),
    }

    scheduler = CronScheduler()
    sources_cfg = runtime_config.get("data_manager", {}).get("sources", {}) or {}
    # seed status with schedules
    initial_status = load_status()
    for source_name, source_cfg in sources_cfg.items():
        if source_name not in schedule_map:
            continue
        schedule = (source_cfg or {}).get("schedule")
        if schedule:
            entry = initial_status.get(source_name, {})
            entry.setdefault("schedule", schedule)
            entry.setdefault("state", "idle")
            initial_status[source_name] = entry
            last_run = entry.get("last_run")
            scheduler.add_job(
                name=source_name,
                cron=schedule,
                callback=lambda name=source_name, last_run=last_run: run_locked(name, lambda: schedule_map[name](last_run)),
            )
    save_status(initial_status)

    if scheduler.jobs:
        scheduler.start()

    app = Flask(
        __name__,
        template_folder=data_manager_cfg.get("template_folder"),
        static_folder=data_manager_cfg.get("static_folder"),
    )

    ingestion_status: Dict[str, object] = {"state": "pending", "step": None, "error": None}

    def set_ingestion_status(state: str, *, step: str | None = None, error: str | None = None) -> None:
        with lock:
            ingestion_status.update({"state": state, "step": step, "error": error})

    def run_initial_ingestion_async() -> None:
        set_ingestion_status("running", step="initializing")
        try:
            with lock:
                data_manager.run_ingestion(progress_callback=lambda step: set_ingestion_status("running", step=step))
            set_ingestion_status("completed", step="done")
        except Exception as exc:
            logger.exception("Initial ingestion failed")
            set_ingestion_status("error", step="failed", error=str(exc))

    ingestion_thread = threading.Thread(target=run_initial_ingestion_async, name="ingestion-thread", daemon=True)
    ingestion_thread.start()

    uploader = FlaskAppWrapper(app, post_update_hook=trigger_update, status_file=status_file)
    def get_ingestion_status():
        with lock:
            return jsonify(dict(ingestion_status))

    app.add_url_rule("/api/ingestion/status", "ingestion_status", get_ingestion_status, methods=["GET"])

    uploader.run(
        debug=data_manager_cfg["flask_debug_mode"],
        port=data_manager_cfg["port"],
        host=data_manager_cfg["host"],
    )


if __name__ == "__main__":
    main()
