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
from src.utils.config_service import ConfigService
from src.utils.env import read_secret
from src.utils.logging import get_logger, setup_logging
from src.utils.postgres_service_factory import PostgresServiceFactory
from src.utils.config_access import get_full_config, get_services_config, get_global_config

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    # set API keys in env for downstream clients
    os.environ["ANTHROPIC_API_KEY"] = read_secret("ANTHROPIC_API_KEY")
    os.environ["OPENAI_API_KEY"] = read_secret("OPENAI_API_KEY")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = read_secret("HUGGING_FACE_HUB_TOKEN")

    factory = PostgresServiceFactory.from_env(password_override=read_secret("PG_PASSWORD"))
    PostgresServiceFactory.set_instance(factory)
    config = get_full_config()
    services_config = get_services_config()
    data_manager_cfg = services_config.get("data_manager", {})
    status_file = Path(get_global_config().get("DATA_PATH")) / "ingestion_status.json"

    data_manager = DataManager(run_ingestion=False, factory=factory)
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
    sources_cfg = config.get("data_manager", {}).get("sources", {}) or {}
    
    # Initialize ConfigService for database access
    config_service: Optional[ConfigService] = None
    try:
        pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **services_config["postgres"],
        }
        config_service = ConfigService(pg_config)
    except Exception as e:
        logger.warning("Could not initialize ConfigService: %s", e)
    
    def load_schedules_from_db() -> Dict[str, str]:
        """Load schedules from database, merging with YAML defaults."""
        db_schedules: Dict[str, str] = {}
        if config_service:
            try:
                db_schedules = config_service.get_source_schedules()
            except Exception as e:
                logger.warning("Could not load schedules from database: %s", e)
        
        # Merge YAML and database schedules (database takes priority)
        result: Dict[str, str] = {}
        all_sources = set(sources_cfg.keys()) | set(db_schedules.keys())
        for source_name in all_sources:
            if source_name not in schedule_map:
                continue
            
            # Database schedule takes priority over YAML
            if source_name in db_schedules and db_schedules[source_name]:
                result[source_name] = db_schedules[source_name]
            else:
                source_cfg = sources_cfg.get(source_name) or {}
                schedule = source_cfg.get("schedule")
                if schedule:
                    result[source_name] = schedule
        
        return result
    
    def create_job_callback(source_name: str) -> Callable[[], None]:
        """Create a callback function for a scheduled job."""
        def callback():
            status = load_status()
            last_run = status.get(source_name, {}).get("last_run")
            run_locked(source_name, lambda: schedule_map[source_name](last_run))
        return callback
    
    # Set up dynamic schedule reloading
    scheduler.set_config_loader(load_schedules_from_db, create_job_callback)
    
    # Load initial schedules from database
    db_schedules = load_schedules_from_db()
    logger.info("Loaded source schedules: %s", db_schedules)
    
    # seed status with schedules
    initial_status = load_status()
    
    for source_name, schedule in db_schedules.items():
        if schedule:
            entry = initial_status.get(source_name, {})
            entry.setdefault("schedule", schedule)
            entry.setdefault("state", "idle")
            initial_status[source_name] = entry
            scheduler.add_job(
                name=source_name,
                cron=schedule,
                callback=create_job_callback(source_name),
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

    def reload_schedules():
        """Trigger a reload of all schedules from the database."""
        try:
            new_schedules = scheduler.reload_schedules()
            # Update status file with new schedules
            status = load_status()
            for source_name, schedule in new_schedules.items():
                if source_name in status:
                    status[source_name]["schedule"] = schedule
            save_status(status)
            return jsonify({"success": True, "schedules": new_schedules, "jobs": scheduler.get_job_status()})
        except Exception as e:
            logger.exception("Failed to reload schedules")
            return jsonify({"success": False, "error": str(e)}), 500

    app.add_url_rule("/api/reload-schedules", "reload_schedules", reload_schedules, methods=["POST"])

    def get_schedule_status():
        """Get current schedule status for all jobs."""
        return jsonify({"jobs": scheduler.get_job_status()})

    app.add_url_rule("/api/schedules", "get_schedules", get_schedule_status, methods=["GET"])

    uploader.run(
        debug=data_manager_cfg["flask_debug_mode"],
        port=data_manager_cfg["port"],
        host=data_manager_cfg["host"],
    )


if __name__ == "__main__":
    main()
