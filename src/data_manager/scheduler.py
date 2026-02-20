from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from croniter import croniter

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CronJob:
    name: str
    cron: str
    callback: Callable[[], None]
    next_run: datetime = field(default_factory=datetime.now)

    def schedule_next(self, base_time: datetime) -> None:
        self.next_run = croniter(self.cron, base_time).get_next(datetime)


class CronScheduler:
    """Simple cron scheduler that runs jobs in a background thread.
    
    Supports dynamic schedule reloading via config polling or explicit reload.
    """

    def __init__(self, poll_interval: float = 1.0, config_poll_interval: float = 60.0) -> None:
        self.poll_interval = poll_interval
        self.config_poll_interval = config_poll_interval
        self.jobs: List[CronJob] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Config reload support
        self._config_loader: Optional[Callable[[], Dict[str, str]]] = None
        self._job_factory: Optional[Callable[[str], Callable[[], None]]] = None
        self._config_hash: Optional[str] = None
        self._last_config_check = datetime.now()

    def set_config_loader(
        self, 
        loader: Callable[[], Dict[str, str]], 
        job_factory: Callable[[str], Callable[[], None]]
    ) -> None:
        """Set callback to load schedules from database and create job callbacks.
        
        Args:
            loader: Function that returns {source_name: cron_expression}
            job_factory: Function that takes source_name and returns a callback
        """
        self._config_loader = loader
        self._job_factory = job_factory
        # Initialize hash with current config
        if loader:
            try:
                schedules = loader()
                self._config_hash = self._hash_config(schedules)
            except Exception as e:
                logger.warning("Failed to initialize config hash: %s", e)

    def _hash_config(self, schedules: Dict[str, str]) -> str:
        """Create a hash of the schedule config for change detection."""
        return hashlib.md5(json.dumps(schedules, sort_keys=True).encode()).hexdigest()

    def add_job(self, name: str, cron: str, callback: Callable[[], None]) -> None:
        with self._lock:
            job = CronJob(name=name, cron=cron, callback=callback)
            job.schedule_next(datetime.now())
            self.jobs.append(job)
            logger.info("Scheduled %s with cron '%s' (next run %s)", name, cron, job.next_run)

    def remove_job(self, name: str) -> bool:
        """Remove a job by name. Returns True if job was found and removed."""
        with self._lock:
            for job in self.jobs:
                if job.name == name:
                    self.jobs.remove(job)
                    logger.info("Removed scheduled job: %s", name)
                    return True
            return False

    def update_job(self, name: str, cron: str) -> bool:
        """Update a job's cron expression. Returns True if job was found and updated."""
        with self._lock:
            for job in self.jobs:
                if job.name == name:
                    if job.cron != cron:
                        job.cron = cron
                        job.schedule_next(datetime.now())
                        logger.info("Updated job %s: cron='%s' (next run %s)", name, cron, job.next_run)
                    return True
            return False

    def reload_schedules(self) -> Dict[str, str]:
        """Manually trigger a schedule reload. Returns the new schedules."""
        if not self._config_loader or not self._job_factory:
            logger.warning("Cannot reload schedules: config_loader or job_factory not set")
            return {}
        
        return self._check_for_config_changes(force=True)

    def _check_for_config_changes(self, force: bool = False) -> Dict[str, str]:
        """Check for config changes and reload if needed. Returns current schedules."""
        if not self._config_loader or not self._job_factory:
            return {}
        
        try:
            new_schedules = self._config_loader()
            new_hash = self._hash_config(new_schedules)
            
            if force or new_hash != self._config_hash:
                if not force:
                    logger.info("Schedule configuration changed, reloading jobs...")
                self._reload_jobs(new_schedules)
                self._config_hash = new_hash
            
            return new_schedules
        except Exception as e:
            logger.warning("Failed to check for config changes: %s", e)
            return {}

    def _reload_jobs(self, new_schedules: Dict[str, str]) -> None:
        """Reload all jobs with new schedules."""
        with self._lock:
            # Get current job names
            current_jobs = {job.name: job for job in self.jobs}
            
            # Remove jobs that no longer have schedules or are disabled
            for name in list(current_jobs.keys()):
                if name not in new_schedules or not new_schedules[name]:
                    self.jobs.remove(current_jobs[name])
                    logger.info("Removed scheduled job: %s", name)
            
            # Update or add jobs
            for name, cron in new_schedules.items():
                if not cron:
                    continue
                
                if name in current_jobs:
                    # Update existing job if cron changed
                    job = current_jobs[name]
                    if job.cron != cron:
                        job.cron = cron
                        job.schedule_next(datetime.now())
                        logger.info("Updated job %s: cron='%s' (next run %s)", name, cron, job.next_run)
                else:
                    # Add new job
                    try:
                        callback = self._job_factory(name)
                        job = CronJob(name=name, cron=cron, callback=callback)
                        job.schedule_next(datetime.now())
                        self.jobs.append(job)
                        logger.info("Added scheduled job %s: cron='%s' (next run %s)", name, cron, job.next_run)
                    except Exception as e:
                        logger.warning("Failed to create job for %s: %s", name, e)

    def get_job_status(self) -> List[Dict[str, str]]:
        """Get status of all scheduled jobs."""
        with self._lock:
            return [
                {
                    "name": job.name,
                    "cron": job.cron,
                    "next_run": job.next_run.isoformat()
                }
                for job in self.jobs
            ]

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now()
            
            # Check for config changes periodically
            if self._config_loader and (now - self._last_config_check).total_seconds() > self.config_poll_interval:
                self._check_for_config_changes()
                self._last_config_check = now
            
            next_wake = None

            with self._lock:
                jobs_snapshot = list(self.jobs)
            
            for job in jobs_snapshot:
                if job.next_run <= now:
                    logger.info("Running scheduled job %s", job.name)
                    try:
                        job.callback()
                    except Exception as exc:
                        logger.warning("Scheduled job %s failed: %s", job.name, exc)
                    job.schedule_next(now)

                if next_wake is None or job.next_run < next_wake:
                    next_wake = job.next_run

            if next_wake:
                sleep_for = max(0.0, (next_wake - datetime.now()).total_seconds())
                time.sleep(min(self.poll_interval, sleep_for))
            else:
                time.sleep(self.poll_interval)
