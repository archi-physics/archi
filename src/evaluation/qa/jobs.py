from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .artifacts import read_json, utc_now, write_json

PROCESS_TERMINATION_GRACE_SECONDS = 5.0
PROCESS_GROUP_POLL_SECONDS = 0.05


class JobConflictError(RuntimeError):
    pass


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    ATTENTION_REQUIRED = "attention_required"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELED = "canceled"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class EvaluationJobManager:
    """Persisted, single-flight execution for provider-backed QA work."""

    def __init__(self, jobs_dir: Path):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="archi-qa"
        )
        self._futures: Dict[str, Future] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._interrupt_stale_jobs()

    def _path(self, job_id: str) -> Path:
        try:
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError) as exc:
            raise LookupError("evaluation job not found") from exc
        return self.jobs_dir / f"{job_id}.json"

    def _interrupt_stale_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = read_json(path)
            except ValueError:
                continue
            if job.get("status") in {
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.CANCEL_REQUESTED.value,
            }:
                job["status"] = JobStatus.INTERRUPTED.value
                job["completed_at"] = utc_now()
                job["error"] = "service restarted before the job completed"
                write_json(path, job)

    def _active(self) -> Optional[Dict[str, Any]]:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = read_json(path)
            except ValueError:
                continue
            if job.get("status") in {
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.CANCEL_REQUESTED.value,
            }:
                return job
        return None

    def start(
        self,
        kind: str,
        work: Callable[[], Optional[Dict[str, Any]]],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if kind != "generate_atoms":
            raise ValueError("unsupported evaluation job kind")
        with self._lock:
            active = self._active()
            if active is not None:
                raise JobConflictError(
                    f"job {active['id']} is already {active['status']}"
                )
            job_id = str(uuid.uuid4())
            job = {
                "id": job_id,
                "kind": kind,
                "status": JobStatus.QUEUED.value,
                "created_at": utc_now(),
                "started_at": None,
                "completed_at": None,
                "context": context or {},
            }
            write_json(self._path(job_id), job)
            try:
                self._futures[job_id] = self._executor.submit(
                    self._execute, job_id, work
                )
            except Exception:
                job["status"] = JobStatus.FAILED.value
                job["completed_at"] = utc_now()
                job["error"] = "could not submit job"
                write_json(self._path(job_id), job)
                raise
        return job

    def start_process(
        self,
        request: Dict[str, Any],
        *,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Start one killable evaluation worker process."""
        with self._lock:
            active = self._active()
            if active is not None:
                raise JobConflictError(
                    f"job {active['id']} is already {active['status']}"
                )
            job_id = str(uuid.uuid4())
            job = {
                "id": job_id,
                "kind": "evaluation",
                "status": JobStatus.QUEUED.value,
                "created_at": utc_now(),
                "started_at": None,
                "completed_at": None,
                "context": context,
            }
            write_json(self._path(job_id), job)
            try:
                self._futures[job_id] = self._executor.submit(
                    self._execute_process, job_id, request
                )
            except Exception:
                job["status"] = JobStatus.FAILED.value
                job["completed_at"] = utc_now()
                job["error"] = "could not submit job"
                write_json(self._path(job_id), job)
                raise
            return job

    def continue_process(self, job_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Resume one persisted attention-required evaluation in the same job."""
        with self._lock:
            job = self.get(job_id)
            if (
                job["kind"] != "evaluation"
                or job["status"] != JobStatus.ATTENTION_REQUIRED.value
            ):
                raise JobConflictError("evaluation job is not awaiting attention")
            active = self._active()
            if active is not None:
                raise JobConflictError(
                    f"job {active['id']} is already {active['status']}"
                )
            job["status"] = JobStatus.QUEUED.value
            job["completed_at"] = None
            job.pop("result", None)
            job.pop("error", None)
            write_json(self._path(job_id), job)
            self._futures[job_id] = self._executor.submit(
                self._execute_process, job_id, request
            )
            return job

    def _execute(
        self, job_id: str, work: Callable[[], Optional[Dict[str, Any]]]
    ) -> None:
        with self._lock:
            job = read_json(self._path(job_id))
            job["status"] = JobStatus.RUNNING.value
            job["started_at"] = utc_now()
            write_json(self._path(job_id), job)
        try:
            result = work() or {}
            if not isinstance(result, dict):
                raise ValueError("job result must be an object")
            with self._lock:
                job = read_json(self._path(job_id))
                if job["status"] == JobStatus.CANCELED.value:
                    return
                job["status"] = JobStatus.COMPLETED.value
                job["completed_at"] = utc_now()
                job["result"] = result
                write_json(self._path(job_id), job)
        except Exception as exc:
            with self._lock:
                job = read_json(self._path(job_id))
                if job["status"] == JobStatus.CANCELED.value:
                    return
                job["status"] = JobStatus.FAILED.value
                job["completed_at"] = utc_now()
                job["error"] = str(exc)
                write_json(self._path(job_id), job)

    def _execute_process(self, job_id: str, request: Dict[str, Any]) -> None:
        result_path = self.jobs_dir / f".{job_id}.result.json"
        with self._lock:
            job = read_json(self._path(job_id))
            if job["status"] == JobStatus.CANCEL_REQUESTED.value:
                return
            job["status"] = JobStatus.RUNNING.value
            job["started_at"] = utc_now()
            write_json(self._path(job_id), job)
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "src.evaluation.qa.worker",
                        json.dumps(request, ensure_ascii=False, sort_keys=True),
                        str(result_path),
                    ],
                    start_new_session=True,
                )
            except Exception as exc:
                job["status"] = JobStatus.FAILED.value
                job["completed_at"] = utc_now()
                job["error"] = str(exc)
                write_json(self._path(job_id), job)
                return
            self._processes[job_id] = process

        return_code = process.wait()
        with self._lock:
            self._processes.pop(job_id, None)
            job = read_json(self._path(job_id))
            if job["status"] in {
                JobStatus.CANCEL_REQUESTED.value,
                JobStatus.CANCELED.value,
            }:
                self._remove_result(result_path)
                return
            try:
                envelope = read_json(result_path)
                if not isinstance(envelope, dict):
                    raise ValueError("worker result must be an object")
                if return_code != 0:
                    error = envelope.get("error")
                    if not isinstance(error, str) or not error:
                        error = f"evaluation worker exited with status {return_code}"
                    raise RuntimeError(error)
                if set(envelope) != {"result"} or not isinstance(
                    envelope["result"], dict
                ):
                    raise ValueError("worker result has an invalid shape")
            except Exception as exc:
                job["status"] = JobStatus.FAILED.value
                job["completed_at"] = utc_now()
                job["error"] = str(exc)
            else:
                job["result"] = envelope["result"]
                if (
                    envelope["result"].get("status")
                    == JobStatus.ATTENTION_REQUIRED.value
                ):
                    job["status"] = JobStatus.ATTENTION_REQUIRED.value
                    job["completed_at"] = None
                else:
                    job["status"] = JobStatus.COMPLETED.value
                    job["completed_at"] = utc_now()
            finally:
                self._remove_result(result_path)
            write_json(self._path(job_id), job)

    @staticmethod
    def _remove_result(result_path: Path) -> None:
        try:
            result_path.unlink()
        except FileNotFoundError:
            pass

    def _mark_canceled(self, job: Dict[str, Any]) -> None:
        job["status"] = JobStatus.CANCELED.value
        job["completed_at"] = job.get("completed_at") or utc_now()
        job.pop("error", None)
        job.pop("result", None)
        write_json(self._path(job["id"]), job)

    def _finish_cancellation(
        self,
        job: Dict[str, Any],
        on_terminated: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        job["completed_at"] = job.get("completed_at") or utc_now()
        write_json(self._path(job["id"]), job)
        if on_terminated is not None:
            on_terminated(job)
        self._mark_canceled(job)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        process_group_id = process.pid
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return

        deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process_group_id, 0)
            except ProcessLookupError:
                return
            time.sleep(PROCESS_GROUP_POLL_SECONDS)

        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        # The group leader may be reaped before a killed descendant has been
        # scheduled to its terminal state. Do not report cancellation while a
        # signal-resistant child can still execute.
        time.sleep(PROCESS_GROUP_POLL_SECONDS)

    def cancel(
        self,
        job_id: str,
        *,
        on_terminated: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            if job["kind"] != "evaluation":
                raise ValueError("only evaluation jobs can be canceled")
            if job["status"] == JobStatus.CANCELED.value:
                return job
            if job["status"] not in {
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.CANCEL_REQUESTED.value,
                JobStatus.ATTENTION_REQUIRED.value,
            }:
                raise JobConflictError(f"job {job_id} is already {job['status']}")
            job["status"] = JobStatus.CANCEL_REQUESTED.value
            write_json(self._path(job_id), job)
            process = self._processes.get(job_id)
            future = self._futures.get(job_id)
            canceled_before_start = future is not None and future.cancel()
            if canceled_before_start:
                self._finish_cancellation(job, on_terminated)
                return self.get(job_id)

        if process is not None:
            self._terminate(process)
        elif future is not None:
            future.result(timeout=5)

        with self._lock:
            terminal = self.get(job_id)
            if terminal["status"] == JobStatus.CANCEL_REQUESTED.value:
                self._finish_cancellation(terminal, on_terminated)
            return self.get(job_id)

    def get(self, job_id: str) -> Dict[str, Any]:
        path = self._path(job_id)
        if not path.is_file():
            raise LookupError("evaluation job not found")
        return read_json(path)

    def list(self) -> List[Dict[str, Any]]:
        jobs = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                jobs.append(read_json(path))
            except ValueError:
                continue
        return sorted(jobs, key=lambda job: job.get("created_at", ""), reverse=True)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.get(job_id)

    def close(self) -> None:
        for job_id in list(self._processes):
            try:
                self.cancel(job_id)
            except (JobConflictError, LookupError, ValueError):
                pass
        self._executor.shutdown(wait=False)
