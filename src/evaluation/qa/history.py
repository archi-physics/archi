# isort: skip_file
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import ijson

from .artifacts import iter_jsonl, read_json, read_jsonl, verify_hashes
from .constants import (
    ATTEMPT_LIFECYCLE_STATUSES_BY_SCHEMA,
    LEGACY_RUN_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
)
from .preparation import load_preparation_records, load_preparation_rows
from .schema import CanceledRunRecord, ConsoleMetadata, RunManifest
from .tool_traces import serialize_tool_call_records

LEGACY_SCHEMA_VERSION = "qa-v0"
_PREPARATION_FILES = {
    LEGACY_SCHEMA_VERSION: ("prepared_items.jsonl", "preparation_results.jsonl"),
    "qa-v1": ("preparation.jsonl",),
    SCHEMA_VERSION: ("preparation.jsonl",),
}


def _history_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]


def _require_exact_fields(
    row: Dict[str, Any], expected: Set[str], *, context: str
) -> None:
    missing = sorted(expected - set(row))
    unknown = sorted(set(row) - expected)
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unknown:
        details.append("unknown: " + ", ".join(unknown))
    if details:
        raise ValueError(f"{context} has invalid fields ({'; '.join(details)})")


class EvaluationHistory:
    """Read-only projection over persisted QA run artifacts."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _run_paths(self) -> List[Path]:
        root = self.runs_dir.resolve()
        return [
            path
            for path in self.runs_dir.iterdir()
            if path.is_dir() and not path.is_symlink() and path.resolve().parent == root
        ]

    def _resolve(self, history_id: str) -> Path:
        if not isinstance(history_id, str) or len(history_id) != 24:
            raise LookupError("evaluation run not found")
        for path in self._run_paths():
            if _history_id(path) == history_id:
                return path
        raise LookupError("evaluation run not found")

    def id_for_path(self, path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved.parent != self.runs_dir.resolve() or not resolved.is_dir():
            raise ValueError("run path is outside the evaluation history")
        return _history_id(resolved)

    def run_path(self, history_id: str) -> Path:
        return self._resolve(history_id)

    def get_run_header(self, history_id: str) -> Dict[str, Any]:
        """Return retry/navigation metadata without hydrating run artifacts."""
        path = self._resolve(history_id)
        manifest = self._load_manifest(path)
        metadata = self._load_console_metadata(path, manifest)
        return {
            "manifest": manifest.to_dict(),
            "metadata": metadata.to_dict() if metadata is not None else {},
            "capabilities": self._capabilities(manifest.schema_version),
        }

    @staticmethod
    def _preparation_files(schema_version: Any) -> Tuple[str, ...]:
        try:
            return _PREPARATION_FILES[schema_version]
        except (KeyError, TypeError) as exc:
            raise ValueError("unsupported run schema") from exc

    @staticmethod
    def _capabilities(schema_version: str) -> Dict[str, bool]:
        return {
            "retry_failed": schema_version
            not in {LEGACY_SCHEMA_VERSION, *LEGACY_RUN_SCHEMA_VERSIONS[:1]}
        }

    @staticmethod
    def _load_manifest(path: Path) -> RunManifest:
        raw_manifest = read_json(path / "manifest.json")
        if not isinstance(raw_manifest, dict):
            raise ValueError("manifest must be an object")
        preparation_files = EvaluationHistory._preparation_files(
            raw_manifest.get("schema_version")
        )
        return RunManifest.from_dict(
            raw_manifest,
            supported_schema_versions=_PREPARATION_FILES,
            preparation_files=preparation_files,
        )

    @staticmethod
    def _load_console_metadata(
        path: Path, manifest: RunManifest
    ) -> Optional[ConsoleMetadata]:
        filename = "console_metadata.json"
        metadata_path = path / filename
        if filename not in manifest.artifacts:
            if metadata_path.is_file():
                raise ValueError(f"run contains undeclared artifact: {filename}")
            return None
        verify_hashes(path, manifest.artifacts, [filename])
        return ConsoleMetadata.from_dict(read_json(metadata_path))

    @staticmethod
    def _load_canceled(path: Path) -> Optional[CanceledRunRecord]:
        canceled_path = path / "canceled.json"
        if not canceled_path.is_file():
            return None
        return CanceledRunRecord.from_dict(read_json(canceled_path))

    @staticmethod
    def _canceled_row(history_id: str, canceled: CanceledRunRecord) -> Dict[str, Any]:
        metadata = canceled.metadata
        return {
            "id": history_id,
            "run_id": canceled.run_id,
            "name": metadata.name or canceled.run_id,
            "status": "canceled",
            "created_at": metadata.created_at or canceled.canceled_at,
            "dataset_id": metadata.dataset_id,
            "dataset_key": metadata.dataset_id or f"canceled:{canceled.run_id}",
            "dataset_name": metadata.dataset_name or "CLI snapshot",
            "profile_id": metadata.profile_id,
            "profile_name": metadata.profile_name,
            "agent_spec": metadata.agent_spec,
            "attempts": canceled.attempts,
            "retry_of_history_id": metadata.retry_of_history_id,
            "retry_number": metadata.retry_number,
            "overall_attempt_pass_rate": None,
            "passed_attempts": None,
            "quality_accounted_attempts": None,
            "attempt_lifecycle_counts": None,
            "technical_failure_rate": None,
            "latency": None,
            "schema_version": SCHEMA_VERSION,
            "capabilities": {"retry_failed": False},
            "valid": True,
        }

    @staticmethod
    def _legacy_preparation_rows(path: Path) -> Iterator[Dict[str, Any]]:
        metadata_fields = {
            "item_id",
            "category",
            "answer_mode",
            "answer_source",
        }
        prepared_fields = metadata_fields | {
            "question",
            "answer",
            "time_sensitive",
            "atom_source",
            "gold_atoms",
        }
        prepared_by_id: Dict[str, Dict[str, Any]] = {}
        for index, row in enumerate(iter_jsonl(path / "prepared_items.jsonl"), 1):
            _require_exact_fields(
                row, prepared_fields, context=f"legacy prepared row {index}"
            )
            item_id = row["item_id"]
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(f"legacy prepared row {index} has an invalid item ID")
            if item_id in prepared_by_id:
                raise ValueError(
                    "legacy preparation contains duplicate prepared item IDs"
                )
            prepared_by_id[item_id] = row

        seen_results = set()
        for index, result in enumerate(
            iter_jsonl(path / "preparation_results.jsonl"), 1
        ):
            status = result.get("status")
            status_fields = {
                "prepared": set(),
                "preparation_failed": {"error"},
                "skipped_time_sensitive": set(),
            }
            if not isinstance(status, str) or status not in status_fields:
                raise ValueError(
                    f"legacy preparation result row {index} has an unsupported status"
                )
            _require_exact_fields(
                result,
                metadata_fields | {"status"} | status_fields[status],
                context=f"legacy preparation result row {index}",
            )
            item_id = result["item_id"]
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(
                    f"legacy preparation result row {index} has an invalid item ID"
                )
            if item_id in seen_results:
                raise ValueError(
                    "legacy preparation contains duplicate result item IDs"
                )
            seen_results.add(item_id)
            if status != "prepared":
                yield result
                continue

            try:
                prepared = prepared_by_id.pop(item_id)
            except KeyError as exc:
                raise ValueError(
                    "legacy prepared result has no matching prepared item"
                ) from exc
            if any(
                prepared[field] != result[field]
                for field in metadata_fields - {"item_id"}
            ):
                raise ValueError("legacy preparation metadata is inconsistent")
            yield {**prepared, "status": "prepared"}

        if prepared_by_id:
            raise ValueError("legacy prepared item has no matching preparation result")

    @staticmethod
    def _load_preparation(path: Path, manifest: RunManifest) -> List[Dict[str, Any]]:
        expected_count = manifest.preparation_input_items
        if manifest.schema_version == LEGACY_SCHEMA_VERSION:
            records = load_preparation_rows(
                EvaluationHistory._legacy_preparation_rows(path),
                expected_count=expected_count,
            )
        else:
            records = load_preparation_records(
                path / "preparation.jsonl", expected_count=expected_count
            )
        return [record.to_dict() for record in records]

    @staticmethod
    def _verify_present(
        path: Path, manifest: RunManifest, filenames: List[str]
    ) -> None:
        present = [name for name in filenames if (path / name).is_file()]
        artifacts = manifest.artifacts
        undeclared = sorted(name for name in present if name not in artifacts)
        if undeclared:
            raise ValueError(
                "run contains undeclared artifact(s): " + ", ".join(undeclared)
            )
        declared_or_present = [
            name for name in filenames if name in artifacts or (path / name).is_file()
        ]
        verify_hashes(path, artifacts, declared_or_present)

    @staticmethod
    def _dataset_identity(
        manifest: RunManifest, metadata: Optional[ConsoleMetadata]
    ) -> Tuple[str, str]:
        dataset_id = metadata.dataset_id if metadata is not None else None
        snapshot = manifest.snapshot
        dataset_key = dataset_id or f"snapshot:{manifest.artifacts[snapshot]}"

        dataset_name = metadata.dataset_name if metadata is not None else None
        if dataset_name is not None:
            return dataset_key, dataset_name.strip()

        source_path = manifest.source_path
        if source_path is None:
            return dataset_key, "CLI snapshot"
        basename = source_path.replace("\\", "/").rsplit("/", 1)[-1].strip()
        return dataset_key, basename or "CLI snapshot"

    @staticmethod
    def _summary_projection(path: Path) -> Dict[str, Any]:
        """Read compact trend fields without hydrating dataset-sized items."""
        summary: Dict[str, Any] = {}
        for field in (
            "overall_attempt_pass_rate",
            "passed_attempts",
            "quality_accounted_attempts",
            "attempt_lifecycle_counts",
        ):
            with path.open("rb") as handle:
                values = ijson.items(handle, field)
                try:
                    value = next(values)
                    summary[field] = (
                        float(value) if isinstance(value, Decimal) else value
                    )
                except StopIteration:
                    continue
        return summary

    @staticmethod
    def _optional_count(summary: Dict[str, Any], field: str) -> Optional[int]:
        value = summary.get(field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"summary {field} must be a non-negative integer")
        return value

    @staticmethod
    def _timestamp_sort_value(value: Any, *, context: str) -> Optional[float]:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError(f"{context} must be a string")
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                f"{context} must be an ISO-8601 timestamp with a timezone"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{context} must be an ISO-8601 timestamp with a timezone")
        return parsed.timestamp()

    @staticmethod
    def _summary_trends(
        summary: Any, schema_version: str = SCHEMA_VERSION
    ) -> Dict[str, Any]:
        if summary is None:
            return {
                "overall_attempt_pass_rate": None,
                "passed_attempts": None,
                "quality_accounted_attempts": None,
                "attempt_lifecycle_counts": None,
                "technical_failure_rate": None,
            }
        if not isinstance(summary, dict):
            raise ValueError("summary must be an object")

        pass_rate = summary.get("overall_attempt_pass_rate")
        if pass_rate is not None and (
            isinstance(pass_rate, bool)
            or not isinstance(pass_rate, (int, float))
            or not 0 <= pass_rate <= 1
        ):
            raise ValueError("summary overall_attempt_pass_rate must be from 0 to 1")
        if pass_rate is not None:
            pass_rate = float(pass_rate)

        passed_attempts = EvaluationHistory._optional_count(summary, "passed_attempts")
        quality_attempts = EvaluationHistory._optional_count(
            summary, "quality_accounted_attempts"
        )
        if (passed_attempts is None) != (quality_attempts is None):
            raise ValueError(
                "summary pass counts must be present or unavailable together"
            )
        if passed_attempts is not None:
            if passed_attempts > quality_attempts:
                raise ValueError(
                    "summary passed_attempts cannot exceed quality_accounted_attempts"
                )
            if quality_attempts == 0 and pass_rate is not None:
                raise ValueError(
                    "summary pass rate must be unavailable for a zero denominator"
                )
            if quality_attempts and pass_rate is not None:
                expected_pass_rate = passed_attempts / quality_attempts
                if abs(pass_rate - expected_pass_rate) > 1e-12:
                    raise ValueError(
                        "summary pass rate does not match its attempt counts"
                    )

        lifecycle = summary.get("attempt_lifecycle_counts")
        technical_failure_rate = None
        if lifecycle is not None:
            if not isinstance(lifecycle, dict):
                raise ValueError("summary attempt_lifecycle_counts must be an object")
            try:
                lifecycle_statuses = ATTEMPT_LIFECYCLE_STATUSES_BY_SCHEMA[
                    schema_version
                ]
            except KeyError as exc:
                raise ValueError("unsupported run schema") from exc
            required_statuses = set(lifecycle_statuses)
            missing = sorted(required_statuses - set(lifecycle))
            unknown = sorted(set(lifecycle) - required_statuses)
            if missing or unknown:
                raise ValueError("summary attempt_lifecycle_counts has invalid fields")
            validated_lifecycle = {}
            for status in lifecycle_statuses:
                count = EvaluationHistory._optional_count(lifecycle, status)
                if count is None:
                    raise ValueError(
                        f"summary attempt_lifecycle_counts.{status} "
                        "must be a non-negative integer"
                    )
                validated_lifecycle[status] = count
            live_validation_failed = EvaluationHistory._optional_count(
                lifecycle, "live_validation_failed"
            )
            if "live_validation_failed" in lifecycle:
                assert live_validation_failed is not None
                validated_lifecycle["live_validation_failed"] = live_validation_failed
            lifecycle = validated_lifecycle
            terminal_attempts = sum(
                lifecycle[status]
                for status in lifecycle_statuses
                if status != "live_validation_failed"
            )
            if terminal_attempts:
                technical_failure_rate = (
                    lifecycle["execution_failed"] + lifecycle["evaluation_failed"]
                ) / terminal_attempts
            if quality_attempts is not None and quality_attempts != (
                lifecycle["scored"] + lifecycle["execution_failed"]
            ):
                raise ValueError(
                    "summary quality-accounted count does not match lifecycle counts"
                )
            if passed_attempts is not None and passed_attempts > lifecycle["scored"]:
                raise ValueError(
                    "summary passed-attempt count cannot exceed scored attempts"
                )

        projection = {
            "overall_attempt_pass_rate": pass_rate,
            "passed_attempts": passed_attempts,
            "quality_accounted_attempts": quality_attempts,
            "attempt_lifecycle_counts": lifecycle,
            "technical_failure_rate": technical_failure_rate,
        }
        if lifecycle is not None and "live_validation_failed" in lifecycle:
            projection["live_validation_failed_attempts"] = lifecycle[
                "live_validation_failed"
            ]
        return projection

    @staticmethod
    def _latency_trend(path: Path, manifest: RunManifest) -> Optional[Dict[str, Any]]:
        filename = "answers.jsonl"
        artifact = path / filename
        if filename not in manifest.artifacts and not artifact.is_file():
            return None
        EvaluationHistory._verify_present(path, manifest, [filename])

        total_attempts = 0
        timed_attempts = 0
        duration_sum = 0
        best_ms = None
        worst_ms = None
        for index, answer in enumerate(iter_jsonl(artifact), 1):
            total_attempts += 1
            if "duration_ms" not in answer:
                continue
            duration_ms = answer["duration_ms"]
            if (
                isinstance(duration_ms, bool)
                or not isinstance(duration_ms, int)
                or duration_ms < 0
            ):
                raise ValueError(
                    f"answer row {index} duration_ms must be a non-negative integer"
                )
            timed_attempts += 1
            duration_sum += duration_ms
            best_ms = duration_ms if best_ms is None else min(best_ms, duration_ms)
            worst_ms = duration_ms if worst_ms is None else max(worst_ms, duration_ms)
        return {
            "total_attempts": total_attempts,
            "timed_attempts": timed_attempts,
            "average_ms": duration_sum / timed_attempts if timed_attempts else None,
            "best_ms": best_ms,
            "worst_ms": worst_ms,
        }

    def list_runs(self, *, cutoff: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if cutoff is not None:
            if not isinstance(cutoff, datetime):
                raise ValueError("cutoff must be a datetime")
            if cutoff.tzinfo is None or cutoff.utcoffset() is None:
                raise ValueError("cutoff must include a timezone")
            cutoff_sort_value = cutoff.astimezone(timezone.utc).timestamp()
        else:
            cutoff_sort_value = None
        rows: List[Tuple[float, Dict[str, Any]]] = []
        for path in self._run_paths():
            history_id = _history_id(path)
            try:
                canceled = self._load_canceled(path)
                if canceled is not None:
                    row = self._canceled_row(history_id, canceled)
                    sort_value = self._timestamp_sort_value(
                        row["created_at"], context="canceled run creation timestamp"
                    )
                    assert sort_value is not None
                    if cutoff_sort_value is None or sort_value >= cutoff_sort_value:
                        rows.append((sort_value, row))
                    continue
                manifest = self._load_manifest(path)
                metadata = self._load_console_metadata(path, manifest)
                phase_timestamps = []
                for phase_name, phase in manifest.phases.items():
                    if not isinstance(phase, dict):
                        continue
                    value = phase.get("completed_at") or phase.get("started_at")
                    sort_value = self._timestamp_sort_value(
                        value,
                        context=f"manifest {phase_name} phase timestamp",
                    )
                    if sort_value is not None:
                        phase_timestamps.append((sort_value, value))
                metadata_created_at = (
                    metadata.created_at if metadata is not None else None
                )
                metadata_sort_value = self._timestamp_sort_value(
                    metadata_created_at,
                    context="run creation timestamp",
                )
                if metadata_sort_value is not None:
                    created_at = metadata_created_at
                    sort_value = metadata_sort_value
                elif phase_timestamps:
                    sort_value, created_at = max(phase_timestamps)
                else:
                    created_at = ""
                    sort_value = float("-inf")
                if (
                    cutoff_sort_value is not None
                    and created_at
                    and sort_value < cutoff_sort_value
                ):
                    continue
                summary = (
                    self._summary_projection(path / "summary.json")
                    if (path / "summary.json").is_file()
                    else None
                )
                self._verify_present(
                    path,
                    manifest,
                    ["summary.json"],
                )
                dataset_key, dataset_name = self._dataset_identity(manifest, metadata)
                trend_summary = self._summary_trends(summary, manifest.schema_version)
                latency = self._latency_trend(path, manifest)
                row = {
                    "id": history_id,
                    "run_id": manifest.run_id,
                    "name": (metadata.name if metadata is not None else None)
                    or manifest.run_id,
                    "status": manifest.status.value,
                    "created_at": created_at,
                    "dataset_id": (
                        metadata.dataset_id if metadata is not None else None
                    ),
                    "dataset_key": dataset_key,
                    "dataset_name": dataset_name,
                    "profile_id": (
                        metadata.profile_id if metadata is not None else None
                    ),
                    "profile_name": (
                        metadata.profile_name if metadata is not None else None
                    ),
                    "agent_spec": (
                        metadata.agent_spec if metadata is not None else None
                    ),
                    "attempts": manifest.attempts,
                    "retry_of_history_id": (
                        metadata.retry_of_history_id if metadata is not None else None
                    ),
                    "retry_number": (
                        metadata.retry_number if metadata is not None else None
                    ),
                    **trend_summary,
                    "latency": latency,
                    "schema_version": manifest.schema_version,
                    "capabilities": self._capabilities(manifest.schema_version),
                    "valid": True,
                }
                if manifest.schema_version == SCHEMA_VERSION:
                    row["contains_live_answers"] = bool(
                        manifest.extra.get("contains_live_answers")
                    )
                rows.append(
                    (
                        sort_value,
                        row,
                    )
                )
            except Exception as exc:
                rows.append(
                    (
                        float("-inf"),
                        {
                            "id": history_id,
                            "name": path.name,
                            "status": "invalid",
                            "created_at": "",
                            "valid": False,
                            "error": str(exc),
                        },
                    )
                )
        return [
            row
            for _sort_value, row in sorted(rows, reverse=True, key=lambda item: item[0])
        ]

    def get_run(self, history_id: str) -> Dict[str, Any]:
        path = self._resolve(history_id)
        canceled = self._load_canceled(path)
        if canceled is not None:
            return {
                "id": history_id,
                "manifest": {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": canceled.run_id,
                    "status": "canceled",
                    "phases": {},
                },
                "metadata": canceled.metadata.to_dict(),
                "capabilities": {"retry_failed": False},
                "cancellation": {"canceled_at": canceled.canceled_at},
                "prepared_items": [],
                "answers": [],
                "evaluation_results": [],
                "report_available": False,
            }
        manifest = self._load_manifest(path)
        detail_names = {
            "summary.json",
            "answers.jsonl",
            "live_checks.jsonl",
            "evaluation_results.jsonl",
            *self._preparation_files(manifest.schema_version),
        }
        detail_size = sum(
            (path / name).stat().st_size
            for name in detail_names
            if (path / name).is_file()
        )
        if detail_size > 25 * 1024 * 1024:
            raise ValueError("run detail exceeds the 25 MB limit")
        metadata = self._load_console_metadata(path, manifest)
        snapshot = manifest.snapshot
        preparation_files = self._preparation_files(manifest.schema_version)
        self._verify_present(
            path,
            manifest,
            [
                snapshot,
                "summary.json",
                *preparation_files,
                *(
                    ("live_checks.jsonl",)
                    if manifest.schema_version == SCHEMA_VERSION
                    else ()
                ),
                "answers.jsonl",
                "evaluation_results.jsonl",
                "report.md",
            ],
        )
        payload: Dict[str, Any] = {
            "id": history_id,
            "manifest": manifest.to_dict(),
            "metadata": metadata.to_dict() if metadata is not None else {},
            "capabilities": self._capabilities(manifest.schema_version),
        }
        preparation = self._load_preparation(path, manifest)
        payload["preparation"] = preparation
        payload["prepared_items"] = [
            record for record in preparation if record["status"] == "prepared"
        ]
        for filename, key in (
            ("summary.json", "summary"),
            ("answers.jsonl", "answers"),
            ("live_checks.jsonl", "live_checks"),
            ("evaluation_results.jsonl", "evaluation_results"),
        ):
            artifact = path / filename
            if artifact.is_file():
                payload[key] = (
                    read_json(artifact)
                    if filename.endswith(".json")
                    else read_jsonl(artifact)
                )
        for index, answer in enumerate(payload.get("answers", []), 1):
            if not isinstance(answer, dict):
                raise ValueError(f"answer row {index} must be an object")
            if "tool_calls" in answer:
                answer["tool_calls"] = serialize_tool_call_records(
                    answer["tool_calls"],
                    context=f"answer row {index}.tool_calls",
                )
        report = path / "report.md"
        payload["report_available"] = report.is_file()
        return payload

    def get_report(self, history_id: str) -> str:
        path = self._resolve(history_id)
        if self._load_canceled(path) is not None:
            raise LookupError("evaluation report not found")
        manifest = self._load_manifest(path)
        report = path / "report.md"
        if not report.is_file():
            raise LookupError("evaluation report not found")
        self._verify_present(path, manifest, ["report.md"])
        return report.read_text(encoding="utf-8")
