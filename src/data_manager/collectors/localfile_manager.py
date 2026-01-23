from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.data_manager.collectors.localfile_resource import LocalFileResource
from src.data_manager.collectors.persistence import PersistenceService
from src.utils.runtime_config_loader import load_runtime_global_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LocalFileManager:
    """Collects local files/directories into the shared data path."""

    def __init__(self, dm_config: Optional[Dict[str, Any]] = None) -> None:
        global_config = load_runtime_global_config()
        self.data_path = Path(global_config["DATA_PATH"])

        sources_config = (dm_config or {}).get("sources", {}) or {}
        self.config = dict(sources_config.get("local_files", {})) if isinstance(sources_config, dict) else {}

        self.enabled = self.config.get("enabled", True)
        base_dir = self.config.get("base_dir")
        self.base_dir: Optional[Path] = Path(base_dir).expanduser() if base_dir else None
        self.overwrite = bool(self.config.get("overwrite", True))
        self.staging_dir = Path(self.config.get("staging_dir") or (self.data_path / "raw_local_files"))

    def collect_all_from_config(self, persistence: PersistenceService) -> None:
        if not self.enabled:
            logger.info("Local files disabled; skipping")
            return
        source_root = self.staging_dir
        if not source_root.exists():
            logger.info("Local files directory does not exist: %s", source_root)
            return

        target_dir = self.data_path / "local_files"
        for file_path in self._iter_files(source_root):
            self._persist_file(file_path, persistence, target_dir, base_dir=self.base_dir or source_root)

    def schedule_collect_local_files(self, persistence: PersistenceService, last_run: Optional[str] = None) -> None:
        """For now simply re-run the configured collection."""
        self.collect_all_from_config(persistence)

    def ingest_uploaded_file(self, upload: FileStorage, persistence: PersistenceService) -> Path:
        """Persist a single uploaded file into the local_files source."""
        if not self.enabled:
            raise ValueError("Local files source is disabled")

        filename = secure_filename(upload.filename or "")
        if not filename:
            raise ValueError("No filename provided")

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        staging_path = self.staging_dir / filename
        upload.save(staging_path)

        target_dir = self.data_path / "local_files"
        return self._persist_file(staging_path, persistence, target_dir, base_dir=self.base_dir or self.staging_dir)

    # internal helpers

    def _iter_files(self, directory: Path) -> Iterable[Path]:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                yield file_path

    def _persist_file(self, path: Path, persistence: PersistenceService, target_dir: Path, *, base_dir: Optional[Path]) -> None:
        try:
            content = path.read_bytes()
        except Exception as exc:
            logger.warning("Failed to read local file %s: %s", path, exc)
            return

        resource = LocalFileResource(file_name=path.name, source_path=path, content=content, base_dir=base_dir)
        effective_target_dir = self._resolve_target_dir(path, target_dir, base_dir)
        try:
            persistence.persist_resource(resource, effective_target_dir, overwrite=self.overwrite)
        except Exception as exc:
            logger.warning("Failed to persist local file %s: %s", path, exc)

    def _resolve_target_dir(self, path: Path, target_dir: Path, base_dir: Optional[Path]) -> Path:
        if not base_dir:
            return target_dir
        try:
            relative_path = path.relative_to(base_dir)
        except ValueError:
            return target_dir
        if relative_path.parent == Path("."):
            return target_dir
        return target_dir / relative_path.parent
