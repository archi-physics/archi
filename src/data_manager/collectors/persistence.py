from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Union

from src.data_manager.collectors.utils.catalog_postgres import PostgresCatalogService
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.data_manager.collectors.resource_base import BaseResource

logger = get_logger(__name__)


class PersistenceService:
    """Shared filesystem persistence for collected resources."""

    def __init__(self, data_path: Path | str, *, pg_config: Dict[str, Any]) -> None:
        self.data_path = Path(data_path)
        self.pg_config = pg_config

        self.catalog = PostgresCatalogService(self.data_path, pg_config=self.pg_config)

    def persist_resource(self, resource: "BaseResource", target_dir: Path, overwrite:bool = False) -> Path:
        """
        Write a resource and its metadata to disk,
        updating the catalog with the unique hash of the file and its metadata.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = resource.get_file_path(target_dir)
        
        # Check if file already exists
        file_existed = file_path.exists()
        
        if file_existed and not overwrite:
            logger.debug("Skipping existing resource %s -> %s", resource.get_hash(), file_path)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = resource.get_content()
            self._write_content(file_path, content)

        # Always update metadata in catalog (even if file already existed)
        metadata = resource.get_metadata()
        metadata_dict = None
        if metadata is not None:
            metadata_dict = self._normalise_metadata(metadata)
            if not metadata_dict:
                raise ValueError("Refusing to persist empty metadata payload")
        else:
            metadata_dict = {}

        # Ensure catalog size_bytes reflects actual persisted file size for all source types.
        try:
            metadata_dict["size_bytes"] = str(file_path.stat().st_size)
        except OSError as exc:
            logger.warning("Could not stat resource file %s for size_bytes: %s", file_path, exc)

        try:
            relative_path = file_path.relative_to(self.data_path).as_posix()
        except ValueError:
            relative_path = str(file_path)

        resource_hash = resource.get_hash()
        logger.debug(f"Stored resource {resource_hash} -> {file_path}")
        self.catalog.upsert_resource(resource_hash, relative_path, metadata_dict)

        return file_path
    
    def delete_resource(self, resource_hash:str, flush: bool = True) -> Path:
        """
        Delete a resource and its metadata from disk,
        updating the catalog accordingly.
        """
        try:
            stored_file = self.catalog.file_index[resource_hash]
        except KeyError as exc:
            raise ValueError(f"Resource hash {resource_hash} not found. {exc}") from exc

        file_path = Path(stored_file)
        if not file_path.is_absolute():
            file_path = (self.data_path / file_path).resolve()

        self._delete_content(file_path)
        self.catalog.delete_resource(resource_hash)

        if flush:
            self.flush_index()

        logger.debug(f"Deleted resource {resource_hash} -> {file_path}")  
        return file_path
    
    def delete_by_metadata_filter(self, key: str, value: str) -> None:
        """
        Remove any resource matching the given metadata key-value pair.
        Removes the resource, metadata files, and wipes both indices accordingly.
        """
        to_remove = self.catalog.get_resource_hashes_by_metadata_filter(key, value)
        deleted = False
        for resource_hash in to_remove:
            self.delete_resource(resource_hash, flush=False)
            deleted = True
        if deleted:
            self.flush_index()

    def reset_directory(self, directory: Path) -> None:
        """Remove all files and folders within the specified directory."""
        if not directory.exists():
            return

        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            else:
                self._remove_tree(item)

        try:
            relative_prefix = directory.relative_to(self.data_path)
        except ValueError:
            relative_prefix = None

        if relative_prefix is not None:
            prefix_parts = relative_prefix.parts
            keys_to_remove = []
            for key, stored in self.catalog.file_index.items():
                stored_path = Path(stored)
                if stored_path.is_absolute():
                    try:
                        stored_path = stored_path.relative_to(self.data_path)
                    except ValueError:
                        continue
                if stored_path.parts[: len(prefix_parts)] == prefix_parts:
                    keys_to_remove.append(key)
            if keys_to_remove:
                for key in keys_to_remove:
                    self.catalog.delete_resource(key)
                self.flush_index()

    def flush_index(self) -> None:
        self.catalog.refresh()

    def _remove_tree(self, path: Path) -> None:
        for item in path.iterdir():
            if item.is_dir():
                self._remove_tree(item)
            else:
                item.unlink()
        path.rmdir()

    def _write_content(
        self,
        file_path: Path,
        content: Union[str, bytes, bytearray],
    ) -> None:
        if content is None:
            raise ValueError("Resource provided no content to persist")

        if isinstance(content, (bytes, bytearray)):
            payload = bytes(content)
            if not payload:
                raise ValueError("Refusing to persist empty binary content")
            file_path.write_bytes(payload)
            return

        if isinstance(content, str):
            if not content:
                raise ValueError("Refusing to persist empty textual content")
            file_path.write_text(content, encoding="utf-8")
            return

        raise TypeError(
            f"Unsupported content type {type(content)!r}; "
            "resources must return str or bytes"
        )

    def _delete_content(self,file_path: Path) -> None:
        file_path.unlink()

    @staticmethod
    def _normalise_metadata(metadata: Any) -> Dict[str, str]:
        if hasattr(metadata, "as_dict"):
            metadata_dict = metadata.as_dict()
        elif isinstance(metadata, dict):
            metadata_dict = metadata
        else:
            metadata_dict = {"value": str(metadata)}

        if not isinstance(metadata_dict, dict):
            raise TypeError("Metadata serialisation must produce a dictionary")

        sanitized: Dict[str, str] = {}
        for key, value in metadata_dict.items():
            if value is None:
                continue
            sanitized[str(key)] = str(value)
        return sanitized
