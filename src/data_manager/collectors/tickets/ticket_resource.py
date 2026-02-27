from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import re

from src.data_manager.collectors.resource_base import BaseResource
from src.data_manager.collectors.utils.metadata import ResourceMetadata


@dataclass
class TicketResource(BaseResource):
    """Standard representation of a collected support ticket."""

    ticket_id: str
    content: str
    source_type: str
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_hash(self) -> str:
        identifier = self._normalise_identifier(self.ticket_id)
        return f"{self.source_type}_{identifier}"

    def get_filename(self) -> str:
        return f"{self.get_hash()}.txt"

    def get_content(self) -> str:
        return self.content

    def get_metadata(self) -> ResourceMetadata:
        metadata_dict = self.metadata or {}
        display_name = metadata_dict.get("display_name") or metadata_dict.get("url")
        if not display_name:
            display_name = f"{self.source_type}:{self.ticket_id}"
        extra = {
            str(k): str(v)
            for k, v in metadata_dict.items()
            if k not in {"display_name"}
        }
        extra.setdefault("ticket_id", self.ticket_id)
        extra.setdefault("source_type", self.source_type)
        if self.created_at:
            extra.setdefault("created_at", str(self.created_at))
        if display_name:
            pattern = r'(\w+)\-\d+'
            display_name = re.search(pattern,display_name).group() if re.search(pattern,display_name) else display_name
            extra["display_name"] = str(display_name)

        return ResourceMetadata(file_name=self.get_filename(), extra=extra)

    @staticmethod
    def _normalise_identifier(identifier: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", identifier)
