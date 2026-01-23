from typing import Any, Dict, List, Optional
from pathlib import Path

from src.data_manager.collectors.persistence import PersistenceService
from src.data_manager.collectors.tickets.integrations.jira import JiraClient
from src.data_manager.collectors.tickets.integrations.redmine_tickets import \
    RedmineClient
from src.data_manager.collectors.tickets.ticket_resource import TicketResource
from src.utils.runtime_config_loader import load_runtime_global_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

global_config = load_runtime_global_config()

class TicketManager:
    """Coordinates ticket integrations and delegates persistence."""

    def __init__(self, dm_config: Optional[Dict[str, Any]] = None) -> None:
        self.data_path = Path(global_config["DATA_PATH"])
        raw_sources = (dm_config or {}).get('sources', {}) if isinstance(dm_config, dict) else {}
        sources_config = dict(raw_sources) if isinstance(raw_sources, dict) else {}

        self.jira_config = dict(sources_config.get('jira', {}))
        self.redmine_config = dict(sources_config.get('redmine', {}))

        self.jira_client = None
        if self.jira_config.get('enabled', False):
            self.jira_client = self._init_client(lambda: JiraClient(self.jira_config), "JIRA")

        self.redmine_client = None
        if self.redmine_config.get('enabled', False):
            self.redmine_client = self._init_client(lambda: RedmineClient(self.redmine_config), "Redmine")

        # cache the projects we have collected
        self.jira_projects = set()
        self.redmine_projects = set()

    def _init_client(self, factory, name: str):
        try:
            return factory()
        except Exception as exc:
            logger.warning(
                f"{name} client unavailable; skipping ticket collection.",
                exc_info=exc,
            )
            return None

    def collect_all_from_config(self, persistence: PersistenceService) -> None:
        if self.jira_client:
            jira_projects = self.jira_config.get("projects", [])
            self.collect_jira(persistence, projects=jira_projects)
        if self.redmine_client:
            redmine_projects = self.redmine_config.get("projects", [])
            self.collect_redmine(persistence, projects=redmine_projects)

    def collect_jira(
        self,
        persistence: PersistenceService,
        projects: List[str],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._collect_from_client(
            self.jira_client, "JIRA",
            persistence=persistence,
            projects=projects,
            **(kwargs or {})
        )

    def collect_redmine(
        self,
        persistence: PersistenceService,
        projects: List[str],
        **kwargs
    ) -> None:
        self._collect_from_client(
            self.redmine_client, "Redmine",
            persistence=persistence,
            projects=projects,
            **kwargs
        )

    def schedule_collect_jira(
        self,
        persistence: PersistenceService,
        last_run: Optional[str],
    ) -> None:
        """
        Update all JIRA projects with tickets since last run
        """

        self._collect_from_client(
            self.jira_client, "JIRA",
            persistence=persistence,
            projects=self.jira_projects,
            since_iso=last_run
        )

    def schedule_collect_redmine(
        self,
        persistence: PersistenceService,
        last_run: Optional[str],
    ) -> None:
        """
        Update all Redmine projects with tickets since last run
        """

        self._collect_from_client(
            self.redmine_client, "Redmine",
            persistence=persistence,
            projects=self.redmine_projects,
            since_iso=last_run
        )

    def _collect_from_client(
        self,
        client,
        name: str,
        persistence: PersistenceService,
        projects: List[str],
        **kwargs,
    ) -> None:
        if client is None:
            return
        try:
            resources = client.collect(projects=projects, **kwargs)
            if name == "JIRA":
                self.jira_projects.update(projects)
                outdir = self.data_path / "jira"
            elif name == "Redmine":
                self.redmine_projects.update(projects)
                outdir = self.data_path / "redmine"
        except Exception as exc:
            logger.warning(
                f"{name} collection failed; skipping remaining tickets from this source.",
                exc_info=exc,
            )
            return

        for resource in resources:
            persistence.persist_resource(resource, outdir)
