from typing import Any, Dict, Iterator, Optional, List

from redminelib import Redmine

from src.data_manager.collectors.tickets.ticket_resource import TicketResource
from src.data_manager.collectors.utils.anonymizer import Anonymizer
from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RedmineClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.redmine = None
        self.redmine_url: Optional[str] = None
        self.redmine_user: Optional[str] = None
        self.redmine_pw: Optional[str] = None
        self.redmine_projects: Optional[List[str]] = None
        self.anonymizer: Optional[Anonymizer] = None
        self.visible: bool = True

        redmine_config = dict(config or {})
        if not redmine_config.get("enabled", False):
            logger.debug("Redmine source disabled; skipping ticket collection")
            return

        self.redmine_url = redmine_config.get("url")
        self.redmine_projects = redmine_config.get("projects", [])
        self.visible = bool(redmine_config.get("visible", True))
        if not self.redmine_url or not self.redmine_projects:
            logger.warning("Redmine config missing url/projects; skipping Redmine collection")
            return

        try:
            self.redmine_user = read_secret("REDMINE_USER")
            self.redmine_pw = read_secret("REDMINE_PW")
        except FileNotFoundError as error:
            logger.warning(
                "Redmine secrets couldn't be found. archi will skip data fetching from Redmine.",
                exc_info=error,
            )
            return

        anonymize_data = redmine_config.get("anonymize_data", True)
        if anonymize_data:
            try:
                self.anonymizer = Anonymizer()
            except Exception as error:
                logger.warning(
                    "Failed to initialise Redmine anonymizer; continuing without anonymization.",
                    exc_info=error,
                )
                self.anonymizer = None

        if not self._verify():
            self.redmine = None
            return

        try:
            self._connect()
        except Exception as error:
            logger.warning(
                "Failed to initialise Redmine client; skipping Redmine collection.",
                exc_info=error,
            )
            self.redmine = None

    def collect(self, projects: List[str], **kwargs) -> Iterator[TicketResource]:
        """Return an iterator of Redmine tickets."""
        logger.info(f"Collecting Redmine tickets from projects: {projects}")
        if not self._verify() or not self.redmine:
            logger.debug(
                "Skipping Redmine collection; client not initialised or credentials missing."
            )
            return iter(())

        for project in projects:
            yield from self._prepare_ticket_resources(project)

    def _prepare_ticket_resources(self, project: str) -> Iterator[TicketResource]:
        project = self._get_project(project)
        closed_issues = self._get_closed_issues(project)
        logger.info(f"Preparing {len(closed_issues)} redmine tickets' data")
        processed_count = 0

        for issue in closed_issues:
            try:
                full_issue = self.redmine.issue.get(issue.id)

                subject = (
                    self.anonymizer.anonymize(full_issue.subject)
                    if self.anonymizer
                    else full_issue.subject
                )

                description = (full_issue.description or "").replace("\n", " ")
                question = (
                    self.anonymizer.anonymize(description)
                    if self.anonymizer
                    else description
                )

                answer = self._extract_answer_from_journals(full_issue.journals)

                if answer and question != answer:
                    issue_id = str(full_issue.id)
                    content = self._format_ticket_content(issue_id, subject, question, answer)

                    metadata: Dict[str, Any] = {
                        "subject": subject,
                        "ticket_provider": "redmine",
                    }

                    created_at = getattr(full_issue, "created_on", None)
                    created_at_str = str(created_at) if created_at else None

                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.debug(f"Processed {processed_count} tickets so far...")

                    yield TicketResource(
                        ticket_id=issue_id,
                        content=content,
                        source_type="ticket",
                        created_at=created_at_str,
                        metadata=metadata,
                    )

            except Exception as error:
                logger.error(f"Error processing ticket {issue.id}: {error}")
                continue

        logger.info(f"Successfully processed {processed_count} redmine tickets")

    def _format_ticket_content(self, issue_id: str, subject: str, question: str, answer: str) -> str:
        lines = [
            f"Redmine issue ID/ticket number: {issue_id}",
            f"Subject: {subject}",
            f"Question: {question}",
            f"Answer: {answer}",
            "",
        ]
        return "\n".join(lines) + "\n"

    def _verify(self) -> bool:
        """Check if necessary secrets are provided to access Redmine."""
        if not all(
            [self.redmine_url, self.redmine_user, self.redmine_pw, self.redmine_projects]
        ):
            logger.debug(
                "Redmine configuration or credentials missing; skipping Redmine collection"
            )
            return False

        return True

    def _connect(self) -> None:
        """Open the redmine web site."""
        logger.info(
            f"Open redmine (URL:{self.redmine_url} U:{self.redmine_user} P:*********)"
        )
        self.redmine = Redmine(
            self.redmine_url, username=self.redmine_user, password=self.redmine_pw
        )

    def _get_project(self, project: str) -> None:
        """Load the project that is responsible to deal with email tickets."""
        return self.redmine.project.get(project)

    def _get_closed_issues(self, project) -> Any:
        return self.redmine.issue.filter(
            project_id=project.id,
            status_id="closed",
        )

    def _extract_answer_from_journals(self, journals: Any) -> str:
        """
        Takes Redmine journal, returns formatted and anonymized answer (most recent one for now)
        if exists, otherwise empty string (checked later)
        """
        answers = []
        answer_tag = _get_answer_tag()
        for record in journals[::-1]:
            note = record.notes
            if note:
                answer = "\n".join(line for line in note.splitlines() if "ISSUE_ID" not in line)
                answer = answer.replace("\n", " ")
                if self.anonymizer:
                    answer = self.anonymizer.anonymize(answer)
                answers.append(answer)

        return answers[-1] if answers else ""
