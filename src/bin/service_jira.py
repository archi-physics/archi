#!/bin/python
import time

from src.interfaces.jira import JiraIssueClient
from src.services.jira_ticket_responder import archi_factory
from src.services.jira_ticket_responder import config as responder_config
from src.services.jira_ticket_responder import service as responder_service
from src.services.jira_ticket_responder import store as responder_store
from src.utils.config_access import get_services_config
from src.utils.env import read_secret
from src.utils.logging import get_logger, setup_logging
from src.utils.postgres_service_factory import PostgresServiceFactory

setup_logging()
logger = get_logger(__name__)


def resolve_jira_pat() -> str:
    pat = read_secret("JIRA_TICKET_RESPONDER_PAT")
    if pat:
        return pat
    raise ValueError("Missing Jira auth: set JIRA_TICKET_RESPONDER_PAT.")


def main() -> None:
    logger.info("Starting Jira ticket responder")

    factory = PostgresServiceFactory.from_env(
        password_override=read_secret("PG_PASSWORD")
    )
    PostgresServiceFactory.set_instance(factory)

    services_config = get_services_config()
    jira_config = responder_config.JiraServiceConfig.from_config(
        services_config.get("jira_ticket_responder", {})
    )
    pat = resolve_jira_pat()
    trigger_store = responder_store.JiraTriggerStore(factory)
    try:
        # This is a workaround to ensure existing deployments work.
        # TODO Should be removed once DB migrations will be in place.
        trigger_store.ensure_schema()
    except Exception:
        logger.error(
            "Failed to ensure Jira responder database schema; refusing to poll.",
            exc_info=True,
        )
        raise

    agent_config = responder_config.resolve_jira_agent_config(services_config)
    archi_instance = archi_factory.build_archi_for_jira(agent_config)

    issue_client = JiraIssueClient(jira_config.url, pat)
    service = responder_service.JiraTicketResponderService(
        config=jira_config,
        issue_client=issue_client,
        archi_instance=archi_instance,
        postgres_factory=factory,
        trigger_store=trigger_store,
        agent_config=agent_config,
    )
    logger.info(
        "Jira ticket responder lookback window: %s days", jira_config.lookback_days
    )

    while True:
        service.poll_once()
        time.sleep(jira_config.poll_interval_minutes * 60)


if __name__ == "__main__":
    main()
