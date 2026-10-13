#!/bin/python
import multiprocessing as mp
import os

from flask import Flask

from src.interfaces.chat_app.api import register_api
from src.interfaces.chat_app.app import FlaskAppWrapper
from src.utils.env import read_secret
from src.utils.logging import get_logger, setup_logging
from src.utils.postgres_service_factory import PostgresServiceFactory
from src.utils.config_access import get_full_config
from src.utils.conversation_schema import ensure_conversation_schema

logger = get_logger(__name__)


def main():
    
    setup_logging()

    # load secrets
    os.environ['ANTHROPIC_API_KEY'] = read_secret("ANTHROPIC_API_KEY")
    os.environ['OPENAI_API_KEY'] = read_secret("OPENAI_API_KEY")
    os.environ['HUGGING_FACE_HUB_TOKEN'] = read_secret("HUGGING_FACE_HUB_TOKEN")

    # Set up shared Postgres services (expects config already in DB)
    factory = PostgresServiceFactory.from_env(password_override=read_secret("PG_PASSWORD"))
    PostgresServiceFactory.set_instance(factory)

    # Playbook schema (idempotent migration for pre-existing volumes) runs once at
    # service startup, not inside the Flask wrapper. A failure must not block
    # startup: the chat flow tolerates a missing side table (playbook tracking
    # degrades; chat keeps working).
    try:
        factory.playbook_service.ensure_schema()
    except Exception as exc:
        logger.error(
            "Could not ensure playbook schema — playbook features will degrade "
            "(no turn tracking, chip-less conversation loads) until it succeeds: %s",
            exc,
        )

    # Conversation pin/archive columns (idempotent migration for pre-existing volumes).
    pg_cfg = {"password": read_secret("PG_PASSWORD"), **get_full_config()["services"]["postgres"]}
    try:
        ensure_conversation_schema(pg_cfg)
    except Exception as exc:
        logger.error(
            "Could not ensure conversation pin/archive schema — "
            "pin and archive features will not work until it succeeds: %s",
            exc,
        )

    # Reload config from Postgres (runtime source of truth)
    config = get_full_config()
    chat_config = config["services"]["chat_app"]

    # Deployment-time fields may not be in Postgres; use sensible defaults
    host = chat_config.get("host", "0.0.0.0")
    port = chat_config.get("port", 7681)
    hostname = chat_config.get("hostname", host)
    external_port = chat_config.get("external_port", port)

    # Resolve template/static folders from installed package location
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _chat_app_dir = os.path.join(os.path.dirname(_pkg_dir), "interfaces", "chat_app")
    template_folder = chat_config.get("template_folder", os.path.join(_chat_app_dir, "templates"))
    static_folder = chat_config.get("static_folder", os.path.join(_chat_app_dir, "static"))

    print(f"Starting Chat Service with (host, port): ({host}, {port})")
    print(f"Accessible externally at (host, port): ({hostname}, {external_port})")

    generate_script(chat_config, static_folder)
    flask_app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder,
    )
    register_api(flask_app)
    app = FlaskAppWrapper(flask_app)
    app.run(debug=True, use_reloader=False, port=port, host=host)


def generate_script(chat_config, static_folder):
    """
    This is not elegant but it creates the javascript file from the template using the config.yaml parameters
    """
    script_template = os.path.join(static_folder, "script.js-template")
    with open(script_template, "r") as f:
        template = f.read()

    filled_template = template.replace('XX-NUM-RESPONSES-XX', str(chat_config.get("num_responses_until_feedback", 3)))
    filled_template = filled_template.replace('XX-TRAINED_ON-XX', str(chat_config.get("trained_on", "")))

    script_file = os.path.join(static_folder, "script.js")
    with open(script_file, "w") as f:
        f.write(filled_template)

    return

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
