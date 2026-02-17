#!/bin/python
import multiprocessing as mp
import os

from flask import Flask

from src.interfaces.chat_app.app import FlaskAppWrapper
from src.utils.env import read_secret
from src.utils.logging import setup_logging
from src.utils.postgres_service_factory import PostgresServiceFactory
from src.utils.config_access import get_full_config


def main():
    
    setup_logging()

    # load secrets
    os.environ['ANTHROPIC_API_KEY'] = read_secret("ANTHROPIC_API_KEY")
    os.environ['OPENAI_API_KEY'] = read_secret("OPENAI_API_KEY")
    os.environ['HUGGING_FACE_HUB_TOKEN'] = read_secret("HUGGING_FACE_HUB_TOKEN")

    # Set up shared Postgres services (expects config already in DB)
    factory = PostgresServiceFactory.from_env(password_override=read_secret("PG_PASSWORD"))
    PostgresServiceFactory.set_instance(factory)

    # Reload config from Postgres (runtime source of truth)
    config = get_full_config()
    chat_config = config["services"]["chat_app"]
    print(f"Starting Chat Service with (host, port): ({chat_config['host']}, {chat_config['port']})")
    print(f"Accessible externally at (host, port): ({chat_config['hostname']}, {chat_config['external_port']})")

    generate_script(chat_config)
    app = FlaskAppWrapper(Flask(
        __name__,
        template_folder=chat_config["template_folder"],
        static_folder=chat_config["static_folder"],
    ))
    app.run(debug=True, use_reloader=False, port=chat_config["port"], host=chat_config["host"])


def generate_script(chat_config):
    """
    This is not elegant but it creates the javascript file from the template using the config.yaml parameters
    """
    script_template = os.path.join(chat_config["static_folder"], "script.js-template")
    with open(script_template, "r") as f:
        template = f.read()

    filled_template = template.replace('XX-NUM-RESPONSES-XX', str(chat_config["num_responses_until_feedback"]))
    filled_template = filled_template.replace('XX-TRAINED_ON-XX', str(chat_config.get("trained_on", "")))

    script_file = os.path.join(chat_config["static_folder"], "script.js")
    with open(script_file, "w") as f:
        f.write(filled_template)

    return

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
