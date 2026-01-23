#!/bin/python
import multiprocessing as mp
import os

from flask import Flask

from src.interfaces.chat_app.app import FlaskAppWrapper
from src.utils.runtime_config_loader import load_runtime_config
from src.utils.env import read_secret
from src.utils.logging import setup_logging


def main():
    
    setup_logging()

    # load secrets
    os.environ['ANTHROPIC_API_KEY'] = read_secret("ANTHROPIC_API_KEY")
    os.environ['OPENAI_API_KEY'] = read_secret("OPENAI_API_KEY")
    os.environ['HUGGING_FACE_HUB_TOKEN'] = read_secret("HUGGING_FACE_HUB_TOKEN")
    
    runtime_config = load_runtime_config()
    chat_config = runtime_config["services"]["chat_app"]
    a2rchi_config = runtime_config["a2rchi"]
    print(f"Starting Chat Service with (host, port): ({chat_config['host']}, {chat_config['port']})")
    print(f"Accessible externally at (host, port): ({chat_config['hostname']}, {chat_config['external_port']})")

    generate_script(chat_config, a2rchi_config, runtime_config["services"])
    app = FlaskAppWrapper(Flask(
        __name__,
        template_folder=chat_config["template_folder"],
        static_folder=chat_config["static_folder"],
    ))
    app.run(debug=True, use_reloader=False, port=chat_config["port"], host=chat_config["host"])


def _resolve_agent_description(a2rchi_config, services_config):
    pipeline_name = services_config.get("chat_app", {}).get("pipeline")
    if not pipeline_name:
        pipeline_name = (a2rchi_config.get("pipelines") or [None])[0]
    pipeline_cfg = a2rchi_config.get("pipeline_map", {}).get(pipeline_name, {}) if pipeline_name else {}
    return pipeline_cfg.get("agent_description", "No description provided")


def generate_script(chat_config, a2rchi_config, services_config):
    """
    This is not elegant but it creates the javascript file from the template using the config.yaml parameters
    """
    script_template = os.path.join(chat_config["static_folder"], "script.js-template")
    with open(script_template, "r") as f:
        template = f.read()

    filled_template = template.replace('XX-NUM-RESPONSES-XX', str(chat_config["num_responses_until_feedback"]))
    agent_description = _resolve_agent_description(a2rchi_config, services_config)
    filled_template = filled_template.replace('XX-TRAINED_ON-XX', str(agent_description))

    script_file = os.path.join(chat_config["static_folder"], "script.js")
    with open(script_file, "w") as f:
        f.write(filled_template)

    return

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
