#!/bin/python
import os

from flask import Flask

from src.interfaces.grader_app.app import FlaskAppWrapper
from src.utils.runtime_config_loader import load_runtime_config
from src.utils.env import read_secret
from src.utils.logging import setup_logging

# set basicConfig for logging and get debug value for flask app
setup_logging()

os.environ['ANTHROPIC_API_KEY'] = read_secret("ANTHROPIC_API_KEY")
os.environ['OPENAI_API_KEY'] = read_secret("OPENAI_API_KEY")
os.environ['HUGGING_FACE_HUB_TOKEN'] = read_secret("HUGGING_FACE_HUB_TOKEN")
runtime_config = load_runtime_config()
grader_config = runtime_config["services"]["grader_app"]

app = FlaskAppWrapper(Flask(
    __name__,
    template_folder=grader_config["template_folder"],
))

app.run(debug=grader_config["flask_debug_mode"], use_reloader=False, port=grader_config["port"], host=grader_config["host"])
