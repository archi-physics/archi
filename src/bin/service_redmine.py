#!/bin/python
import os
import time

from src.interfaces.redmine_mailer_integration import redmine
from src.utils.runtime_config_loader import load_runtime_config
from src.utils.env import read_secret
from src.utils.logging import setup_logging

# set basicConfig for logging
setup_logging()

# set openai
os.environ['OPENAI_API_KEY'] = read_secret("OPENAI_API_KEY")
os.environ['HUGGING_FACE_HUB_TOKEN'] = read_secret("HUGGING_FACE_HUB_TOKEN")

# temporary hack to prevent redmine, mailbox, and chat services from all
# starting DataManager at the same time; eventually replace this with
# more robust solution
time.sleep(30)

print("Starting Redmine Service")
runtime_config = load_runtime_config()
redmine_config = runtime_config["services"]["redmine_mailbox"]
redmine = redmine.Redmine('Redmine_Helpdesk')

while True:
    redmine.load()
    redmine.process_new_issues()
    redmine.process_resolved_issues()
    time.sleep(int(redmine_config["redmine_update_time"]))
