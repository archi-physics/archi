import json
import os
import time
from threading import Thread

import requests
from flask import Flask

from src.a2rchi.a2rchi import A2rchi
from src.data_manager.data_manager import DataManager
from src.utils.runtime_config_loader import load_runtime_config
from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)

class MattermostAIWrapper:
    def __init__(self):
        # initialize and update vector store
        self.data_manager = DataManager(run_ingestion=False)

        # intialize chain
        self.a2rchi = A2rchi()

    def __call__(self, post):

        # form the formatted history using the post
        formatted_history = []

        post_str = post['message']
        formatted_history.append(("User", post_str)) 

        # call chain
        answer = self.a2rchi(formatted_history)["answer"]
        logger.debug('ANSWER = ',answer)

        return answer, post_str

class Mattermost:
    """
    Class to go through unresolved posts in Mattermost and suggest answers.
    Filter feed for new posts and propose answers.
    Also filter for new posts that have been resolved and add to vector store.
    For now, just iterate through all posts and send replies for unresolved.
    """
    def __init__(self):

        logger.info('Mattermost::INIT')

        self.mattermost_config = load_runtime_config()["utils"].get("mattermost", None)
        
        # mattermost webhook for reading questions/sending responses
        self.mattermost_url = 'https://mattermost.web.cern.ch/'
        self.mattermost_webhook = read_secret("MATTERMOST_WEBHOOK")
        self.mattermost_channel_id_read = read_secret("MATTERMOST_CHANNEL_ID_READ")
        self.mattermost_channel_id_write = read_secret("MATTERMOST_CHANNEL_ID_WRITE")
        self.PAK = read_secret("MATTERMOST_PAK")        
        self.mattermost_headers = {
            'Authorization': f'Bearer {self.PAK}',
            'Content-Type': 'application/json'
        }

        logger.debug('mattermost_webhook =', self.mattermost_webhook)
        logger.debug('mattermost_channel_id_read =', self.mattermost_channel_id_read)
        logger.debug('mattermost_channel_id_write =', self.mattermost_channel_id_write)
        logger.debug('PAK =', self.PAK)

        # initialize MattermostAIWrapper
        self.ai_wrapper = MattermostAIWrapper()

        #
        self.min_next_post_file = "/root/data/LPC2025/min_next_post.json"

    def post_response(self, response):

#        TODO: support writing in a dedicated mattermost_channel_id_write
#        url = f"{self.mattermost_url}/api/v4/posts"
#        print('GOING TO WRITE HERE: ',url)

#        payload = {
#            "channel_id": self.mattermost_channel_id_write,
#            "message": response
#        }
        # send response to MM
        #r = requests.post(url, data=json.dumps(payload), headers=self.mattermost_headers)
        r = requests.post(self.mattermost_webhook, data=json.dumps({"text": response,"channel" : "town-square"}), headers=self.mattermost_headers)

        return

    def write_min_next_post(self, answered_key):
        try:
            # create directory if it does not exist
            os.makedirs(os.path.dirname(self.min_next_post_file), exist_ok=True)
            with open(self.min_next_post_file, "w") as f:
                json.dump({"answered_id": answered_key}, f)
            logger.info(f"Updated answered_id {answered_key}")
        except Exception as e:
            logger.debug(f"ERROR - Failed to write answered_key to file: {e}")

    def get_active_posts(self):

        content = f"api/v4/channels/{self.mattermost_channel_id_read}/posts"
        r = requests.get(self.mattermost_url + content, headers=self.mattermost_headers)
        active_posts={}
        for id in r.json()["order"]:
            active_posts[id]=r.json()["posts"][id]["message"]
        return active_posts

    def filter_posts(self, posts, excluded_user_id):
        system_types = {
            "system_join_team",
            "system_join_channel",
            "system_add_to_channel",
            "system_leave_team",
            "system_leave_channel",
            "system_remove_from_channel",
        }

        filtered = []

        for post in posts.values():
            if post.get("user_id") == excluded_user_id:
                continue  # Skip this user

            if post.get("type") in system_types:
                continue  # Skip system messages

            filtered.append(post)

        return filtered

    def get_last_post(self):

        content = f"api/v4/channels/{self.mattermost_channel_id_read}/posts"

        r = requests.get(self.mattermost_url + content, headers=self.mattermost_headers)
        data = r.json()
        posts = data.get('posts', {})
        excluded_a2rchi_id = "ajb6wyizpinqir7m16owntod7o"

        filtered_posts = self.filter_posts(posts, excluded_user_id=excluded_a2rchi_id)
        sorted_posts = sorted(filtered_posts, key=lambda x: x['create_at'], reverse=True)

        if sorted_posts:
            latest = sorted_posts[0]
            logger.info(f"User ID: {latest['user_id']}")
            logger.info(f"Message: {latest['message']}")
        else:
            logger.debug("Mattermost: No messages found.")

        return sorted_posts[0]

    def checkAnswerExist(self, tmpID):

        # Check if file exists
        if not os.path.exists(self.min_next_post_file):
            logger.info("File does not exist, creating new one.")
            data = {"answered_id": []}  # Initialize with empty list
            return False
        else:
            # Load existing data
            with open(self.min_next_post_file, "r") as f:
                data = json.load(f)
                logger.info("Loaded data:", data)

        answered_ids = data.get("answered_id", [])

        # Ensure it's a list
        if not isinstance(answered_ids, list):
            answered_ids = [answered_ids]

        # Only append if not already in the list (optional)
        if tmpID in answered_ids:
            logger.info(f"{tmpID} already exists")
            return True
        else:
            answered_ids.append(tmpID)
            data["answered_id"] = answered_ids  # Overwrite with new ID
            with open(self.min_next_post_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Added {tmpID} for next iterations")
            return False

    # for now just processes "main" posts, i.e. not replies/follow-ups
    def process_posts(self):

        try:
            # get last post
            topic = self.get_last_post()
        except Exception as e:
            logger.error("ERROR - Failed to parse feed due to the following exception:")
            logger.error(str(e))
            return

        if self.checkAnswerExist(topic['id']) :
             # no need to answer someone already answered
             logger.info('no need to answer someone already answered')
        else:
            # otherwise, process it
            try:
                answer, post_str = self.ai_wrapper(topic)
                print('topic',topic,' \n ANSWER: ',answer)
                postedMM = self.post_response(answer)
                post_str = self.write_min_next_post(topic['id'])

            except Exception as e:
                logger.error(f"ERROR - Failed to process post {topic['id']} due to the following exception:")
                logger.error(str(e))
