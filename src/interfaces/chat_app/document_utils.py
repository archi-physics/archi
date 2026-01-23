import hashlib
import os
import yaml
from pathlib import Path

from src.utils.runtime_config_loader import load_runtime_config
from src.utils.env import read_secret
from src.utils.logging import get_logger
from src.data_manager.collectors.persistence import PersistenceService
from src.data_manager.collectors.scrapers.scraped_resource import \
    ScrapedResource

logger = get_logger(__name__)

def simple_hash(input_string):
    """
    Takes an input string and outputs a hash
    """

    # perform the hash operation using hashlib
    identifier = hashlib.md5()
    identifier.update(input_string.encode('utf-8'))
    hash_value = str(int(identifier.hexdigest(), 16))

    return hash_value


def file_hash(filename):
    """
    Takes an input filename and converts it to a hash.
    
    However, unlike `simple_hash` this method keeps the file extension
    at the end of the name
    """

    return simple_hash(filename)[0:12] + "." + filename.split(".")[-1]

def add_uploaded_file(target_dir, file, file_extension) -> ScrapedResource:

    resource = ScrapedResource(
                url=file.filename,
                content=file.read(),
                suffix=file_extension,
                source_type="files",
                metadata={
                    "content_type": file.content_type,
                    "title": file.filename
                },
            )
    files_hash = file_hash(file.filename)
    file.save(os.path.join(target_dir, files_hash))
    return resource

def get_filename_from_hash(hash_string, data_path, filehashes_yaml_file="manual_file_hashes.yaml"):
    """
    Given a file hash, returns the original file name from the map chat_app
    """
    try:
        # load existing accounts or initialize as empty dictionary
        with open(os.path.join(data_path, filehashes_yaml_file), 'r') as file:
            filenames_dict = yaml.safe_load(file) or {}
    except FileNotFoundError:
        filenames_dict = {}

    return filenames_dict[hash_string] if hash_string in filenames_dict else None


def remove_url_from_sources(url: str, sources_path: str):
    config = load_runtime_config()
    pg_config = {
        "password": read_secret("PG_PASSWORD"),
        **config["services"]["postgres"],
    }
    persistence = PersistenceService(sources_path, pg_config=pg_config)
    persistence.delete_by_metadata_filter("url", url)


def add_username_password(username, password, salt, accounts_path, file_name='accounts.yaml'):
    """
    Given username and password in string format, this method hashes the password
    concatenated with a salt (stored as an environment variable) and saves it to 
    the accounts yaml file.

    Keys in the yaml file are usernames and passwords are the hashes of the password
    plus the salt.

    Adding a username and password for an existing username will overwrite the password
    """
    hash = simple_hash(password + salt)
    try:
        # load existing accounts or initialize as empty dictionary
        with open(os.path.join(accounts_path, file_name), 'r') as file:
            accounts = yaml.safe_load(file) or {}
    except FileNotFoundError:
        accounts = {}

    # check if the username already exists
    if username in accounts:
        logger.info(f"Username '{username}' already exists.")
        return

    # add the new username and hashed password to the accounts dictionary
    accounts[username] = hash

    # write the updated dictionary back to the YAML file
    with open(os.path.join(accounts_path, file_name), 'w') as file:
        yaml.dump(accounts, file)


def check_credentials(username, password, salt, accounts_path, file_name='accounts.yaml'):
    """
    Given username and password in string format, this methods returns true if the 
    username and password match the stored username and password. This is done by 
    hashing the password and checking that it is the same as the hashed password
    which is stored under the specific username.

    Keys in the yaml file are usernames and passwords are the hashes of the password
    plus the salt.
    """

    # create the hash
    hash = simple_hash(password + salt)

    # open the accounts dictionary (if file exists, account dictionary is null)
    try:
        # load existing accounts or initialize as empty dictionary
        with open(os.path.join(accounts_path, file_name), 'r') as file:
            accounts = yaml.safe_load(file) or {}
            logger.info(f"Accounts are: {accounts}")

    except FileNotFoundError:
        accounts = {}

    # check if hash matches the hash in the accounts dictionary
    return (username in accounts) and (accounts[username] == hash)
