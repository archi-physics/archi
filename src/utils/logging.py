import logging
import os


# ignore debug logs from these modules, too verbose :)
ignore_debug_modules = [
    "urllib3.connectionpool",
    "filelock",
    "httpcore",
    "openai._base_client",
    "unstructured.trace",
    "chardet.charsetprober"
]

logging_verboseLevel = [
    logging.CRITICAL,
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    logging.DEBUG,
]

def setup_logging():
    verbosity = int(os.getenv("VERBOSITY", 3))

    format_str = '(%(asctime)s) [%(name)s] %(levelname)s: %(message)s'

    level = logging_verboseLevel[max(0, min(4, verbosity))]
    logging.basicConfig(
        level=level,
        format=format_str,
        force=True
    )

    # need to override werkzeug which Flask uses
    logging.getLogger('werkzeug').setLevel(level)

    if verbosity == 4:
        for module in ignore_debug_modules:
            logging.getLogger(module).setLevel(logging_verboseLevel[3])


def setup_cli_logging(verbosity):
    
    if verbosity > 3: # high verbose mode
        format_str = '[%(name)s] %(levelname)s: %(message)s'
    else: # low verbose mode
        format_str = '[archi] %(message)s'
    level = logging_verboseLevel[max(0, min(4, verbosity))]
    logging.basicConfig(
        level=level,
        format=format_str,
        force=True
    )

def get_logger(name, verbosity=None):
    logger = logging.getLogger(name)
    if verbosity is not None:
        logger.setLevel(logging_verboseLevel[max(0, min(4, verbosity))])
    return logger
