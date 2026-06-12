import logging
from datetime import datetime
import os

log_path = os.path.join(os.getcwd() , 'logs')
os.makedirs(log_path, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter( '[%(asctime)s] [%(lineno)d] [%(name)s] [%(levelname)s] - %(message)s')
    ## File handler

    file_handler=logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(formatter)

    ##Console handler
    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

