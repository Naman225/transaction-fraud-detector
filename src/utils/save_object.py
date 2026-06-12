import os
import pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_object(file_path: str, obj):
    """
    Saves an object to a specified file path using pickle.
    Creates parent directories if they don't exist.
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Successfully saved object to {file_path}")
    except Exception as e:
        logger.error(f"Error while saving object to {file_path}: {e}")
        raise e

def load_object(file_path: str):
    """
    Loads an object from a specified file path using pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Successfully loaded object from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error while loading object from {file_path}: {e}")
        raise e
