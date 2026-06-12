import pandas as pd
import os 
import kagglehub
from src.utils.logger import get_logger

logger = get_logger(__name__)
class DataIngestion:
    def __init__(self):
        pass
    def load_data(self):
        logger.info("Data Ingestion started ... ")
        path = "./data/creditcard.csv"
        (f"Checking directory: {path}")
        
        os.makedirs("./data", exist_ok=True)
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:        
            dir_path  = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            file_path = os.path.join(dir_path, "creditcard.csv")
            df = pd.read_csv(file_path)
            df.to_csv(path,index=False)
        logger.info("Data Ingestion Successfully Completed")
        return df
