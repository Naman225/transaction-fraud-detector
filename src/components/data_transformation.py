from src.components.data_ingestion import DataIngestion
from sklearn.preprocessing import RobustScaler 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from src.utils.save_object import save_object
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self):
        pass
    def scaling_data(self,X_train,X_test):
        scaler = RobustScaler()
        feature_scale = ['Time' ,'Amount']

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[feature_scale]=scaler.fit_transform(X_train[feature_scale])
        X_test_scaled[feature_scale]=scaler.transform(X_test[feature_scale])
        save_object("artifacts/model/scaler.pkl", scaler)
        return X_train_scaled ,X_test_scaled
    
    def visualize_scaling_data(self, X_train ,X_train_scaled):
        fig, axes = plt.subplots(1,2,figsize=(12,5))

        sns.boxplot(data=X_train[['Amount' , 'Time']] ,ax=axes[0])
        axes[0].set_title('Features Before Scaling')
        axes[0].set_ylabel('Values')
        sns.boxplot(data=X_train_scaled[['Amount' , 'Time']], ax=axes[1])
        axes[1].set_title('Features After Scaling')
        axes[1].set_ylabel('Scaled Values')
        plt.tight_layout()
        plt.show()

    def load_data(self,df):
        logger.info("Data Transformation Started ...")

        X = df.drop(columns='Class')
        y=df['Class']
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

        X_train_scaled , X_test_scaled = self.scaling_data(X_train , X_test)
        self.visualize_scaling_data(X_train , X_train_scaled)
        
        ##Applying Smote

        logger.info(f"Before SMOTE - Fraud Count : {sum(y_train == 1)} , Genuine Count : {sum(y_train==0)}")

        smote = SMOTE(random_state=42)

        X_train_resampled , y_train_resampled = smote.fit_resample(X_train_scaled,y_train)
        logger.info(f"After SMOTE  - Fraud Count: {sum(y_train_resampled == 1)}, Genuine Count: {sum(y_train_resampled == 0)}")
        
        logger.info("Data Transformation Successfully Completed")
        return X_train_resampled ,X_train_scaled , X_test_scaled , y_train ,y_train_resampled , y_test




