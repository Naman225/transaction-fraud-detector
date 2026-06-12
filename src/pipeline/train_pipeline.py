from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TrainingPipeline():
    def __init__(self):
        self.ingestion = DataIngestion()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()

    def run(self):
        ##Ingest Data
        df = self.ingestion.load_data()

        ## Transform Data
        X_train_resampled,X_train_scaled,X_test_scaled,y_train,y_train_resampled,y_test = self.transformation.load_data(df)

        ##Train && Evaluate
        baseline_metrics = self.trainer.train_evaluate_without_SMOTE(X_train_scaled,X_test_scaled,y_train,y_test)     
        smote_metrics = self.trainer.train_and_evaluate(X_train_resampled,X_test_scaled,y_train_resampled,y_test)     

        ## Select and save best model
        best_name, best_model = self.trainer.save_best_model()
        
        ## Generate artifacts
        self.trainer.global_feature_importance(X_train_resampled)
        self.trainer.plot_confusion_matrices(X_test_scaled, y_test)
        self.trainer.generating_local_shap(X_test_scaled)

        ## Log all artifacts to MLflow
        self.trainer.log_experiment_artifacts()

        return best_name

if __name__ == "__main__":
    training = TrainingPipeline()
    training.run()