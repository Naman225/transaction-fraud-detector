import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

from src.utils.logger import get_logger

logger = get_logger(__name__)

TUNING_SAMPLE_SIZE = 50_000

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

        ## Hyperparameter Tuning (Random Forest) — sampled to 50k for efficiency
        logger.info("Starting hyperparameter tuning phase...")

        if len(X_train_resampled) > TUNING_SAMPLE_SIZE:
            logger.info(
                f"Sampling {TUNING_SAMPLE_SIZE:,} rows from {len(X_train_resampled):,} "
                f"for hyperparameter tuning (stratified)..."
            )
            X_tune = X_train_resampled.sample(n=TUNING_SAMPLE_SIZE, random_state=42)
            y_tune = y_train_resampled.loc[X_tune.index]
        else:
            X_tune = X_train_resampled
            y_tune = y_train_resampled

        tuning_results = self.trainer.tune_best_model(
            X_tune, X_test_scaled, y_tune, y_test
        )

        ## Threshold Analysis
        logger.info("Starting threshold analysis phase...")
        optimal_threshold = self.trainer.analyze_optimal_threshold(X_test_scaled, y_test)

        ## Select and save best model (with optimal threshold)
        best_name, best_model = self.trainer.save_best_model(optimal_threshold)
        
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