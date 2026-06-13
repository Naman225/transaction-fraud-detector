import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import seaborn as sns  
import warnings
import logging
import json

from src.utils.save_object import save_object
from src.utils.logger import get_logger

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (classification_report, average_precision_score, 
                             precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score,
                             precision_recall_curve)
from sklearn.model_selection import RandomizedSearchCV
import shap
import mlflow
import mlflow.sklearn

# Suppress all python warnings and silence MLflow warnings
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, artifact_dir_1="artifacts/metrics", artifact_dir_2="artifacts/model"):
        self.artifact_dir_1 = artifact_dir_1
        self.artifact_dir_2 = artifact_dir_2
        os.makedirs(self.artifact_dir_1, exist_ok=True) 
        os.makedirs(self.artifact_dir_2, exist_ok=True) 
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            "Xgboost": XGBClassifier(tree_method='hist', random_state=42, n_jobs=-1),
            "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        }
        self.trained_models = {}
        self.model_scores = {}

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("CreditCard-Fraud-Detection")

    def _log_model_run(self, name, model, y_test, y_preds, y_proba):
        """Helper: log params + metrics + model to the current active MLflow run."""
        
        precision = precision_score(y_test, y_preds)
        recall = recall_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds)
        roc_auc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()

        # Parameters
        mlflow.log_param("model_name", name)
        mlflow.log_params(model.get_params())

        # Metrics
        mlflow.log_metrics({
            "precision": precision, "recall": recall,
            "f1_score": f1, "roc_auc": roc_auc, "auprc": auprc
        })

        # Model artifact
        mlflow.sklearn.log_model(model, name="model")

        return {"precision": precision, "recall": recall, "f1": f1,
                "roc_auc": roc_auc, "auprc": auprc,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp}

    def train_evaluate_without_SMOTE(self, X_train_scaled, X_test_scaled, y_train, y_test):
        logger.info("Starting model training on scaled baseline data (No SMOTE)... ")
        metrics_list = []

        # Start parent run if not already active
        if mlflow.active_run() is None:
            self.parent_run = mlflow.start_run(run_name="Model-Training-Pipeline")
            logger.info("Started parent MLflow run: Model-Training-Pipeline")

        with mlflow.start_run(run_name="Baseline-Experiment", nested=True):
            mlflow.set_tag("experiment_type", "Baseline")

            for name, model in self.models.items():
                with mlflow.start_run(run_name=name, nested=True):
                    logger.debug(f"Fitting model: {name} ")
                    model.fit(X_train_scaled, y_train)
                    self.trained_models[name] = model

                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    y_preds = model.predict(X_test_scaled)

                    m = self._log_model_run(name, model, y_test, y_preds, y_proba)

                    # --- LightGBM Baseline Diagnostic ---
                    if name == "LightGBM":
                        pred_counts = np.bincount(y_preds, minlength=2)
                        mean_fraud_prob = y_proba.mean()
                        majority_collapse = (pred_counts[1] == 0)
                        logger.info(
                            f"[LightGBM Diagnostic] Prediction distribution: "
                            f"Genuine={pred_counts[0]}, Fraud={pred_counts[1]} | "
                            f"Mean fraud probability={mean_fraud_prob:.6f} | "
                            f"Majority class collapse={majority_collapse}"
                        )
                        mlflow.set_tag("majority_class_collapse", str(majority_collapse))
                        mlflow.set_tag("diagnostic_note",
                            "Extreme class imbalance without resampling causes "
                            "LightGBM leaf-wise growth to ignore minority class signal")

                    logger.debug(f"[{name}] Classification Report:\n{classification_report(y_test, y_preds)}")
                    logger.info(f"[{name}] Baseline Evaluation Complete. AUPRC: {m['auprc']:.4f}")

                    metrics_list.append({
                        "Model": name, "Experiment_Type": "Baseline",
                        "AUPRC": m["auprc"], "Precision": m["precision"],
                        "Recall": m["recall"], "F1_Score": m["f1"],
                        "ROC_AUC": m["roc_auc"],
                        "True_Negatives": m["tn"], "False_Positives": m["fp"],
                        "False_Negatives": m["fn"], "True_Positives": m["tp"]
                    })

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(os.path.join(self.artifact_dir_1, "model_metrics_baseline.csv"), index=False)
        logger.info(f"Saved baseline metrics to {self.artifact_dir_1}/model_metrics_baseline.csv")
        return metrics_df


    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        logger.info("Training on SMOTE balanced Data ... ")
        metrics_list = []

        # Start parent run if not already active
        if mlflow.active_run() is None:
            self.parent_run = mlflow.start_run(run_name="Model-Training-Pipeline")
            logger.info("Started parent MLflow run: Model-Training-Pipeline")

        with mlflow.start_run(run_name="SMOTE-Experiment", nested=True):
            mlflow.set_tag("experiment_type", "SMOTE")

            for name, model in self.models.items():
                with mlflow.start_run(run_name=name, nested=True):
                    logger.debug(f"Fitting model: {name} ")
                    model.fit(X_train, y_train)
                    self.trained_models[name] = model

                    y_proba = model.predict_proba(X_test)[:, 1]
                    y_preds = model.predict(X_test)

                    m = self._log_model_run(name, model, y_test, y_preds, y_proba)
                    self.model_scores[name] = m["auprc"]

                    logger.debug(f"[{name}] Classification Report:\n{classification_report(y_test, y_preds)}")
                    logger.info(f"[{name}] SMOTE Evaluation Complete. AUPRC: {m['auprc']:.4f}")

                    metrics_list.append({
                        "Model": name, "Experiment_Type": "SMOTE",
                        "AUPRC": m["auprc"], "Precision": m["precision"],
                        "Recall": m["recall"], "F1_Score": m["f1"],
                        "ROC_AUC": m["roc_auc"],
                        "True_Negatives": m["tn"], "False_Positives": m["fp"],
                        "False_Negatives": m["fn"], "True_Positives": m["tp"]
                    })

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(os.path.join(self.artifact_dir_1, "model_metrics_smote.csv"), index=False)
        logger.info(f"Saved SMOTE metrics to {self.artifact_dir_1}/model_metrics_smote.csv")
        return metrics_df


    def save_best_model(self, optimal_threshold=None):
        """
        Select best model based on SMOTE AUPRC
        and save locally.
        """

        if not self.trained_models:
            logger.error("No trained models found.")
            return

        if not self.model_scores:
            logger.error("No model scores found.")
            return

        best_name = max(
            self.model_scores,
            key=self.model_scores.get
        )

        best_model = self.trained_models[best_name]

        best_auprc = self.model_scores[best_name]

        model_path = os.path.join(
            self.artifact_dir_2,
            "model.pkl"
        )

        save_object(model_path, best_model)

        threshold = float(optimal_threshold) if optimal_threshold is not None else 0.5

        metadata = {
            "best_model": best_name,
            "best_auprc": float(best_auprc),
            "threshold": threshold
        }

        metadata_path = os.path.join(
            self.artifact_dir_2,
            "metadata.json"
        )

        with open(metadata_path, "w") as f:
            json.dump(
                metadata,
                f,
                indent=4
            )

        logger.info(
            f"Best model '{best_name}' "
            f"(AUPRC={best_auprc:.4f}, threshold={threshold:.4f}) "
            f"saved to {model_path}"
        )

        return best_name, best_model


    def tune_best_model(self, X_train, X_test, y_train, y_test):
        """
        Runs RandomizedSearchCV on Random Forest to find optimal hyperparameters.
        Logs the tuning results to MLflow and updates the trained model if improved.
        """
        logger.info("Starting hyperparameter tuning for Random Forest...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        search = RandomizedSearchCV(
            rf, param_grid,
            scoring='average_precision',
            cv=3, n_iter=10,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        best_rf = search.best_estimator_
        cv_auprc = search.best_score_
        best_params = search.best_params_

        logger.info(f"Tuning complete — CV AUPRC: {cv_auprc:.4f}")
        logger.info(f"Best params: {best_params}")

        # Evaluate on test set
        y_proba = best_rf.predict_proba(X_test)[:, 1]
        y_preds = best_rf.predict(X_test)

        test_precision = precision_score(y_test, y_preds)
        test_recall = recall_score(y_test, y_preds)
        test_f1 = f1_score(y_test, y_preds)
        test_roc_auc = roc_auc_score(y_test, y_proba)
        test_auprc = average_precision_score(y_test, y_proba)

        logger.info(f"Tuning complete — CV AUPRC: {cv_auprc:.4f}, Test AUPRC: {test_auprc:.4f}")

        # Log to MLflow
        with mlflow.start_run(run_name="Hyperparameter-Tuning", nested=True):
            mlflow.set_tag("tuning_method", "RandomizedSearchCV")
            mlflow.set_tag("tuned_model", "Random Forest")
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                "cv_auprc": cv_auprc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_roc_auc": test_roc_auc,
                "test_auprc": test_auprc
            })
            mlflow.sklearn.log_model(best_rf, name="tuned_model")

        # Save tuning results
        tuning_results = {
            "best_params": best_params,
            "cv_auprc": float(cv_auprc),
            "test_metrics": {
                "precision": float(test_precision),
                "recall": float(test_recall),
                "f1": float(test_f1),
                "roc_auc": float(test_roc_auc),
                "auprc": float(test_auprc)
            },
            "previous_auprc": float(self.model_scores.get("Random Forest", 0)),
            "improved": test_auprc > self.model_scores.get("Random Forest", 0)
        }

        tuning_path = os.path.join(self.artifact_dir_1, "tuning_results.json")
        with open(tuning_path, "w") as f:
            json.dump(tuning_results, f, indent=4)
        logger.info(f"Saved tuning results to {tuning_path}")

        # Update model if improved
        if test_auprc > self.model_scores.get("Random Forest", 0):
            self.trained_models["Random Forest"] = best_rf
            self.model_scores["Random Forest"] = test_auprc
            logger.info(
                f"Tuned Random Forest IMPROVED AUPRC: "
                f"{tuning_results['previous_auprc']:.4f} → {test_auprc:.4f}"
            )
        else:
            logger.info(
                f"Tuned Random Forest did NOT improve AUPRC: "
                f"{tuning_results['previous_auprc']:.4f} vs {test_auprc:.4f}. Keeping original."
            )

        return tuning_results

    def analyze_optimal_threshold(self, X_test, y_test):
        """
        Finds the threshold that maximizes F1 score using the precision-recall curve
        of the best model. Generates a threshold analysis visualization.
        """
        if not self.model_scores:
            logger.error("No model scores available for threshold analysis.")
            return 0.5

        best_name = max(self.model_scores, key=self.model_scores.get)
        best_model = self.trained_models[best_name]

        logger.info(f"Analyzing optimal threshold for {best_name}...")

        y_proba = best_model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        # Compute F1 at each threshold (precision and recall arrays are 1 longer than thresholds)
        precision_t = precision[:-1]
        recall_t = recall[:-1]
        f1_scores = np.where(
            (precision_t + recall_t) > 0,
            2 * (precision_t * recall_t) / (precision_t + recall_t),
            0
        )

        optimal_idx = f1_scores.argmax()
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_f1 = float(f1_scores[optimal_idx])
        optimal_precision = float(precision_t[optimal_idx])
        optimal_recall = float(recall_t[optimal_idx])

        logger.info(
            f"Optimal threshold: {optimal_threshold:.4f} "
            f"(F1={optimal_f1:.4f}, Precision={optimal_precision:.4f}, Recall={optimal_recall:.4f})"
        )

        # --- Generate Threshold Analysis Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Precision, Recall, F1 vs Threshold
        axes[0].plot(thresholds, precision_t, label='Precision', color='#2196F3', linewidth=2)
        axes[0].plot(thresholds, recall_t, label='Recall', color='#FF9800', linewidth=2)
        axes[0].plot(thresholds, f1_scores, label='F1 Score', color='#4CAF50', linewidth=2)
        axes[0].axvline(x=optimal_threshold, color='#F44336', linestyle='--', linewidth=1.5,
                        label=f'Optimal Threshold = {optimal_threshold:.3f}')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title(f'Threshold Analysis — {best_name}')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Right: Precision-Recall Curve
        axes[1].plot(recall, precision, color='#9C27B0', linewidth=2)
        axes[1].axhline(y=optimal_precision, color='#F44336', linestyle=':', alpha=0.5)
        axes[1].axvline(x=optimal_recall, color='#F44336', linestyle=':', alpha=0.5)
        axes[1].plot(optimal_recall, optimal_precision, 'r*', markersize=15,
                     label=f'Optimal (P={optimal_precision:.3f}, R={optimal_recall:.3f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Optimal Threshold Analysis — F1 Maximized at {optimal_threshold:.3f}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(self.artifact_dir_1, "threshold_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved threshold analysis plot to: {plot_path}")
        plt.close(fig)

        # Log to MLflow
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_metrics({
                "optimal_threshold": optimal_threshold,
                "optimal_f1": optimal_f1,
                "optimal_precision": optimal_precision,
                "optimal_recall": optimal_recall
            })
            mlflow.log_artifact(plot_path, artifact_path="evaluation")

        return optimal_threshold

    def global_feature_importance(self, X_train):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        tree_models = ["Random Forest", "Xgboost", "LightGBM"]
        has_plots = False
        for i, name in enumerate(tree_models):
            if name not in self.trained_models:
                logger.warning(f"Model '{name}' missing from trained history. Skipping plot.")
                continue
            has_plots = True
            model = self.trained_models[name]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]      
            feature_names = [X_train.columns[idx] for idx in indices]
            sns.barplot(
                    x=importances[indices], 
                    y=feature_names, 
                    ax=axes[i], 
                    palette="copper",
                    hue=feature_names,
                    legend=False
                )
            axes[i].set_title(f"Top 10 Features: {name}")
            axes[i].set_xlabel("Relative Importance Score")
        if has_plots:
            plt.suptitle("Global Feature Importance Comparison Matrix")
            plt.tight_layout()
            plot_path = os.path.join(self.artifact_dir_1, "global_feature_importance.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance matrix visualization to: {plot_path}")
            plt.show()
        else:
            plt.close(fig)


    def plot_confusion_matrices(self, X_test, y_test):
        """Generates and saves confusion matrix heatmaps for all trained models."""
        model_names = list(self.trained_models.keys())
        n_models = len(model_names)
        if n_models == 0:
            logger.warning("No trained models available for confusion matrix plots.")
            return

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for i, name in enumerate(model_names):
            model = self.trained_models[name]
            y_preds = model.predict(X_test)
            cm = confusion_matrix(y_test, y_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'])
            axes[i].set_title(f"{name}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

        plt.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(self.artifact_dir_1, "confusion_matrices.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix heatmaps to: {plot_path}")
        plt.show()


    def generating_local_shap(self, X_test_scaled):

        xgb_model = self.trained_models.get("Xgboost")
        if xgb_model is not None:
            logger.info("Calculating SHAP values for XGBoost (Sampling up to 500 records)...")
            X_sample = X_test_scaled.sample(min(500, len(X_test_scaled)), random_state=42)
            
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(X_sample)
            
            plt.figure(figsize=(11, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title("SHAP Feature Impact Breakdown on Fraud Predictions (XGBoost)")
            plt.tight_layout()
            shap_path = os.path.join(self.artifact_dir_1, "shap_summary_xgboost.png")
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to: {shap_path}")
            plt.show()
        else:
            logger.error("XGBoost model was not found in trained records. Skipping SHAP extraction.")


    def log_experiment_artifacts(self):
        """Logs all generated artifacts (plots, CSVs, best model) to the active MLflow run or a new one, and ends the parent run."""
        if not self.model_scores:
            logger.warning("No model scores available. Skipping MLflow artifact logging.")
            return

        best_name = max(self.model_scores, key=self.model_scores.get)
        best_auprc = self.model_scores[best_name]

        # Use the active parent run if available
        active_run = mlflow.active_run()
        
    
        if active_run is not None:
            mlflow.set_tag("best_model", best_name)
            mlflow.log_metric("best_auprc", best_auprc)

            # Log all evaluation artifacts (PNGs + CSVs)
            for fname in os.listdir(self.artifact_dir_1):
                fpath = os.path.join(self.artifact_dir_1, fname)
                if os.path.isfile(fpath):
                    mlflow.log_artifact(fpath, artifact_path="evaluation")

            # Log best model pkl
            model_path = os.path.join(self.artifact_dir_2, "model.pkl")
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")
            logger.info(
            f"Logged all experiment artifacts to active parent MLflow run. "
            f"Best model: {best_name}")

            # Log scaler pkl
            # Log scaler pkl
            scaler_path = os.path.join(
                self.artifact_dir_2,
                "scaler.pkl"
            )

            if os.path.exists(scaler_path):
                mlflow.log_artifact(
                    scaler_path,
                    artifact_path="model"
                )

            # Log metadata.json
            metadata_path = os.path.join(
                self.artifact_dir_2,
                "metadata.json"
            )

            if os.path.exists(metadata_path):
                mlflow.log_artifact(
                    metadata_path,
                    artifact_path="model"
                )
        else:
            # Fallback if no active run is present
            with mlflow.start_run(run_name=f"Best-Model-{best_name}"):
                mlflow.set_tag("best_model", best_name)
                mlflow.log_metric("best_auprc", best_auprc)

                for fname in os.listdir(self.artifact_dir_1):
                    fpath = os.path.join(self.artifact_dir_1, fname)
                    if os.path.isfile(fpath):
                        mlflow.log_artifact(fpath, artifact_path="evaluation")

                model_path = os.path.join(self.artifact_dir_2, "model.pkl")
                if os.path.exists(model_path):
                    mlflow.log_artifact(model_path, artifact_path="model")

                scaler_path = os.path.join(
                    self.artifact_dir_2,
                    "scaler.pkl"
                )

                if os.path.exists(scaler_path):
                    mlflow.log_artifact(
                        scaler_path,
                        artifact_path="model"
                    )

                metadata_path = os.path.join(
                    self.artifact_dir_2,
                    "metadata.json"
                )

                if os.path.exists(metadata_path):
                    mlflow.log_artifact(
                        metadata_path,
                        artifact_path="model"
                    )

                logger.info(f"Logged all experiment artifacts to new MLflow run. Best model: {best_name}")

    def close_mlflow_run(self):

        if mlflow.active_run():

            logger.info(
                "Closing MLflow run..."
            )

            mlflow.end_run()
