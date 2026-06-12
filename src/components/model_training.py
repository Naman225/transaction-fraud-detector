import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import seaborn as sns  

from src.utils.save_object import save_object
from src.utils.logger import get_logger

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (classification_report, average_precision_score, 
                             precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score)
import shap
import mlflow
import mlflow.sklearn

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

        with mlflow.start_run(run_name="Baseline-Experiment"):
            mlflow.set_tag("experiment_type", "Baseline")

            for name, model in self.models.items():
                with mlflow.start_run(run_name=name, nested=True):
                    logger.debug(f"Fitting model: {name} ")
                    model.fit(X_train_scaled, y_train)
                    self.trained_models[name] = model

                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    y_preds = model.predict(X_test_scaled)

                    m = self._log_model_run(name, model, y_test, y_preds, y_proba)

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

        with mlflow.start_run(run_name="SMOTE-Experiment"):
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


    def save_best_model(self):
        """Selects the best model by AUPRC (from SMOTE experiment) and saves it to artifacts/model/model.pkl"""
        if not self.trained_models:
            logger.error("No trained models found. Cannot save.")
            return
        if not self.model_scores:
            logger.error("No model scores recorded. Run train_and_evaluate() first.")
            return

        best_name = max(self.model_scores, key=self.model_scores.get)
        best_model = self.trained_models[best_name]
        best_auprc = self.model_scores[best_name]

        model_path = os.path.join(self.artifact_dir_2, "model.pkl")
        save_object(model_path, best_model)
        logger.info(f"Best model: '{best_name}' (AUPRC={best_auprc:.4f}) saved to {model_path}")
        return best_name, best_model

        
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
        """Logs all generated artifacts (plots, CSVs, best model) to a final MLflow run."""
        if not self.model_scores:
            logger.warning("No model scores available. Skipping MLflow artifact logging.")
            return

        best_name = max(self.model_scores, key=self.model_scores.get)
        best_auprc = self.model_scores[best_name]

        with mlflow.start_run(run_name=f"Best-Model-{best_name}"):
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

            logger.info(f"Logged all experiment artifacts to MLflow. Best model: {best_name}")
