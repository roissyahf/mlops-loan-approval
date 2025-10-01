import os
import sys
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy.sparse as sp

import logging

# Load environment variables
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PKL = REPO_ROOT / "model" / "model.pkl"
MODEL_PKL.parent.mkdir(exist_ok=True, parents=True)

class _TrainAndEvaluateModel:
    """
      __call__(X, y) -> (fitted_pipeline, metrics_dict)
      Logs params/metrics/artifacts to MLflow (DagsHub) and writes model/model.pkl
    """
    def __init__(self):
        # Dagshub and MLflow credentials will be passed via environment variables
        dagshub.init(
            repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"),
            repo_name="MLOps-Loan-Approval",
            mlflow=True
            )
        
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "loan-default")
        mlflow.set_experiment(exp_name)
        warnings.filterwarnings("ignore")

        # Default columns
        self.num_cols = [
            "person_income", "loan_int_rate", "credit_score",
            "loan_amnt", "loan_percent_income", "person_age"
        ]
        self.cat_cols = ["previous_loan_defaults_on_file"]

        # XGBoost params
        self.params = {
            "n_estimators": 50,
            "learning_rate": 0.027575,
            "max_depth": 28,
            "subsample": 0.5,
            "colsample_bytree": 0.8,
            "gamma": 0.5,
            "reg_alpha": 0,
            "reg_lambda": 3.968421,
            "min_child_weight": 7,
            "random_state": 42,
            "eval_metric": "logloss",
        }

    def __call__(self, X: pd.DataFrame, y: pd.Series, run_name: str = "train-eval"):
        if mlflow.active_run():
            mlflow.end_run()

        # Splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Build preprocessors based on columns actually exist
        num_cols_present = [c for c in self.num_cols if c in X_train.columns]
        cat_cols_present = [c for c in self.cat_cols if c in X_train.columns]

        binary_transformer = OrdinalEncoder(categories=[['No', 'Yes']]) if cat_cols_present else "drop"
        numerical_transformer = StandardScaler() if num_cols_present else "drop"

        preprocessor = ColumnTransformer(transformers=[
            ('cat', binary_transformer, cat_cols_present),
            ('num', numerical_transformer, num_cols_present)
        ])

        model = XGBClassifier(**self.params)
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        # Input example
        input_example = X_train.iloc[0:5]

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(self.params)

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

            accuracy = float(clf.score(X_test, y_test))
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else float("nan")

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "auc": auc,
            }
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Confusion Matrix
            conf_mat = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Oranges")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "plots/confusion_matrix.png")
            plt.close()

            # SHAP
            try:
                pre = clf.named_steps["preprocessor"]
                xgb = clf.named_steps["classifier"]
                X_test_tx = pre.transform(X_test)
                if sp.issparse(X_test_tx):
                    X_test_tx = X_test_tx.toarray()
                try:
                    feature_names = pre.get_feature_names_out()
                except Exception:
                    feature_names = [f"f{i}" for i in range(X_test_tx.shape[1])]
                rng = np.random.RandomState(42)
                n = min(300, X_test_tx.shape[0])
                X_sample = X_test_tx[rng.choice(X_test_tx.shape[0], size=n, replace=False)]
                explainer = shap.TreeExplainer(xgb)
                shap_values = explainer.shap_values(X_sample)
                plt.figure()
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.tight_layout()
                mlflow.log_figure(plt.gcf(), "plots/shap_beeswarm.png")
                plt.close()
            except Exception as e:
                logging.error(f"SHAP plotting failed: {e}")
                pass

            # Save model for runtime service
            joblib.dump(clf, MODEL_PKL)

            # Log to MLflow
            input_example = X_train.iloc[:5].copy()
            model_info = mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                input_example=input_example,
                registered_model_name="XGB-retraining"
            )

            # Promote to Production stage
            client = MlflowClient()
            client.transition_model_version_stage(
                name="XGB-retraining", # this is MLFLOW_MODEL_NAME
                version=model_info.registered_model_version,
                stage="Production" # this is MLFLOW_MODEL_STAGE
                )

            logging.info(f"[train] run_id={run.info.run_id} AUC={auc:.4f} F1={f1:.4f}")
            return clf, metrics

train_and_evaluate_model = _TrainAndEvaluateModel()

def fetch_new_data_from_bigquery(bq_table_id: str) -> pd.DataFrame:
    """
    Fetches new prediction events from BigQuery for a given number of days.
    """
    logging.info(f"Fetching new data from BigQuery view: {bq_table_id}...")
    client = bigquery.Client(project=os.getenv("GCP_PROJECT_ID"))
    
    query = f"""
        SELECT
            person_age,
            person_income,
            loan_amnt,
            loan_percent_income,
            loan_int_rate,
            credit_score,
            loan_amnt,
            previous_loan_defaults_on_file,
            loan_status
        FROM `{bq_table_id}`
    """
    
    try:
        query_job = client.query(query)
        df = query_job.to_dataframe()
        logging.info(f"Successfully fetched {len(df)} new records.")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch data from BigQuery: {e}")
        sys.exit(1)

def main():
    """
    Main function to execute the automated retraining pipeline.
    """
    logging.info("Starting automated retraining pipeline.")
    
    # Check for required environment variables
    logging.info("GCP Project ID: %s", os.getenv("GCP_PROJECT_ID"))
    logging.info("BigQuery Dataset: %s", os.getenv("BIGQUERY_DATASET"))

    # 1. Pull original training data from DVC
    # GitHub Actions workflow handles `dvc pull` before running this script
    logging.info("DVC pull completed by GitHub Actions. Reading original data...")
    original_data_path = "data/processed/train_data.csv" 
    if not os.path.exists(original_data_path):
        logging.error(f"Original data file not found at {original_data_path}. Ensure your DVC setup is correct.")
        sys.exit(1)
        
    original_df = pd.read_csv(original_data_path)
    logging.info(f"Original dataset size: {len(original_df)} records.")

    # 2. Fetch new data from BigQuery
    BQ_TABLE_VIEW_ID = f"{os.getenv('GCP_PROJECT_ID')}.{os.getenv('BIGQUERY_DATASET')}.prediction_events_view"
    new_data = fetch_new_data_from_bigquery(BQ_TABLE_VIEW_ID)
    
    # 3. Combine data, ensuring the schema matches for both.
    # BigQuery query maps the `prediction` field to `loan_status` to match the original data.
    logging.info("Combining original and new data...")
    combined_df = pd.concat([original_df, new_data], ignore_index=True)
    
    # Drop duplicates in case there's overlap in logs and original dataset
    combined_df.drop_duplicates(inplace=True)
    logging.info(f"Combined dataset size: {len(combined_df)} records.")
    
    # 4. Train the new model using the provided class
    X = combined_df.drop(columns=["loan_status"])
    y = combined_df["loan_status"]

    try:
        model, metrics = train_and_evaluate_model(X, y, run_name=f"retrain-{datetime.now().strftime('%Y-%m-%d')}")
        logging.info("Retraining completed successfully!")
    except Exception as e:
        logging.error(f"Retraining failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
