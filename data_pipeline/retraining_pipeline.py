import os
import pandas as pd
import mlflow
from datetime import datetime
import json
import subprocess
import logging
import joblib
from convert_logs import convert_jsonl_to_csv, combine_with_training_data
from simple_preprocessing import preprocess_data
from modelling_refactor import train_and_evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrainingPipeline:
    def __init__(self, 
                 original_data_path="data/processed/train_data.csv",
                 new_data_path="data/simulation/current.jsonl",
                 model_output_path="model/model.pkl"):
        
        self.original_data_path = original_data_path
        self.new_data_path = new_data_path
        self.model_output_path = model_output_path
        self.retrain_data_path = "data/processed/retrain_data.csv"
        
        # MLflow setup
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', ''))
        mlflow.set_experiment("loan-retraining")


    def should_retrain(self):
        """Determine if retraining is needed"""
        # Check 1: Minimum data threshold
        if not os.path.exists(self.new_data_path):
            logger.info("No new data found")
            return False, "No new data"
            
        # Count new predictions
        with open(self.new_data_path, 'r') as f:
            new_records = sum(1 for line in f)
            
        if new_records < 5:  # Minimum threshold: set 5 for testing
            logger.info(f"Only {new_records} new records, need at least 5")
            return False, f"Insufficient data: {new_records} records"
            
        # Check 2: Time-based (every 4 weeks)
        try:
            last_retrain = self.get_last_retrain_date()
            days_since = (datetime.now() - last_retrain).days
            
            if days_since >= 28:  # 4 weeks
                return True, f"Scheduled retrain: {days_since} days since last"
                
        except Exception as e:
            logger.warning(f"Could not check last retrain date: {e}")
            return True, "First time retraining"
            
        return False, "No retraining needed"


    def get_last_retrain_date(self):
        """Get date of last retraining from MLflow"""
        try:
            experiment = mlflow.get_experiment_by_name("loan-retraining")
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                    order_by=["start_time DESC"], 
                                    max_results=1)
            
            if not runs.empty:
                return pd.to_datetime(runs.iloc[0]['start_time'])
            else:
                return datetime(2025, 7, 1)  # Default old date
                
        except Exception:
            return datetime(2025, 7, 1)


    def prepare_data(self):
        """Convert JSONL to CSV and combine with original data"""
        
        # Convert new data
        new_csv_path = "data/processed/new_predictions.csv"
        converted_path = convert_jsonl_to_csv(self.new_data_path, new_csv_path)
        
        if not converted_path:
            raise ValueError("Failed to convert JSONL to CSV")
            
        # Combine with original training data
        combined_path = combine_with_training_data(
            new_csv_path, self.original_data_path, self.retrain_data_path)
        
        return combined_path


    def retrain_model(self):
        """Execute the retraining pipeline"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("retrain_date", datetime.now().isoformat())
            mlflow.log_param("original_data", self.original_data_path)
            mlflow.log_param("new_data_source", self.new_data_path)
            
            # Prepare data
            logger.info("Preparing retraining data...")
            retrain_data_path = self.prepare_data()
            
            # Load and preprocess
            data = pd.read_csv(retrain_data_path)
            logger.info(f"Retraining with {len(data)} total records")
            
            X, y = preprocess_data(data) # reuse existing preprocessing function, ensure it's available
            
            # Train model (reuse existing training function, ensure it's available)
            model, metrics = train_and_evaluate_model(X, y)
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Save local model file
            joblib.dump(model, self.model_output_path)
            
            # Track with DVC
            self.track_with_dvc()
            
            logger.info("Retraining completed successfully!")
            return model, metrics


    def track_with_dvc(self):
        """Track new model and data with DVC"""
        try:
            # Track retrain data
            subprocess.run(["dvc", "add", self.retrain_data_path], check=True)
            
            # Track new model
            subprocess.run(["dvc", "add", self.model_output_path], check=True)
            
            # Commit to git
            subprocess.run(["git", "add", 
                          f"{self.retrain_data_path}.dvc", 
                          f"{self.model_output_path}.dvc"], check=True)
            
            commit_msg = f"Retrain model {datetime.now().strftime('%Y-%m-%d')}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            
            # Push to DVC remote
            subprocess.run(["dvc", "push"], check=True)
            
            logger.info("Model and data tracked with DVC")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC tracking failed: {e}")

    def run(self):
        """Main retraining pipeline"""
        logger.info("Starting retraining pipeline...")
        
        # Check if retraining is needed
        should_retrain, reason = self.should_retrain()
        logger.info(f"Retrain decision: {should_retrain}, reason: {reason}")
        
        if not should_retrain:
            return {"status": "skipped", "reason": reason}
        
        try:
            # Execute retraining
            model, metrics = self.retrain_model()
            
            return {
                "status": "success", 
                "reason": reason,
                "metrics": metrics,
                "model_path": self.model_output_path
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {"status": "failed", "error": str(e)}



if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    result = pipeline.run()
    print(json.dumps(result, indent=2))