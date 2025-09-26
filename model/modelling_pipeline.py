import os
import pandas as pd
import mlflow
from datetime import datetime
import json
import subprocess
import logging
import joblib
from simple_preprocessing import preprocess_data
from modelling_refactor import train_and_evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, 
                 original_data_path="data/processed/train_data.csv",
                 new_data_path="data/simulation/current.jsonl",
                 model_output_path="model/model.pkl"):
        
        self.original_data_path = original_data_path
        self.new_data_path = new_data_path
        self.model_output_path = model_output_path
        
        # MLflow setup
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', ''))
        mlflow.set_experiment("loan-training")


    def train_model(self):
        """Execute the training pipeline"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("train_date", datetime.now().isoformat())
            mlflow.log_param("original_data", self.original_data_path)
            
            # Prepare data
            logger.info("Preparing training data...")
            train_data_path = self.prepare_data()
            
            # Load and preprocess
            data = pd.read_csv(train_data_path)
            logger.info(f"Training with {len(data)} total records")
            
            X, y = preprocess_data(data) # reuse existing preprocessing function
            
            # Train model (reuse existing training function)
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
            
            logger.info("Training completed successfully!")
            return model, metrics


    def track_with_dvc(self):
        """Track new model and data with DVC"""
        try:
            # Track train data
            subprocess.run(["dvc", "add", self.train_data_path], check=True)

            # Track new model
            subprocess.run(["dvc", "add", self.model_output_path], check=True)
            
            # Commit to git
            subprocess.run(["git", "add", 
                          f"{self.retrain_data_path}.dvc", 
                          f"{self.model_output_path}.dvc"], check=True)
            
            commit_msg = f"Train model {datetime.now().strftime('%Y-%m-%d')}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            
            # Push to DVC remote
            subprocess.run(["dvc", "push"], check=True)
            
            logger.info("Model and data tracked with DVC")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC tracking failed: {e}")

    def run(self):
        """Main training pipeline"""
        logger.info("Starting training pipeline...")
        
        try:
            # Execute training
            model, metrics = self.train_model()

            return {
                "status": "success",
                "metrics": metrics,
                "model_path": self.model_output_path
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "failed", "error": str(e)}



if __name__ == "__main__":
    pipeline = TrainingPipeline()
    result = pipeline.run()
    print(json.dumps(result, indent=2))