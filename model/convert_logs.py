import pandas as pd
import json
import os
from datetime import datetime

def convert_jsonl_to_csv(jsonl_path, output_path):
    """Convert JSONL prediction logs to CSV for retraining"""
    
    if not os.path.exists(jsonl_path):
        print(f"JSONL file not found: {jsonl_path}")
        return None
    
    # Read JSONL
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                continue
    
    if not data:
        print("No valid data found in JSONL")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter only prediction features (exclude metadata)
    feature_cols = [col for col in df.columns if col in 
                   ["person_age", "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income",
        "credit_score", "previous_loan_defaults_on_file", "prediction"]]
    
    df_features = df[feature_cols]
    
    # Remove duplicates and nulls
    df_clean = df_features.dropna().drop_duplicates()

    # Rename prediction column to match training data
    if 'prediction' in df_clean.columns:
        df_clean = df_clean.rename(columns={'prediction': 'loan_status'})
    
    # Save
    df_clean.to_csv(output_path, index=False)
    print(f"Converted {len(df_clean)} records to {output_path}")
    
    return output_path


def combine_with_training_data(new_data_path, original_data_path, output_path):
    """Combine new data with original training data"""
    
    # Load datasets
    new_df = pd.read_csv(new_data_path)
    original_df = pd.read_csv(original_data_path)
    
    # Ensure same columns
    common_cols = list(set(new_df.columns) & set(original_df.columns))
    new_df = new_df[common_cols]
    original_df = original_df[common_cols]
    
    # Combine
    combined_df = pd.concat([original_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    
    # Save
    combined_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved: {len(combined_df)} total records")
    
    return output_path

if __name__ == "__main__":
    # Test locally
    jsonl_path = "data/simulation/current.jsonl"
    new_csv = "data/processed/new_predictions.csv"
    retrain_csv = "data/processed/retrain_data.csv"
    original_csv = "data/processed/train_data.csv"
    
    # Convert and combine
    if convert_jsonl_to_csv(jsonl_path, new_csv):
        combine_with_training_data(new_csv, original_csv, retrain_csv)