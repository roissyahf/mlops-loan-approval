import pandas as pd
import json
import os

def convert_jsonl_to_csv(jsonl_path, output_path):
    """Convert JSONL prediction logs to CSV for drift monitoring"""
    
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
    
    # Filter only prediction features (exclude metadata, prediction result)
    feature_cols = [col for col in df.columns if col in 
                   ["person_age", "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income",
        "credit_score", "previous_loan_defaults_on_file"]]
    
    df_features = df[feature_cols]
    
    # Remove duplicates and nulls
    df_clean = df_features.dropna().drop_duplicates()
    
    # Save
    df_clean.to_csv(output_path, index=False)
    print(f"Converted {len(df_clean)} records to {output_path}")
    
    return output_path



if __name__ == "__main__":
    # Test locally
    jsonl_path = "data/simulation/current.jsonl"
    new_csv = "data/simulation/prediction_dev_log.csv"
    
    # Convert
    if convert_jsonl_to_csv(jsonl_path, new_csv):
        print(f"Conversion successful: {new_csv}")