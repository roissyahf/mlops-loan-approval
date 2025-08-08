import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import time

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict(input_dict):
    start_time = time.time()

    input_df = pd.DataFrame([input_dict])
    features = input_df[[
        "person_age",
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "credit_score",
        "previous_loan_defaults_on_file"
    ]]

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0].tolist() if hasattr(model, "predict_proba") else []

    latency = time.time() - start_time
    return {
        "prediction": int(prediction),
        "probabilities": probabilities,
        "latency": latency
    }
