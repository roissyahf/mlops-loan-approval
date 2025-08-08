import mlflow
import dagshub
import os
from dotenv import load_dotenv
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import warnings

load_dotenv()

# set dagshub repository
dagshub.init(repo_owner='roissyahfk', repo_name='MLOps-Loan-Approval', mlflow=True)

# set tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# set experiment name
mlflow.set_experiment("Loan Approval Model Experiment")

# read train and test set
train_path = os.path.join(os.getcwd(), '../data/processed/train_data.csv')
test_path = os.path.join(os.getcwd(), '../data/processed/test_data.csv')
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# add pipeline to process features standarization & encoding in train set
## split categorical and numerical features
num_cols = df_train[['person_income', 'loan_int_rate', 'credit_score', 'loan_amnt',
                    'loan_percent_income', 'person_age']].columns.tolist()
cat_cols = ['previous_loan_defaults_on_file']

## initialize encoder & standarization
binary_transformer = OrdinalEncoder(categories=[['No', 'Yes']])
numerical_transformer = StandardScaler()
    
## set up processing pipeline
preprocessor = ColumnTransformer(transformers=[
                ('cat', binary_transformer, cat_cols),
                ('num', numerical_transformer, num_cols)])

# split features and target variable
X_train = df_train.drop(columns=['loan_status'])
y_train = df_train['loan_status']
X_test = df_test.drop(columns=['loan_status'])
y_test = df_test['loan_status']

# take input example
input_example = X_train[0:5]

# XGB best parameters (from model tuning result xgb_tuning_t2_13)
n_estimators = 50 #
learning_rate = 0.027575 #
max_depth = 28 #
subsample = 0.5 #
colsample_bytree = 0.8 #
gamma = 0.5 #
reg_alpha = 0 #
reg_lambda = 3.968421 #
min_child_weight = 7 #
random_state = 42

# ignore warnings
warnings.filterwarnings("ignore")

# start MLflow run
with mlflow.start_run(run_name="XGB best-model-params"):

    # use autolog
    mlflow.autolog()

    # create model
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        eval_metric='logloss',    # for binary classification
        random_state=random_state)
    
    # create model pipeline
    clc_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
            ])

    # start training
    clc_pipeline.fit(X_train, y_train)

    # log model
    mlflow.sklearn.log_model(
        sk_model=clc_pipeline,
        artifact_path="model",
        input_example=input_example,
        registered_model_name="XGB best-model-params"
    )

    # log metrics
    accuracy = clc_pipeline.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    