from dotenv import load_dotenv
import os
import mlflow
import dagshub
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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

# split features and target variable
X_train = df_train.drop(columns=['loan_status'])
y_train = df_train['loan_status']
X_test = df_test.drop(columns=['loan_status'])
y_test = df_test['loan_status']

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

# take input example
input_example = X_train[0:5]

# log datasets
dataset_version = "v1.0"
dataset_path = "../data/processed/train_data.csv"

# create local directory if it doesn't exist
os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# set random seed for reproducibility
np.random.seed(42)

# define hyperparameter search space
n_estimators_range = np.arange(10, 1001)         # integers from 10 to 1000
max_depth_range = np.arange(1, 51)               # integers from 1 to 50
learning_rate_range = np.linspace(0.01, 0.3, 100)  # float values
subsample_range = np.linspace(0.5, 1.0, 6)
colsample_bytree_range = np.linspace(0.5, 1.0, 6)
gamma_range = np.linspace(0, 5, 11)
reg_alpha_range = np.linspace(0, 1, 11)
reg_lambda_range = np.linspace(0.1, 5, 20)
min_child_weight_range = np.arange(1, 11)

# number of random search iterations
n_iterations = 20

best_precision = 0
best_params = {}

for i in range(n_iterations):
    # randomly select hyperparameters
    n_estimators = np.random.choice(n_estimators_range)
    max_depth = np.random.choice(max_depth_range)
    learning_rate = np.random.choice(learning_rate_range)
    subsample = np.random.choice(subsample_range)
    colsample_bytree = np.random.choice(colsample_bytree_range)
    gamma = np.random.choice(gamma_range)
    reg_alpha = np.random.choice(reg_alpha_range)
    reg_lambda = np.random.choice(reg_lambda_range)
    min_child_weight = np.random.choice(min_child_weight_range)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "gamma": gamma,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "min_child_weight": min_child_weight
    }

    run_name = f"xgb_random_search_t2_{i+1}"
    with mlflow.start_run(run_name=run_name):
        # Train XGB model
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            eval_metric='logloss',
            random_state=42)

        # create model pipeline
        clc_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
            ])
        
        # start training
        clc_pipeline.fit(X_train, y_train)

        # Evaluate model on test set
        accuracy = clc_pipeline.score(X_test, y_test)
        y_pred = clc_pipeline.predict(X_test)
        y_pred_proba = clc_pipeline.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
    
        # MANUAL LOGGING
        # log dataset
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("dataset_path", dataset_path)

        # log model parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("reg_alpha", reg_alpha)
        mlflow.log_param("reg_lambda", reg_lambda)
        mlflow.log_param("min_child_weight", min_child_weight)

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("auc", auc)

        # log model
        ## save model with a unique name
        model_filename = f"model/XGB_tuning_t2_{i+1}.joblib"
        joblib.dump(clc_pipeline, model_filename)

        mlflow.sklearn.log_model(
            sk_model=clc_pipeline,
            artifact_path="model",
            input_example=input_example)

        ## log artifacts (data, model)
        mlflow.log_artifact(dataset_path, artifact_path="datasets")
        mlflow.log_artifact(model_filename, artifact_path="model_artifacts")

        # log matrix plot
        ## log confusion matrix
        conf_matrix_name = f"plots/confusion_matrix_t2_{i+1}.png"
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Oranges")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(conf_matrix_name)
        plt.close()

        ## log feature importance plot
        feat_importance_name = f"plots/feature_importance_t2_{i+1}.png"
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(7))
        plt.title('Top 7 Feature Importance')
        plt.savefig(feat_importance_name)
        plt.close()

        ## log artifacts (plots)
        mlflow.log_artifact(conf_matrix_name)
        mlflow.log_artifact(feat_importance_name)

        # Save the best model
        if precision > best_precision:
            best_precision = precision
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            }
            mlflow.sklearn.log_model(
                sk_model=clc_pipeline,
                artifact_path="model",
                input_example=input_example
            )

print(f"Best Precision: {best_precision}")
print(f"Best Parameters: {best_params}")