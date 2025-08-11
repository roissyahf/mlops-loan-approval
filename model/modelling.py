import mlflow
import dagshub
import os
from dotenv import load_dotenv
import pandas as pd
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap
import scipy.sparse as sp


# Load environment variables
load_dotenv()

# Init dagshub connection
dagshub.init(
    repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"),
    repo_name="MLOps-Loan-Approval",
    mlflow=True
)

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

# Suppress warnings
warnings.filterwarnings("ignore")

# Load data
train_path = os.path.join(os.getcwd(), 'data/processed/train_data.csv')
test_path = os.path.join(os.getcwd(), 'data/processed/test_data.csv')
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Feature columns
num_cols = ['person_income', 'loan_int_rate', 'credit_score', 'loan_amnt',
            'loan_percent_income', 'person_age']
cat_cols = ['previous_loan_defaults_on_file']

# Preprocessing pipeline
binary_transformer = OrdinalEncoder(categories=[['No', 'Yes']])
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
    ('cat', binary_transformer, cat_cols),
    ('num', numerical_transformer, num_cols)
])

# Feature-target split
X_train = df_train.drop(columns=['loan_status'])
y_train = df_train['loan_status']
X_test = df_test.drop(columns=['loan_status'])
y_test = df_test['loan_status']

# Input example
input_example = X_train.iloc[0:5]

# Best model parameters
params = {
    'n_estimators': 50,
    'learning_rate': 0.027575,
    'max_depth': 28,
    'subsample': 0.5,
    'colsample_bytree': 0.8,
    'gamma': 0.5,
    'reg_alpha': 0,
    'reg_lambda': 3.968421,
    'min_child_weight': 7,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# Avoid using autologging due to library error dependency
if mlflow.active_run():
    mlflow.end_run()

# Start MLflow run
with mlflow.start_run(run_name="XGB-best-model-manual-logging"):

    # Log parameters manually
    mlflow.log_params(params)

    # Build pipeline
    model = XGBClassifier(**params)
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Fit model
    clf_pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf_pipeline.predict(X_test)
    y_pred_proba = clf_pipeline.predict_proba(X_test)[:, 1]

    accuracy = clf_pipeline.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred_proba)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1)
    mlflow.log_metric("auc", auc)

    # Log matrix plot
    ## log confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Oranges")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "plots/confusion_matrix.png")
    plt.close()

    ## log feature importance plot
    feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(7))
    plt.title('Top 7 Feature Importance')
    mlflow.log_figure(plt.gcf(), "plots/feature_importance.png")
    plt.close()

    ## log SHAP value plot
    ### access fitted steps
    pre = clf_pipeline.named_steps["preprocessor"]
    xgb = clf_pipeline.named_steps["classifier"]
    
    ### transform X_test to what the classifier actually saw
    X_test_tx = pre.transform(X_test)
    if sp.issparse(X_test_tx):
        X_test_tx = X_test_tx.toarray()

    ### feature names (works for ColumnTransformer in sklearn >=1.0)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_test_tx.shape[1])]

    ### see a manageable sample for speed
    rng = np.random.RandomState(42)
    n = min(300, X_test_tx.shape[0])
    sample_idx = rng.choice(X_test_tx.shape[0], size=n, replace=False)
    X_sample = X_test_tx[sample_idx]

    ### SHAP values for XGBoost classifier
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)

    ### SHAP plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "plots/shap_beeswarm.png")
    plt.close()

    # Save model locally
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf_pipeline, "model/model.pkl")

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=clf_pipeline, #
        artifact_path="model",
        input_example=input_example,
        registered_model_name="XGB-best-model-manual"
    )

    print(f"Run completed. Accuracy: {accuracy:.4f}. F1 Score: {f1:.4f}")
