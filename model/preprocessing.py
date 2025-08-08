import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split


input_file_path = os.path.join(os.getcwd(), '../data/raw/loan_data.csv')
output_dir = os.path.join(os.getcwd(), "../data/processed")
os.makedirs(output_dir, exist_ok=True)
train_file_path = os.path.join(output_dir, "train_data.csv")
test_file_path = os.path.join(output_dir, "test_data.csv")


def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset successfully loaded with shape: {df.shape}")
    return df


def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Removed NaN and duplicate rows. New shape: {df.shape}")
    return df


def handle_outliers(df):
    df = df[
        (df['person_age'] <= 70)
        & (df['person_income'] <= 3000000)
        & (df['person_emp_exp'] <= 40)
    ]
    print(f"Outliers removed. New shape: {df.shape}")
    return df


def split_data(df, target_col='loan_status'):
    X = df.drop(columns=['person_gender', 'person_education',
       'person_emp_exp', 'person_home_ownership', 'loan_intent',
       'cb_person_cred_hist_length', target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def save_to_csv(X_train, X_test, y_train, y_test):
    pd.concat([X_train, y_train], axis=1).to_csv(train_file_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_file_path, index=False)
    print("Data saved to data/processed/train_data.csv and data/processed/test_data.csv")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    df_raw = load_data(input_file_path)
    df_cleaned = clean_data(df_raw)
    df_no_outliers = handle_outliers(df_cleaned)

    X_train, X_test, y_train, y_test = split_data(df_no_outliers) 

    # we will handle feature standarization & feature encoding in modeling pipeline directly
    save_to_csv(X_train, X_test, y_train, y_test)