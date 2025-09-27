import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

class _PreprocessData:
    """
    Callable preprocessor:
      - __call__(df) -> (X, y)  for in-memory pipelines
      - run_from_csv(input_csv, train_out, test_out) -> writes two CSVs
    """

    def __init__(self, target_col: str = "loan_status"):
        self.target_col = target_col
        # columns to drop BEFORE split
        self._drop_cols = [
            "person_gender",
            "person_education",
            "person_emp_exp",
            "person_home_ownership",
            "loan_intent",
            "cb_person_cred_hist_length",
            self.target_col,
        ]

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna().drop_duplicates()
        # simple outlier trimming
        if "person_age" in df.columns:
            df = df[df["person_age"] <= 70]
        if "person_income" in df.columns:
            df = df[df["person_income"] <= 3_000_000]
        if "person_emp_exp" in df.columns:
            df = df[df["person_emp_exp"] <= 40]
        return df

    def __call__(self, df: pd.DataFrame):
        """
        Accepts a DataFrame and returns (X, y) for training/retraining.
        """
        df = self._basic_clean(df.copy())

        # Ensure target exists
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data.")

        # Build X, y with existing feature selection
        keep_cols = [c for c in df.columns if c not in self._drop_cols]
        X = df[keep_cols]
        y = df[self.target_col]
        return X, y

    def run_from_csv(
        self,
        input_csv: str | os.PathLike,
        train_out: str | os.PathLike = PROC_DIR / "train_data.csv",
        test_out: str | os.PathLike = PROC_DIR / "test_data.csv",
        test_size: float = 0.25,
        random_state: int = 42,
    ):
        """
        Accept a single CSV, clean, split, and write 2 CSVs (train/test)
        with the TARGET column included in each output.
        """
        df_raw = pd.read_csv(input_csv)
        df_clean = self._basic_clean(df_raw)

        keep_cols = [c for c in df_clean.columns if c not in self._drop_cols]
        X = df_clean[keep_cols]
        y = df_clean[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        pd.concat([X_train, y_train], axis=1).to_csv(train_out, index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(test_out, index=False)

        return str(train_out), str(test_out)

preprocess_data = _PreprocessData()

if __name__ == "__main__":
    raw_csv = REPO_ROOT / "data" / "raw" / "loan_data.csv"
    preprocess_data.run_from_csv(raw_csv)
