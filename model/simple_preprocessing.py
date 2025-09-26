from __future__ import annotations
import pandas as pd
from typing import Tuple, Iterable
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


class PreprocessData:
    """
    Unified preprocessing:
      - __call__(df) -> (X, y)
      - from_csv(path) -> (X, y)
      - build_preprocessor() -> sklearn ColumnTransformer
    """

    NUM_COLS: list[str] = [
        "person_income", "loan_int_rate", "credit_score",
        "loan_amnt", "loan_percent_income", "person_age"
    ]
    CAT_COLS: list[str] = ["previous_loan_defaults_on_file"]
    TARGET_COL: str = "loan_status"

    def __init__(self):
        self.preprocessor = self.build_preprocessor()

    def _require_columns(self, df: pd.DataFrame, cols: Iterable[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __call__(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Accepts a DataFrame and returns (X, y)."""
        self._require_columns(df, self.NUM_COLS + self.CAT_COLS + [self.TARGET_COL])
        X = df[self.CAT_COLS + self.NUM_COLS].copy()
        y = df[self.TARGET_COL].copy()
        return X, y

    def from_csv(self, csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Helper: read a CSV and return (X, y)."""
        df = pd.read_csv(csv_path)
        return self(df)

    def build_preprocessor(self) -> ColumnTransformer:
        binary = OrdinalEncoder(
            categories=[['No', 'Yes']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        numeric = StandardScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', binary, self.CAT_COLS),
                ('num', numeric, self.NUM_COLS),
            ],
            remainder='drop'
        )
        return preprocessor


# Expose a singleton-style instance to import it directly
preprocess_data = PreprocessData()
