"""
train_model.py
----------------
Trains a Random Forest model that predicts the Payable_Cost of a medicine
treatment from the details that are actually known BEFORE the cost is
computed (patient info, medicine info, treatment info, insurance/discount %).

Note on feature selection (important):
Total_Cost, Cost_per_Day and Cost_per_Unit are all derived FROM Total_Cost,
which is only a few arithmetic steps away from Payable_Cost
(Payable_Cost = Total_Cost * (1 - Discount%) * (1 - Insurance%), roughly).
Using them as inputs would leak the answer and isn't realistic for a form a
user fills out before treatment, so this script intentionally excludes them
and predicts Payable_Cost straight from the raw patient/medicine/treatment
fields. This mirrors the notebook's feature engineering (Age_Group,
BMI_Category, Has_Insurance) but keeps only inputs a real user could supply.

Run:  python train_model.py
Outputs:
  model/model.joblib     -> trained sklearn Pipeline (preprocessing + model)
  model/metadata.json    -> dropdown options + feature list for the frontend
  model/metrics.json     -> evaluation metrics
"""

import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

RANDOM_STATE = 42
DATA_PATH = "data.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive extra features purely from raw input columns (no leakage)."""
    df = df.copy()

    age_bins = [0, 18, 35, 50, 65, 999]
    age_labels = ["Child", "Young Adult", "Adult", "Senior Adult", "Elderly"]
    df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

    bmi_bins = [0, 18.5, 24.9, 29.9, 999]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["BMI_Category"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels, right=True)

    df["Has_Insurance"] = (df["Insurance_Pct"] > 0).astype(int)

    return df


NUMERIC_FEATURES = [
    "Age", "BMI", "Dosage_mg", "Quantity", "Duration_Days",
    "Insurance_Pct", "Discount_Pct", "Has_Insurance",
]

CATEGORICAL_FEATURES = [
    "Gender", "Chronic_Condition", "Medicine_Name", "Category", "Type",
    "Generic_or_Branded", "Manufacturer", "Severity", "Treatment_Type",
    "Region", "Age_Group", "BMI_Category",
]

TARGET = "Payable_Cost"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = np.log1p(df[TARGET])  # log-transform the (skewed) target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    # Evaluate (convert predictions back to real cost scale with expm1)
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print("Evaluation on held-out test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    joblib.dump(pipeline, "model/model.joblib", compress=3)
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Metadata for building the frontend form (dropdown options, min/max, etc.)
    metadata = {
        "numeric_features": {
            "Age": {"min": int(df["Age"].min()), "max": int(df["Age"].max())},
            "BMI": {"min": float(df["BMI"].min()), "max": float(df["BMI"].max())},
            "Dosage_mg": {"min": float(df["Dosage_mg"].min()), "max": float(df["Dosage_mg"].max())},
            "Quantity": {"min": int(df["Quantity"].min()), "max": int(df["Quantity"].max())},
            "Duration_Days": {"min": int(df["Duration_Days"].min()), "max": int(df["Duration_Days"].max())},
            "Insurance_Pct": {"min": 0, "max": 100},
            "Discount_Pct": {"min": 0, "max": 100},
        },
        "categorical_features": {
            col: sorted(df[col].dropna().unique().tolist())
            for col in [
                "Gender", "Chronic_Condition", "Medicine_Name", "Category", "Type",
                "Generic_or_Branded", "Manufacturer", "Severity", "Treatment_Type", "Region",
            ]
        },
    }
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved: model/model.joblib, model/metadata.json, model/metrics.json")


if __name__ == "__main__":
    main()
