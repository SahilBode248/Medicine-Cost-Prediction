"""
Vercel Python Serverless Function
POST /api/predict
Body (JSON): the raw form fields listed in FEATURE_SPEC below.
Returns: {"predicted_payable_cost": <float>}

The model is loaded once at import time so warm invocations reuse it
instead of re-loading the ~9MB joblib file on every request.
"""

import json
import os
from http.server import BaseHTTPRequestHandler

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.joblib")
_pipeline = joblib.load(MODEL_PATH)

REQUIRED_FIELDS = [
    "Age", "Gender", "BMI", "Chronic_Condition", "Medicine_Name", "Category",
    "Type", "Generic_or_Branded", "Manufacturer", "Dosage_mg", "Quantity",
    "Duration_Days", "Severity", "Treatment_Type", "Region",
    "Insurance_Pct", "Discount_Pct",
]

NUMERIC_FIELDS = [
    "Age", "BMI", "Dosage_mg", "Quantity", "Duration_Days",
    "Insurance_Pct", "Discount_Pct",
]


def engineer_features(row: dict) -> pd.DataFrame:
    """Same derivation used at training time — keep these two in sync."""
    df = pd.DataFrame([row])

    age_bins = [0, 18, 35, 50, 65, 999]
    age_labels = ["Child", "Young Adult", "Adult", "Senior Adult", "Elderly"]
    df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

    bmi_bins = [0, 18.5, 24.9, 29.9, 999]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["BMI_Category"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels, right=True)

    df["Has_Insurance"] = (df["Insurance_Pct"] > 0).astype(int)
    return df


def predict(payload: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    row = dict(payload)
    for f in NUMERIC_FIELDS:
        try:
            row[f] = float(row[f])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{f}' must be numeric")

    df = engineer_features(row)
    log_pred = _pipeline.predict(df)[0]
    predicted_cost = float(np.expm1(log_pred))
    predicted_cost = max(predicted_cost, 0.0)

    return {"predicted_payable_cost": round(predicted_cost, 2)}


class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        self._set_headers(200)
        self.wfile.write(json.dumps({
            "status": "ok",
            "message": "POST patient/medicine details as JSON to this endpoint to get a predicted payable cost.",
            "required_fields": REQUIRED_FIELDS,
        }).encode())

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body or b"{}")
            result = predict(payload)
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode())
        except ValueError as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": f"Internal error: {str(e)}"}).encode())
