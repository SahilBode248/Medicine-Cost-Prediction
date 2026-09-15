# Medicine Treatment Cost Estimator

Predicts the **Payable_Cost** of a treatment (Random Forest, R² ≈ 0.93 on held-out
test data) from patient, medicine, and treatment details — deployable on Vercel
as static frontend + Python serverless functions.

## What changed vs. the notebook

The notebook computed `Cost_per_Day`, `Cost_per_Unit`, and used `Total_Cost` in
the same feature-engineering pipeline as `Payable_Cost`. Those are all derived
from `Total_Cost`, which is only one arithmetic step from `Payable_Cost` — so
using them as model inputs leaks the answer and isn't something a real user
would type into a form anyway. `train_model.py` retrains a model that predicts
`Payable_Cost` directly from the raw inputs only (age, medicine, dosage,
quantity, duration, severity, insurance %, discount %, etc.), keeping the
notebook's `Age_Group` / `BMI_Category` / `Has_Insurance` engineered features
since those only depend on inputs. This is the version that's actually usable
in production.

## Project structure

```
├── api/
│   ├── predict.py       # POST /api/predict  -> {"predicted_payable_cost": ...}
│   └── metadata.py      # GET  /api/metadata -> dropdown options for the form
├── model/
│   ├── model.joblib      # trained sklearn Pipeline (preprocessing + RandomForest)
│   ├── metadata.json     # dropdown options / numeric ranges (generated)
│   └── metrics.json      # MAE / RMSE / R2 on the test set (generated)
├── public/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── train_model.py         # regenerates everything in model/
├── requirements.txt
├── vercel.json
└── data.csv                # training data (only needed to retrain)
```

## Deploy to Vercel

1. Push this folder to a GitHub repo.
2. In Vercel: **New Project → Import** the repo. Framework preset: "Other".
   No build command needed — `public/` is served as static files and
   everything in `api/` is auto-detected as Python serverless functions.
3. Deploy. Your form will be at `/`, the API at `/api/predict` and `/api/metadata`.

That's it — no environment variables or extra config required.

## Run locally

```bash
npm install -g vercel      # if you don't already have the CLI
vercel dev
```

This serves the static site and both Python functions on `localhost:3000`
exactly like production.

## Retraining the model

```bash
pip install -r requirements.txt
python train_model.py
```

This overwrites `model/model.joblib`, `model/metadata.json`, and
`model/metrics.json`. Commit the updated `model/` folder — the API loads the
`.joblib` file directly, it doesn't retrain on request.

## ⚠️ A known Vercel constraint

Vercel Python serverless functions have a **250 MB unzipped size limit**.
`scikit-learn` + `pandas` + `numpy` + `scipy` (scikit-learn's own dependency)
together can get close to that ceiling. This project has been kept as lean as
possible (one shared `model.joblib`, no unused dependencies), and it's within
the range where these ML-on-Vercel deployments commonly work — but if your
Vercel build fails with a size/bundle error, you have two easy fallbacks:

- **Split the deployment**: keep the frontend (`public/`) on Vercel, and host
  just `api/predict.py` + `model/` as a tiny Flask/FastAPI app on a Python-friendly
  host like Render, Railway, or Fly.io (all have generous free tiers). Then
  point `script.js`'s `fetch("/api/predict")` calls at that host's URL instead.
- **Convert the model to ONNX** with `skl2onnx` and swap `scikit-learn` for
  `onnxruntime` at inference time — `onnxruntime` alone is a fraction of the
  size of `scikit-learn` + `scipy`, which reliably resolves the limit. Ask if
  you'd like this version built out.

## API reference

**POST `/api/predict`**
```json
{
  "Age": 52, "Gender": "Female", "BMI": 24.8, "Chronic_Condition": "No",
  "Medicine_Name": "Montelukast", "Category": "Respiratory", "Type": "Tablet",
  "Generic_or_Branded": "Generic", "Manufacturer": "Sun Pharma",
  "Dosage_mg": 10, "Quantity": 40, "Duration_Days": 20, "Severity": "Mild",
  "Treatment_Type": "Specialized Treatment", "Region": "South",
  "Insurance_Pct": 40, "Discount_Pct": 11
}
```
→ `{"predicted_payable_cost": 1368.62}`

**GET `/api/metadata`** → dropdown options and numeric ranges used to build the form.
