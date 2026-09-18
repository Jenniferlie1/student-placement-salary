# Student Placement & Salary Prediction System

This repository contains an end-to-end machine learning project designed to predict a student's campus placement status and estimated salary package based on their academic and skill profile. The system is deployed as a full-stack application with a FastAPI backend serving the models and a Streamlit frontend for user interaction.

## Project Overview

To help university career centers guide students more effectively, this project predicts two things from a single student profile: whether the student is likely to be **placed**, and if so, their **estimated salary (LPA)**. This lets career counselors identify students who may need additional support (extra certifications, projects, interview prep) before placement season, and gives students a data-driven benchmark for where they stand.

## Application Architecture

The project follows a decoupled, full-stack architecture:

1. **Machine Learning Models** — Two models are trained from the same student profile data: an `SVC` classifier for placement status and a `RandomForestRegressor` for salary. Both share a `ColumnTransformer` preprocessing step (`StandardScaler` for numeric features, `OneHotEncoder` for categorical features) wrapped in their own scikit-learn `Pipeline`, tracked with MLflow, and serialized with `joblib` for deployment.
2. **Backend API (FastAPI)** — A FastAPI service loads both serialized pipelines and exposes a single `/predict` endpoint. It accepts a student's profile (validated via a Pydantic schema), runs it through both models, and returns the predicted placement status and salary.
3. **Frontend Application (Streamlit)** — An interactive form lets users enter a student's profile, sends it to the FastAPI backend, and displays the placement result, predicted salary, a skill comparison chart, and a running history of past predictions.

## Project Structure

```
.
├── data_ingestion.py         # ingest_data(): loads and validates B.csv
├── preprocessing.py          # preprocess(): feature/target split, train/test split (classification + regression)
├── train.py                    # train(): SVC + RandomForestRegressor pipelines, MLflow logging, artifact export
├── evaluation.py              # evaluate(): loads models from MLflow, computes accuracy + R2
├── pipeline.py                 # orchestrates ingestion -> preprocessing -> train -> evaluate -> approval decision
├── student_fastAPI.py         # FastAPI backend, serves /predict
├── student_streamlit.py       # Streamlit frontend, calls the FastAPI backend
├── app.py                      # standalone Streamlit variant (loads models directly, no API)
├── B.csv                        # raw input data
├── student_placement___salary_notebook.ipynb   # exploratory notebook
├── ingested/                    # generated: cleaned data (B.csv)
├── artifacts/                   # generated: classification_model.pkl, regression_model.pkl
└── mlruns/                      # generated: MLflow tracking store
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python pipeline.py
```

Steps:
1. **Ingestion** — validates `B.csv` isn't empty, copies it to `ingested/B.csv`.
2. **Preprocessing** — drops `placement_status`, `salary_package_lpa`, `student_id` from features; splits into classification target (`placement_status`) and regression target (`salary_package_lpa`) using the same stratified train/test split.
3. **Training** — fits an `SVC` pipeline for placement and a `RandomForestRegressor` pipeline for salary, logs both to MLflow under the `Student-Pipeline` experiment, saves them to `artifacts/classification_model.pkl` and `artifacts/regression_model.pkl`.
4. **Evaluation** — reloads both models from the MLflow run, reports classification accuracy and regression R².
5. **Deployment decision** — approves if accuracy ≥ `0.8` and R² ≥ `0.5`.

View MLflow runs:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

## Running the Application

Needs `artifacts/classification_model.pkl` and `artifacts/regression_model.pkl` to already exist (run the pipeline first). Start both services in separate terminals:

```bash
# Terminal 1 — backend
uvicorn student_fastAPI:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
streamlit run student_streamlit.py
```

The Streamlit app calls the API at `http://127.0.0.1:8000/predict` — update `API_URL` in `student_streamlit.py` if the backend runs on a different host (e.g. when deploying the two services separately).

## How Prediction Works

The sidebar form collects a student's academic scores (SSC/HSC/degree %, CGPA), entrance exam and skill scores, activity counts (internships, projects, certifications, backlogs), attendance, and extracurricular involvement.

On submit, `student_streamlit.py` sends the form data as JSON to the FastAPI `/predict` endpoint. `student_fastAPI.py` validates it against the `StudentInput` schema, runs it through both pipelines, and returns `placement_status` (0/1) and `salary_package_lpa`. The frontend then shows a Placed/Not Placed badge, the predicted salary, a technical-vs-soft-skill bar chart, and appends the result to the session's prediction history.

## Known Gotchas

- **Hardcoded API URL**: `student_streamlit.py` points at `http://127.0.0.1:8000/predict`. This breaks if the backend isn't running locally on port 8000 — update it (or use an env var) before deploying frontend and backend on separate hosts.
- **Version mismatch**: unpickling `classification_model.pkl` / `regression_model.pkl` requires the same `scikit-learn` version used at training time. Pin it in `requirements.txt`.
- **`app.py` is a separate, non-API variant** — it loads both `.pkl` files directly inside Streamlit instead of calling FastAPI. Keep it in sync manually if you update the input schema, or remove it if the FastAPI version is the one being shipped.
- **`evaluate("FILL_RUN_ID")`** in `evaluation.py`'s `__main__` block is a placeholder — running the file directly needs a real MLflow run ID; `pipeline.py` passes it automatically from `train()`.
