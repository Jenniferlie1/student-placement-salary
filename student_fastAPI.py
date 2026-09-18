from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Student Placement API")

clf = joblib.load("artifacts/classification_model.pkl")
reg = joblib.load("artifacts/regression_model.pkl")


class StudentInput(BaseModel):
    gender: str
    ssc_percentage: int
    hsc_percentage: int
    degree_percentage: int
    cgpa: float
    entrance_exam_score: int
    technical_skill_score: int
    soft_skill_score: int
    internship_count: int
    live_projects: int
    work_experience_months: int
    certifications: int
    attendance_percentage: int
    backlogs: int
    extracurricular_activities: str


@app.post("/predict")
def predict(data: StudentInput):

    df = pd.DataFrame([data.dict()])

    pred_class = clf.predict(df)[0]
    pred_salary = reg.predict(df)[0]

    return {
        "placement_status": int(pred_class),
        "salary_package_lpa": float(round(pred_salary, 2))
    }