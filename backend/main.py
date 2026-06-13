from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import psycopg2
import os
import json
import joblib
from typing import List, Optional

app = FastAPI(title="Diabetes Risk API", version="1.0")

MODEL_DIR = os.getenv("MODEL_OUTPUT_DIR", "/app/models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

_model = None
_model_meta = None


def load_model():
    global _model, _model_meta
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            _model_meta = json.load(f)
    return _model, _model_meta


# LabelEncoder sorts classes alphabetically: "Female" -> 0, "Male" -> 1
GENDER_MAP = {"Female": 0, "Male": 1}

# Ordinal encodings for lifestyle features, must match ml/train_model.py
PHYSICAL_ACTIVITY_MAP = {"Low": 0, "Medium": 1, "High": 2}
SMOKING_MAP = {"Never": 0, "Former": 1, "Current": 2}
ALCOHOL_MAP = {"None": 0, "Moderate": 1, "High": 2}


def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "diabetes_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


class PatientRecord(BaseModel):
    id: int
    age: int
    gender: str
    bmi: float
    glucose: float
    blood_pressure: float
    hba1c: float
    insulin: float
    skin_thickness: float
    pregnancies: int
    physical_activity: str
    smoking_status: str
    alcohol_consumption: str
    family_history: int
    diabetes_risk: float
    label: int
    recorded_at: str


PATIENT_COLUMNS = (
    "id,age,gender,bmi,glucose,blood_pressure,hba1c,"
    "insulin,skin_thickness,pregnancies,"
    "physical_activity,smoking_status,alcohol_consumption,family_history,"
    "diabetes_risk,label,recorded_at"
)


@app.get("/patients", response_model=List[PatientRecord])
def get_patients(limit: int = Query(100, le=5000), label: Optional[int] = None):
    conn = get_db()
    cur = conn.cursor()
    if label is not None:
        cur.execute(
            f"SELECT {PATIENT_COLUMNS} "
            "FROM patient_records WHERE label=%s ORDER BY recorded_at DESC LIMIT %s",
            (label, limit)
        )
    else:
        cur.execute(
            f"SELECT {PATIENT_COLUMNS} "
            "FROM patient_records ORDER BY recorded_at DESC LIMIT %s",
            (limit,)
        )
    rows = cur.fetchall()
    conn.close()
    return [
        PatientRecord(
            id=r[0], age=r[1], gender=r[2], bmi=r[3],
            glucose=r[4], blood_pressure=r[5], hba1c=r[6],
            insulin=r[7], skin_thickness=r[8], pregnancies=r[9],
            physical_activity=r[10], smoking_status=r[11],
            alcohol_consumption=r[12], family_history=r[13],
            diabetes_risk=r[14], label=r[15], recorded_at=str(r[16])
        ) for r in rows
    ]


@app.get("/stats")
def get_stats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total,
            AVG(glucose) as avg_glucose,
            AVG(bmi) as avg_bmi,
            AVG(hba1c) as avg_hba1c,
            SUM(label) as diabetic_count,
            ROUND(AVG(label)*100, 2) as diabetic_pct
        FROM patient_records
    """)
    r = cur.fetchone()
    conn.close()
    return {
        "total_records": r[0],
        "avg_glucose": round(r[1], 2),
        "avg_bmi": round(r[2], 2),
        "avg_hba1c": round(r[3], 2),
        "diabetic_count": r[4],
        "diabetic_percentage": r[5],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


class PredictionInput(BaseModel):
    age: int
    gender: str
    bmi: float
    glucose: float
    blood_pressure: float
    hba1c: float
    insulin: float
    skin_thickness: float
    pregnancies: int = 0
    physical_activity: str = "Medium"
    smoking_status: str = "Never"
    alcohol_consumption: str = "None"
    family_history: int = 0


class PredictionOutput(BaseModel):
    diabetes_risk: float
    label: int
    model_name: str


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    model, meta = load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model topilmadi. Avval `docker compose --profile tools run --rm ml-trainer` ni ishga tushiring.",
        )

    if payload.gender not in GENDER_MAP:
        raise HTTPException(status_code=400, detail="gender must be 'Male' or 'Female'")
    if payload.physical_activity not in PHYSICAL_ACTIVITY_MAP:
        raise HTTPException(status_code=400, detail="physical_activity must be 'Low', 'Medium' or 'High'")
    if payload.smoking_status not in SMOKING_MAP:
        raise HTTPException(status_code=400, detail="smoking_status must be 'Never', 'Former' or 'Current'")
    if payload.alcohol_consumption not in ALCOHOL_MAP:
        raise HTTPException(status_code=400, detail="alcohol_consumption must be 'None', 'Moderate' or 'High'")

    row = {
        "age": payload.age,
        "bmi": payload.bmi,
        "glucose": payload.glucose,
        "blood_pressure": payload.blood_pressure,
        "hba1c": payload.hba1c,
        "insulin": payload.insulin,
        "skin_thickness": payload.skin_thickness,
        "pregnancies": payload.pregnancies,
        "gender_enc": GENDER_MAP[payload.gender],
        "physical_activity_enc": PHYSICAL_ACTIVITY_MAP[payload.physical_activity],
        "smoking_status_enc": SMOKING_MAP[payload.smoking_status],
        "alcohol_consumption_enc": ALCOHOL_MAP[payload.alcohol_consumption],
        "family_history": payload.family_history,
    }
    features = [[row[col] for col in meta["feature_cols"]]]

    risk = float(model.predict_proba(features)[0][1])
    label = int(model.predict(features)[0])

    return PredictionOutput(
        diabetes_risk=round(risk, 4),
        label=label,
        model_name=meta["model_name"],
    )