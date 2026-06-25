"""
AmbuPredict — FastAPI Backend
NeuO In-Spire Resuscitator Automation System
ML-powered ventilation outcome prediction API
MongoDB integration for persistent patient data storage
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import os
from fastapi.responses import JSONResponse
from datetime import datetime

# MongoDB
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

app = FastAPI(
    title       = "AmbuPredict API",
    description = "Predicts ventilation outcome for NeuO In-Spire patients",
    version     = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── MongoDB Connection ────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")
mongo_client = None
db = None
patients_col = None

def connect_mongo():
    global mongo_client, db, patients_col
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.server_info()  # test connection
        db = mongo_client["ambupredict"]
        patients_col = db["patients"]
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"⚠️ MongoDB connection failed: {e}")
        mongo_client = None
        patients_col = None

if MONGO_URI:
    connect_mongo()

# ── Load ML Models ────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, '..', 'models')

model        = joblib.load(os.path.join(MODELS_DIR, "ambu_rf_model.pkl"))
scaler       = joblib.load(os.path.join(MODELS_DIR, "ambu_scaler.pkl"))
encoders     = joblib.load(os.path.join(MODELS_DIR, "ambu_encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "ambu_feature_cols.pkl"))

VALID_GENDERS    = ["Male", "Female"]
VALID_MODES      = ["Low", "Medium", "High", "Assist Control"]
VALID_CONDITIONS = ["Cardiac Arrest", "Drug Overdose", "Post-Surgical",
                    "Respiratory Failure", "Stroke/CNS", "Trauma"]
VALID_COMORBIDS  = ["None", "Hypertension", "Diabetes", "COPD", "Heart Disease"]
VALID_BPMS       = [15, 20, 25, 30]


class PatientInput(BaseModel):
    bpm: int          = Field(..., example=25)
    mode: str         = Field(..., example="High")
    peep: int         = Field(..., ge=0, le=20, example=10)
    lpm: float        = Field(..., ge=0, le=20, example=8.0)
    age: int          = Field(..., ge=1, le=120, example=45)
    gender: str       = Field(..., example="Male")
    condition: str    = Field(..., example="Respiratory Failure")
    comorbidity: str  = Field(..., example="None")
    pulse: int        = Field(..., ge=20, le=250, example=95)
    bp_systolic: int  = Field(..., ge=60, le=250, example=120)
    bp_diastolic: int = Field(..., ge=30, le=150, example=80)
    gcs_score: int    = Field(..., ge=3, le=15, example=12)
    cvs_score: int    = Field(..., ge=1, le=5, example=3)
    spo2_before: int  = Field(..., ge=30, le=100, example=72)
    spo2_5min: int    = Field(..., ge=30, le=100, example=76)
    spo2_10min: int   = Field(..., ge=30, le=100, example=80)
    spo2_15min: int   = Field(..., ge=30, le=100, example=84)
    spo2_20min: int   = Field(..., ge=30, le=100, example=87)
    spo2_25min: int   = Field(..., ge=30, le=100, example=90)
    spo2_30min: int   = Field(..., ge=30, le=100, example=93)

    @validator('bpm')
    def bpm_valid(cls, v):
        if v not in VALID_BPMS: raise ValueError(f"BPM must be one of {VALID_BPMS}")
        return v
    @validator('mode')
    def mode_valid(cls, v):
        if v not in VALID_MODES: raise ValueError(f"Mode must be one of {VALID_MODES}")
        return v
    @validator('gender')
    def gender_valid(cls, v):
        if v not in VALID_GENDERS: raise ValueError(f"Gender must be one of {VALID_GENDERS}")
        return v
    @validator('condition')
    def condition_valid(cls, v):
        # Allow standard conditions OR any custom condition
        standard = ["Cardiac Arrest", "Drug Overdose", "Post-Surgical",
                    "Respiratory Failure", "Stroke/CNS", "Trauma"]
        if v not in standard:
            # Store as-is for custom conditions
            return v
        return v
    @validator('comorbidity')
    def comorbidity_valid(cls, v):
        if v not in VALID_COMORBIDS: raise ValueError(f"Comorbidity must be one of {VALID_COMORBIDS}")
        return v


class PatientAddInput(PatientInput):
    outcome: int = Field(..., description="Actual outcome: 1 for Positive, 0 for Negative")
    photo_urls: list = Field(default=[], description="Cloudinary photo URLs")


class PredictionResponse(BaseModel):
    probability: float
    probability_pct: float
    outcome: str
    risk_level: str
    clinical_advice: str
    top_factors: list


def build_features(p: PatientInput) -> pd.DataFrame:
    ge = encoders['gender'].transform([p.gender])[0]
    ce = encoders['condition'].transform([p.condition])[0]
    co = encoders['comorbidity'].transform([p.comorbidity])[0]
    me = encoders['mode'].transform([p.mode])[0]
    delta = p.spo2_30min - p.spo2_before
    raw = pd.DataFrame([[
        p.bpm, me, p.peep, p.lpm, p.age, ge, ce, co,
        p.pulse, p.bp_systolic, p.bp_diastolic, p.gcs_score, p.cvs_score,
        p.spo2_before, p.spo2_5min, p.spo2_10min, p.spo2_15min,
        p.spo2_20min, p.spo2_25min, p.spo2_30min,
        delta, delta/30, p.spo2_15min,
        int(p.age > 70), int(p.gcs_score < 8), p.bp_systolic/(p.pulse+1)
    ]], columns=feature_cols)
    return pd.DataFrame(scaler.transform(raw), columns=feature_cols)


@app.get("/", tags=["Health"])
def root():
    mongo_status = "connected" if patients_col is not None else "disconnected"
    return {
        "status": "ok",
        "service": "AmbuPredict API v2.0",
        "mongodb": mongo_status
    }


@app.get("/meta", tags=["Meta"])
def get_valid_values():
    return {
        "bpm_options": VALID_BPMS, "mode_options": VALID_MODES,
        "gender_options": VALID_GENDERS, "condition_options": VALID_CONDITIONS,
        "comorbidity_options": VALID_COMORBIDS, "peep_options": [0,5,10,15,20],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    """Predict ventilation outcome for a single patient."""
    try:
        features = build_features(patient)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature encoding error: {str(e)}")
    try:
        prob  = float(model.predict_proba(features)[0][1])
        label = "Positive" if prob >= 0.5 else "Negative"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

    if prob >= 0.70:
        risk   = "HIGH"
        advice = "High probability of effective ventilation. Proceed with current settings. Monitor SpO₂ every 5 min."
    elif prob >= 0.45:
        risk   = "MODERATE"
        advice = "Moderate probability. Monitor closely. If SpO₂ not improving by 15 min, adjust BPM or Mode upward."
    else:
        risk   = "LOW"
        advice = "Low probability of success. Consider increasing BPM/Mode, adjusting PEEP, or escalating to full ventilator."

    top_idx = np.argsort(model.feature_importances_)[::-1][:5]
    top_factors = [{"feature": feature_cols[i], "importance": round(float(model.feature_importances_[i]), 4)} for i in top_idx]

    return PredictionResponse(
        probability=round(prob,4), probability_pct=round(prob*100,1),
        outcome=label, risk_level=risk, clinical_advice=advice, top_factors=top_factors,
    )


@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(patients: list[PatientInput]):
    """Predict outcomes for up to 50 patients at once."""
    if len(patients) > 50:
        raise HTTPException(status_code=400, detail="Max 50 patients per batch.")
    results = []
    for i, p in enumerate(patients):
        try:
            prob = float(model.predict_proba(build_features(p))[0][1])
            results.append({"index": i+1, "probability_pct": round(prob*100,1),
                "outcome": "Positive" if prob>=0.5 else "Negative",
                "risk_level": "HIGH" if prob>=0.70 else ("MODERATE" if prob>=0.45 else "LOW")})
        except Exception as e:
            results.append({"index": i+1, "error": str(e)})
    return {"total": len(patients), "results": results}


@app.post("/add_patient", tags=["Data"])
def add_patient(patient: PatientAddInput):
    """Add a new patient to MongoDB and trigger model retraining."""

    new_row = {
        'age': patient.age, 'gender': patient.gender,
        'condition': patient.condition, 'comorbidity': patient.comorbidity,
        'bpm': patient.bpm, 'mode': patient.mode,
        'peep': patient.peep, 'lpm': patient.lpm,
        'pulse': patient.pulse, 'bp_systolic': patient.bp_systolic,
        'bp_diastolic': patient.bp_diastolic,
        'gcs_score': patient.gcs_score, 'cvs_score': patient.cvs_score,
        'spo2_before': patient.spo2_before,
        'spo2_5min': patient.spo2_5min, 'spo2_10min': patient.spo2_10min,
        'spo2_15min': patient.spo2_15min, 'spo2_20min': patient.spo2_20min,
        'spo2_25min': patient.spo2_25min, 'spo2_30min': patient.spo2_30min,
        'outcome': patient.outcome,
        'photo_urls': patient.photo_urls,
        'added_at': datetime.utcnow().isoformat()
    }

    # ── Save to MongoDB ───────────────────────────────────────────────────────
    mongo_saved = False
    if patients_col is not None:
        try:
            patients_col.insert_one(new_row)
            mongo_saved = True
            print(f"✅ Patient saved to MongoDB")
        except Exception as e:
            print(f"⚠️ MongoDB save failed: {e}")

    # ── Also save to CSV as backup ────────────────────────────────────────────
    data_path = os.path.join(BASE, '..', 'data', 'ambu_patient_data.csv')
    try:
        df = pd.read_csv(data_path)
        new_row_csv = {k: v for k, v in new_row.items() if k != 'added_at'}
        new_row_csv['patient_id'] = f"P{str(len(df)+1).zfill(4)}"
        df = pd.concat([df, pd.DataFrame([new_row_csv])], ignore_index=True)
        df.to_csv(data_path, index=False)
    except Exception as e:
        print(f"⚠️ CSV save failed: {e}")

    # ── Retrain using MongoDB data + original CSV ─────────────────────────────
    try:
        REQUIRED = ['age','gender','condition','comorbidity','bpm','mode','peep','lpm',
                    'pulse','bp_systolic','bp_diastolic','gcs_score','cvs_score',
                    'spo2_before','spo2_5min','spo2_10min','spo2_15min',
                    'spo2_20min','spo2_25min','spo2_30min','outcome']

        # Load original CSV data - keep only required columns
        df_csv = pd.read_csv(data_path)
        df_csv = df_csv[[c for c in REQUIRED if c in df_csv.columns]]

        # Load MongoDB data if connected - exclude non-training fields
        if patients_col is not None:
            mongo_records = list(patients_col.find(
                {}, {'_id': 0, 'added_at': 0, 'photo_urls': 0}
            ))
            if mongo_records:
                df_mongo = pd.DataFrame(mongo_records)
                df_mongo = df_mongo[[c for c in REQUIRED if c in df_mongo.columns]]
                df_all = pd.concat([df_csv, df_mongo], ignore_index=True)
            else:
                df_all = df_csv
        else:
            df_all = df_csv

        # Final cleanup - ensure no list/object columns remain
        df_all = df_all[[c for c in REQUIRED if c in df_all.columns]]
        df_all = df_all.dropna(subset=['outcome'])
        df_all = df_all.reset_index(drop=True)

        from .retrain import run_retraining_with_data
        run_retraining_with_data(df_all)

        # Reload models into memory
        global model, scaler, encoders, feature_cols
        model        = joblib.load(os.path.join(MODELS_DIR, "ambu_rf_model.pkl"))
        scaler       = joblib.load(os.path.join(MODELS_DIR, "ambu_scaler.pkl"))
        encoders     = joblib.load(os.path.join(MODELS_DIR, "ambu_encoders.pkl"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "ambu_feature_cols.pkl"))

        total_patients = len(df_all)
        return {
            "status": "success",
            "message": f"Patient saved {'to MongoDB' if mongo_saved else 'locally'} and model retrained with {total_patients} total patients.",
            "mongodb_saved": mongo_saved,
            "total_patients": total_patients
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/patients/count", tags=["Data"])
def get_patient_count():
    """Get total number of patients in MongoDB."""
    if patients_col is None:
        return {"mongodb": "disconnected", "count": 0}
    try:
        count = patients_col.count_documents({})
        return {"mongodb": "connected", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/all", tags=["Data"])
def get_all_patients():
    """Get all patients stored in MongoDB."""
    if patients_col is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected.")
    try:
        records = list(patients_col.find({}, {'_id': 0}))
        return {"total": len(records), "patients": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Admin System ──────────────────────────────────────────────────────────────
import hashlib
import secrets
from bson import ObjectId

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_token(token: str) -> bool:
    if db is None:
        return False
    return db["admin_tokens"].find_one({"token": token}) is not None

class LoginInput(BaseModel):
    username: str
    password: str

class AddUserInput(BaseModel):
    username: str
    password: str
    role: str = "admin"


@app.post("/admin/login", tags=["Admin"])
def admin_login(data: LoginInput):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    # Check if any admin exists, if not create default
    if db["admins"].count_documents({}) == 0:
        db["admins"].insert_one({
            "username": "ambuadmin",
            "password": hash_password("NeuO@2024#Inspire"),
            "role": "superadmin"
        })
    
    user = db["admins"].find_one({"username": data.username})
    if not user or user["password"] != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = secrets.token_hex(32)
    db["admin_tokens"].insert_one({"token": token, "username": data.username})
    return {"token": token, "username": data.username, "role": user.get("role", "admin")}


@app.get("/admin/patients", tags=["Admin"])
def admin_get_patients(authorization: str = None):
    from fastapi import Header
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    try:
        patients = list(db["patients"].find({}))
        for p in patients:
            p["_id"] = str(p["_id"])
        return {"total": len(patients), "patients": patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/patient/{patient_id}", tags=["Admin"])
def admin_delete_patient(patient_id: str):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    try:
        result = db["patients"].delete_one({"_id": ObjectId(patient_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Patient not found")
        return {"status": "success", "message": "Patient deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/retrain", tags=["Admin"])
def admin_retrain():
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    try:
        import pandas as pd
        mongo_records = list(db["patients"].find({}, {'_id': 0, 'added_at': 0, 'photo_urls': 0}))
        if not mongo_records:
            raise HTTPException(status_code=400, detail="No patient data to retrain with")
        
        df_mongo = pd.DataFrame(mongo_records)
        data_path = os.path.join(BASE, '..', 'data', 'ambu_patient_data.csv')
        df_csv = pd.read_csv(data_path)
        
        REQUIRED = ['age','gender','condition','comorbidity','bpm','mode','peep','lpm',
                    'pulse','bp_systolic','bp_diastolic','gcs_score','cvs_score',
                    'spo2_before','spo2_5min','spo2_10min','spo2_15min',
                    'spo2_20min','spo2_25min','spo2_30min','outcome']
        df_all = pd.concat([df_csv, df_mongo], ignore_index=True)
        df_all = df_all[[c for c in REQUIRED if c in df_all.columns]]
        
        from .retrain import run_retraining_with_data
        run_retraining_with_data(df_all)
        
        global model, scaler, encoders, feature_cols
        model        = joblib.load(os.path.join(MODELS_DIR, "ambu_rf_model.pkl"))
        scaler       = joblib.load(os.path.join(MODELS_DIR, "ambu_scaler.pkl"))
        encoders     = joblib.load(os.path.join(MODELS_DIR, "ambu_encoders.pkl"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "ambu_feature_cols.pkl"))
        
        return {"status": "success", "message": f"Model retrained with {len(df_all)} patients"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/add_user", tags=["Admin"])
def admin_add_user(data: AddUserInput):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    if db["admins"].find_one({"username": data.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    db["admins"].insert_one({
        "username": data.username,
        "password": hash_password(data.password),
        "role": data.role
    })
    return {"status": "success", "message": f"User '{data.username}' added successfully"}