import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE, '..')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'ambu_patient_data.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RANDOM_SEED = 42

FEATURE_COLS = [
    'bpm', 'mode', 'peep', 'lpm',
    'age', 'gender', 'condition', 'comorbidity',
    'pulse', 'bp_systolic', 'bp_diastolic', 'gcs_score', 'cvs_score',
    'spo2_before', 'spo2_5min', 'spo2_10min',
    'spo2_15min',  'spo2_20min', 'spo2_25min', 'spo2_30min',
    'spo2_delta', 'spo2_slope', 'spo2_at_15',
    'age_flag', 'gcs_flag', 'bp_pulse_ratio'
]


def _process_and_train(df: pd.DataFrame):
    """Core training logic — works on any DataFrame passed in."""

    # Fix NaN values
    df['comorbidity'] = df['comorbidity'].fillna('None')
    df = df.dropna(subset=['outcome', 'gender', 'condition', 'mode'])

    df_proc = df.copy()
    # Drop all non-training columns
    drop_cols = ['patient_id', '_id', 'added_at', 'photo_urls']
    for col in drop_cols:
        if col in df_proc.columns:
            df_proc.drop(columns=[col], inplace=True)

    # Drop any remaining columns with list/object values that aren't needed
    for col in df_proc.columns:
        if col not in FEATURE_COLS + ['outcome', 'gender', 'condition', 'comorbidity', 'mode',
                                       'age', 'bpm', 'peep', 'lpm', 'pulse', 'bp_systolic',
                                       'bp_diastolic', 'gcs_score', 'cvs_score', 'spo2_before',
                                       'spo2_5min', 'spo2_10min', 'spo2_15min', 'spo2_20min',
                                       'spo2_25min', 'spo2_30min']:
            df_proc.drop(columns=[col], inplace=True)

    # Encode categorical columns
    le_gender    = LabelEncoder()
    le_condition = LabelEncoder()
    le_comorbid  = LabelEncoder()
    le_mode      = LabelEncoder()

    le_gender.fit(["Male", "Female"])
    le_condition.fit(["Cardiac Arrest", "Drug Overdose", "Post-Surgical",
                      "Respiratory Failure", "Stroke/CNS", "Trauma"])
    le_comorbid.fit(["None", "Hypertension", "Diabetes", "COPD", "Heart Disease"])
    le_mode.fit(["Low", "Medium", "High", "Assist Control"])

    df_proc['gender']      = le_gender.transform(df_proc['gender'])
    df_proc['condition']   = le_condition.transform(df_proc['condition'])
    df_proc['comorbidity'] = le_comorbid.transform(df_proc['comorbidity'])
    df_proc['mode']        = le_mode.transform(df_proc['mode'])

    # Feature engineering
    df_proc['spo2_delta']     = df_proc['spo2_30min'] - df_proc['spo2_before']
    df_proc['spo2_slope']     = df_proc['spo2_delta'] / 30
    df_proc['spo2_at_15']     = df_proc['spo2_15min']
    df_proc['age_flag']       = (df_proc['age'] > 70).astype(int)
    df_proc['gcs_flag']       = (df_proc['gcs_score'] < 8).astype(int)
    df_proc['bp_pulse_ratio'] = df_proc['bp_systolic'] / (df_proc['pulse'] + 1)

    X = df_proc[FEATURE_COLS].copy()
    y = df_proc['outcome'].copy()

    # Scale
    scaler   = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    # SMOTE for class imbalance
    smote = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # Train Random Forest
    best_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    best_model.fit(X_res, y_res)

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'ambu_rf_model.pkl'))
    joblib.dump(scaler,     os.path.join(MODELS_DIR, 'ambu_scaler.pkl'))
    joblib.dump({
        'gender'      : le_gender,
        'condition'   : le_condition,
        'comorbidity' : le_comorbid,
        'mode'        : le_mode
    }, os.path.join(MODELS_DIR, 'ambu_encoders.pkl'))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, 'ambu_feature_cols.pkl'))

    print(f"✅ Retraining completed with {len(df)} patients. Models saved.")


def run_retraining():
    """Original function — loads CSV and retrains."""
    print("Loading data for retraining from", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    _process_and_train(df)


def run_retraining_with_data(df: pd.DataFrame):
    """New function — accepts DataFrame directly (CSV + MongoDB combined)."""
    print(f"Retraining with {len(df)} total patients (CSV + MongoDB)...")
    _process_and_train(df)


if __name__ == "__main__":
    run_retraining()