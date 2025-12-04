import os
import numpy as np
import pandas as pd
import pickle
from functools import lru_cache

# -------- Robust file loader (handles typos like symtoms_df.csv) ----------
def _read_csv_try(paths):
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    # If nothing found, return empty df with a warning-like column
    return pd.DataFrame()

# ------------------------ Cached loaders ----------------------------------
@lru_cache(maxsize=1)
def load_data():
    base = "datasets"
    description = _read_csv_try([f"{base}/description.csv"])
    precautions = _read_csv_try([f"{base}/precautions_df.csv"])
    medications = _read_csv_try([f"{base}/medications.csv", f"{base}/medication.csv"])
    diets       = _read_csv_try([f"{base}/diets.csv"])
    workout     = _read_csv_try([f"{base}/workout_df.csv"])
    training    = _read_csv_try([f"{base}/training.csv"])
    symptoms_df = _read_csv_try([f"{base}/symptoms_df.csv", f"{base}/symtoms_df.csv"])
    severity    = _read_csv_try([f"{base}/symptom-severity.csv"])
    return description, precautions, medications, diets, workout, training, symptoms_df, severity

@lru_cache(maxsize=1)
def load_model():
    with open("models/svc.pkl", "rb") as f:
        return pickle.load(f)

# ------------------------ Build vocab/labels ------------------------------
def build_symptom_dict(training: pd.DataFrame):
    if training.empty:
        return {}
    # assume last column is the target (prognosis/disease)
    symptom_cols = list(training.columns[:-1])
    return {sym: i for i, sym in enumerate(symptom_cols)}

DEFAULT_DISEASES_MAP = {
    15:'Fungal infection',4:'Allergy',16:'GERD',9:'Chronic cholestasis',14:'Drug Reaction',
    33:'Peptic ulcer diseae',1:'AIDS',12:'Diabetes ',17:'Gastroenteritis',6:'Bronchial Asthma',
    23:'Hypertension ',30:'Migraine',7:'Cervical spondylosis',32:'Paralysis (brain hemorrhage)',
    28:'Jaundice',29:'Malaria',8:'Chicken pox',11:'Dengue',37:'Typhoid',40:'hepatitis A',
    19:'Hepatitis B',20:'Hepatitis C',21:'Hepatitis D',22:'Hepatitis E',3:'Alcoholic hepatitis',
    36:'Tuberculosis',10:'Common Cold',34:'Pneumonia',13:'Dimorphic hemmorhoids(piles)',
    18:'Heart attack',39:'Varicose veins',26:'Hypothyroidism',24:'Hyperthyroidism',
    25:'Hypoglycemia',31:'Osteoarthristis',5:'Arthritis',0:'(vertigo) Paroymsal  Positional Vertigo',
    2:'Acne',38:'Urinary tract infection',35:'Psoriasis',27:'Impetigo'
}

def get_label_decoder(model):
    """
    Prefer model.classes_ when it contains readable labels.
    Fallback to DEFAULT_DISEASES_MAP (from your original code).
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        # Likely encoded ints → use default mapping
        return lambda y: DEFAULT_DISEASES_MAP.get(int(y), str(y))

    # If classes_ are strings (disease names), decode directly
    if all(isinstance(c, str) for c in classes):
        idx_map = {i: c for i, c in enumerate(classes)}
        return lambda y: idx_map.get(int(y), str(y))

    # classes_ are ints → use default mapping
    return lambda y: DEFAULT_DISEASES_MAP.get(int(y), str(y))

# ------------------------ Recommendation helper ---------------------------
def fetch_recommendations(disease_name: str,
                          description: pd.DataFrame,
                          precautions: pd.DataFrame,
                          medications: pd.DataFrame,
                          diets: pd.DataFrame,
                          workout: pd.DataFrame):
    desc = ""
    if not description.empty and "Disease" in description.columns:
        d = description[description["Disease"] == disease_name]["Description"]
        desc = " ".join(map(str, d.tolist()))

    pre_list = []
    if not precautions.empty and "Disease" in precautions.columns:
        cols = [c for c in ["Precaution_1","Precaution_2","Precaution_3","Precaution_4"] if c in precautions.columns]
        row = precautions[precautions["Disease"] == disease_name][cols]
        if not row.empty:
            pre_list = [x for x in row.iloc[0].tolist() if pd.notna(x)]

    meds = []
    if not medications.empty and "Disease" in medications.columns:
        meds = medications[medications["Disease"] == disease_name]["Medication"].dropna().tolist()

    diet = []
    if not diets.empty and "Disease" in diets.columns:
        diet = diets[diets["Disease"] == disease_name]["Diet"].dropna().tolist()

    wrkout = []
    if not workout.empty and "disease" in workout.columns:
        wrkout = workout[workout["disease"] == disease_name]["workout"].dropna().tolist()

    return desc, pre_list, meds, diet, wrkout

# ------------------------ Inference utilities -----------------------------
def _safe_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    return ex / s if s != 0 else np.ones_like(x)/len(x)

def topk_predictions(model, X: np.ndarray, k: int = 5):
    """Return (indices, scores probs 0..1). Works with SVC (probability or decision_function)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X.reshape(1, -1))[0]
        idx = np.argsort(probs)[::-1][:k]
        return idx, probs[idx]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X.reshape(1, -1))[0]
        probs = _safe_softmax(scores)  # convert to pseudo-probabilities
        idx = np.argsort(probs)[::-1][:k]
        return idx, probs[idx]
    else:
        # Fallback: only top-1 via predict
        y = model.predict(X.reshape(1, -1))[0]
        return np.array([int(y)]), np.array([1.0])