import sys
sys.path.append(".")

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

app = FastAPI(title="Hospital Fall Risk Detection API")

# Load model
model = XGBClassifier()
model.load_model("models/fall_risk_model.json")

CLASSES = {0: "high_risk", 1: "medium_risk", 2: "safe"}

class PoseFeatures(BaseModel):
    spine_angle: float
    lateral_lean: float
    left_knee_angle: float
    right_knee_angle: float
    hip_visibility: float
    knee_visibility: float

@app.get("/")
def root():
    return {"status": "ok", "message": "Fall Risk Detection API is running"}

@app.post("/predict")
def predict(features: PoseFeatures):
    X = np.array([[
        features.spine_angle,
        features.lateral_lean,
        features.left_knee_angle,
        features.right_knee_angle,
        features.hip_visibility,
        features.knee_visibility
    ]])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "prediction": CLASSES[int(pred)],
        "confidence": round(float(proba.max()), 4),
        "probabilities": {
            "high_risk": round(float(proba[0]), 4),
            "medium_risk": round(float(proba[1]), 4),
            "safe": round(float(proba[2]), 4)
        }
    }

