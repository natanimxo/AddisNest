"""
================================================
AddisNest — Rental Fraud Detection API
================================================
Install:
    pip install fastapi uvicorn joblib scikit-learn xgboost pandas numpy scipy

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Node.js backend calls:
    POST http://localhost:8000/predict
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
MODEL_PATH = "model_output/fraud_detector.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'.\n"
        "Run train.py first to generate it."
    )

bundle           = joblib.load(MODEL_PATH)
preprocessor     = bundle["preprocessor"]
model            = bundle["model"]
fraud_type_model = bundle.get("fraud_type_model")
label_encoder    = bundle.get("label_encoder")
THRESHOLD        = bundle["decision_threshold"]
FEATURES         = bundle["feature_names"]
NUMERIC_FEATURES     = bundle["numeric_features"]
CATEGORICAL_FEATURES = bundle["categorical_features"]

print(f"✓ Model loaded")
print(f"✓ Threshold: {THRESHOLD}")
print(f"✓ Features: {len(FEATURES)}")

# ── AREA TIER MAP ─────────────────────────────────────────────────────────────
AREA_CONFIGS = {
    "high_end": [
        "Bole", "Bole Michael", "Hayahulet", "Bole Medhane Alem",
        "Urael", "Atlas", "Kazanchis", "Sarbet", "Bisrate Gebriel",
        "Yeka", "Gerji", "Betel",
    ],
    "mid": [
        "CMC", "Megenagna", "Summit", "Ayat", "Wollo Sefer",
        "Lebu", "Jemo", "Gulele", "Lemi Kura", "Mekanisa",
        "Gurd Shola", "Lideta", "Piassa", "Gotera", "Kotebe",
        "Shola", "Lamberet", "Ayer Tena", "Adisu Gebeya", "Kera",
    ],
    "budget": [
        "Kolfe Keranio", "Akaki Kality", "Gofa", "Nifas Silk Lafto",
        "Shiro Meda", "Entoto", "Koye Feche", "Saris", "Bole Bulbula",
        "Addis Ketema", "Lafto", "Saris Abo", "Kirkos",
    ],
}
TIER_MAP     = {area: tier for tier, areas in AREA_CONFIGS.items() for area in areas}
TIER_ENCODED = {"budget": 1, "mid": 2, "high_end": 3}

PRICE_PER_SQM = {
    "Condominium":        {"high_end": 450, "mid": 340, "budget": 220},
    "Townhouse":          {"high_end": 650, "mid": 430, "budget": 270},
    "Apartment Building": {"high_end": 700, "mid": 400, "budget": 260},
    "Villa Compound":     {"high_end": 510, "mid": 270, "budget": 150},
}

GEN_RATES = {
    "Villa Compound":     {"high_end": 0.80, "mid": 0.45, "budget": 0.12},
    "Apartment Building": {"high_end": 0.65, "mid": 0.28, "budget": 0.05},
    "Condominium":        {"high_end": 0.04, "mid": 0.02, "budget": 0.01},
    "Townhouse":          {"high_end": 0.28, "mid": 0.10, "budget": 0.02},
}

ELEV_RATES = {
    "Villa Compound":     {"high_end": 0.004, "mid": 0.001, "budget": 0.000},
    "Apartment Building": {"high_end": 0.18,  "mid": 0.06,  "budget": 0.015},
    "Condominium":        {"high_end": 0.000, "mid": 0.000, "budget": 0.000},
    "Townhouse":          {"high_end": 0.000, "mid": 0.000, "budget": 0.000},
}

SEC_RATES = {
    "Villa Compound":     {"high_end": 0.92, "mid": 0.72, "budget": 0.42},
    "Apartment Building": {"high_end": 0.68, "mid": 0.22, "budget": 0.05},
    "Condominium":        {"high_end": 0.35, "mid": 0.18, "budget": 0.06},
    "Townhouse":          {"high_end": 0.65, "mid": 0.28, "budget": 0.08},
}

PARK_RATES = {
    "Villa Compound":     {"high_end": 0.96, "mid": 0.90, "budget": 0.78},
    "Apartment Building": {"high_end": 0.62, "mid": 0.35, "budget": 0.12},
    "Condominium":        {"high_end": 0.28, "mid": 0.15, "budget": 0.08},
    "Townhouse":          {"high_end": 0.82, "mid": 0.62, "budget": 0.45},
}

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AddisNest Fraud Detection API",
    description="Detects fraudulent rental listings in Addis Ababa.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class ListingRequest(BaseModel):
    price:            float = Field(..., gt=0)
    bedrooms:         int   = Field(..., ge=0)
    bathrooms:        int   = Field(..., ge=0)
    size_sqm:         float = Field(..., gt=0)
    floor_number:     int   = Field(..., ge=0)
    building_type:    str
    area:             str
    furnished:        int   = Field(..., ge=0, le=1)
    has_generator:    int   = Field(..., ge=0, le=1)
    has_parking:      int   = Field(..., ge=0, le=1)
    has_security:     int   = Field(..., ge=0, le=1)
    has_elevator:     int   = Field(..., ge=0, le=1)
    listing_age_days: int   = Field(default=1,  ge=1)
    views:            int   = Field(default=10, ge=0)
    contact_clicks:   int   = Field(default=2,  ge=0)

    class Config:
        extra = "ignore"   # silently ignores unknown fields from the backend

class PredictionResponse(BaseModel):
    is_fraud:          bool
    fraud_probability: float
    fraud_type:        str
    confidence:        str
    verdict:           str
    threshold_used:    float

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def build_features(listing: ListingRequest) -> pd.DataFrame:
    from scipy.special import expit

    d    = listing.model_dump()
    btype = d["building_type"]
    tier  = TIER_MAP.get(d["area"], "mid")

    def smooth_scale(v, midpoint=0.5, steepness=5):
        return float(expit(steepness * (v - midpoint)))

    def dev_score(obs, rate, mid=0.15, steep=8):
        rarity = (1 - rate) if obs == 1 else rate
        raw    = abs(obs - rate) * (0.3 + 0.7 * rarity)
        return float(np.clip(smooth_scale(raw, midpoint=mid, steepness=steep), 0, 1))

    price_per_sqm    = d["price"] / max(d["size_sqm"], 1)
    price_per_bedroom = d["price"] / (d["bedrooms"] + 1)
    size_per_bedroom  = d["size_sqm"] / (d["bedrooms"] + 1)
    engagement_rate   = d["contact_clicks"] / (d["views"] + 1)
    engagement_velocity = d["contact_clicks"] / (d["listing_age_days"] + 1)
    amenity_count     = d["has_generator"] + d["has_elevator"] + d["has_parking"] + d["has_security"]

    gen_rate  = GEN_RATES.get(btype,  GEN_RATES["Apartment Building"]).get(tier, 0.10)
    elev_rate = ELEV_RATES.get(btype, ELEV_RATES["Apartment Building"]).get(tier, 0.05)
    sec_rate  = SEC_RATES.get(btype,  SEC_RATES["Apartment Building"]).get(tier, 0.10)
    park_rate = PARK_RATES.get(btype, PARK_RATES["Apartment Building"]).get(tier, 0.10)

    prod_generator_anomaly = dev_score(d["has_generator"], gen_rate)
    prod_security_anomaly  = dev_score(d["has_security"],  sec_rate)
    prod_parking_anomaly   = dev_score(d["has_parking"],   park_rate)

    if btype in ("Condominium", "Townhouse"):
        prod_elevator_anomaly = 0.92 if d["has_elevator"] == 1 else 0.03
    else:
        prod_elevator_anomaly = dev_score(d["has_elevator"], elev_rate, mid=0.2, steep=7)

    expected_sqm = PRICE_PER_SQM.get(btype, PRICE_PER_SQM["Apartment Building"]).get(tier, 400)
    log_r        = np.log(np.clip(price_per_sqm / max(expected_sqm, 1), 0.1, 10.0))
    prod_price_anomaly = float(np.clip(abs(log_r) / 2.5, 0.0, 1.0))

    age   = d["listing_age_days"]
    exp_r, std_r = (0.12, 0.08) if age <= 3 else ((0.18, 0.10) if age <= 10 else (0.15, 0.12))
    prod_engagement_anomaly = float(np.clip(abs(engagement_rate - exp_r) / (std_r * 3), 0, 1))

    prod_studio_premium_anomaly = 0.0
    if d["bedrooms"] == 0:
        pc = amenity_count
        prod_studio_premium_anomaly = 0.75 if pc >= 3 else (0.45 if pc >= 2 else 0.10)

    price_position = d["price"] / max(d["size_sqm"] * expected_sqm, 1)

    # price_zscore — approximate (no group stats available at inference)
    price_zscore = float(np.clip((price_per_sqm - expected_sqm) / max(expected_sqm * 0.3, 1), -5, 5))

    row = {
        "price":            d["price"],
        "bedrooms":         d["bedrooms"],
        "bathrooms":        d["bathrooms"],
        "size_sqm":         d["size_sqm"],
        "floor_number":     d["floor_number"],
        "furnished":        d["furnished"],
        "has_generator":    d["has_generator"],
        "has_parking":      d["has_parking"],
        "has_security":     d["has_security"],
        "has_elevator":     d["has_elevator"],
        "listing_age_days": d["listing_age_days"],
        "views":            d["views"],
        "contact_clicks":   d["contact_clicks"],
        "price_per_sqm":    price_per_sqm,
        "price_per_bedroom": price_per_bedroom,
        "size_per_bedroom": size_per_bedroom,
        "area_tier_encoded": TIER_ENCODED.get(tier, 2),
        "engagement_rate":  engagement_rate,
        "engagement_velocity": engagement_velocity,
        "is_condo":         int(btype == "Condominium"),
        "is_villa":         int(btype == "Villa Compound"),
        "is_townhouse":     int(btype == "Townhouse"),
        "is_apartment":     int(btype == "Apartment Building"),
        "amenity_count":    amenity_count,
        "prod_generator_anomaly":      prod_generator_anomaly,
        "prod_elevator_anomaly":       prod_elevator_anomaly,
        "prod_security_anomaly":       prod_security_anomaly,
        "prod_parking_anomaly":        prod_parking_anomaly,
        "prod_price_anomaly":          prod_price_anomaly,
        "prod_engagement_anomaly":     prod_engagement_anomaly,
        "prod_studio_premium_anomaly": prod_studio_premium_anomaly,
        "price_position":  price_position,
        "price_zscore":    price_zscore,
        "building_type":   btype,
        "area":            d["area"],
    }

    return pd.DataFrame([row])

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "AddisNest Fraud Detection API",
        "version": "1.0.0",
        "status":  "running",
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "threshold": THRESHOLD}

@app.get("/info")
def info():
    return {
        "model":              "XGBoost",
        "features":           len(FEATURES),
        "decision_threshold": THRESHOLD,
        "metrics":            bundle.get("metrics", {}),
        "supported_building_types": [
            "Condominium", "Apartment Building", "Townhouse", "Villa Compound"
        ],
        "supported_areas": list(TIER_MAP.keys()),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(listing: ListingRequest):
    try:
        df = build_features(listing)

        # Align columns to what model expects
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in FEATURES if c in df.columns]]
        df = df.reindex(columns=FEATURES, fill_value=0)

        X           = preprocessor.transform(df)
        probability = float(model.predict_proba(X)[0][1])
        is_fraud    = probability >= THRESHOLD

        # Fraud type classification (only if flagged as fraud)
        detected_type = "none"
        if is_fraud and fraud_type_model is not None and label_encoder is not None:
            type_pred     = fraud_type_model.predict(X)
            detected_type = label_encoder.inverse_transform(type_pred)[0]

        # Confidence label
        if probability >= 0.80:   confidence = "Very High"
        elif probability >= 0.65: confidence = "High"
        elif probability >= 0.55: confidence = "Moderate"
        elif probability >= 0.40: confidence = "Low"
        else:                     confidence = "Very Low"

        verdict = (
            "This listing shows strong signs of fraud and should be reviewed."
            if is_fraud else
            "This listing appears legitimate based on available signals."
        )

        return PredictionResponse(
            is_fraud          = bool(is_fraud),
            fraud_probability = round(probability, 4),
            fraud_type        = detected_type,
            confidence        = confidence,
            verdict           = verdict,
            threshold_used    = THRESHOLD,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
