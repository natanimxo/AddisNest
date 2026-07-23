# AddisNest — Rental Listing Fraud Detection

AddisNest is an AI-powered rental platform for Addis Ababa. This repo contains the fraud detection system — the ML model that flags suspicious listings before they reach users.

---

## What this does

Rental fraud is a real problem in Addis — landlords inflate prices, fake amenities like generators and elevators, or list properties in areas they're not actually in. This system takes a rental listing and returns a fraud probability score plus the likely fraud type.

The model was trained on a synthetic dataset of 20,000 listings generated to mirror the real Addis rental market, calibrated against actual scraped listings from Jiji, Engocha, and the Bet Afri Telegram channel (644 real listings, Sep 2025 – May 2026).

---

## Project structure

```
├── addis_rental_fraud_v5_final.py   # dataset generator
├── train.py                          # model training + evaluation
├── eda.py                            # exploratory data analysis
├── api.py                            # FastAPI prediction endpoint
├── model_output/
│   ├── fraud_detector.joblib         # trained model bundle
│   ├── confusion_matrices.png
│   ├── roc_pr_curves.png
│   ├── feature_importance.png
│   ├── score_distribution.png
│   ├── fraud_type_detection.png
│   └── shap_summary.png
└── eda_output/
    └── *.png                         # EDA plots
```

---

## How to run it

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Generate the dataset**
```bash
python addis_rental_fraud_v5_final.py
```
This creates `addis_rental_fraud_v5_final.csv` and three split files (train/val/test).

**3. Run EDA (optional but recommended)**
```bash
python eda.py
```
Saves 13 plots to `eda_output/`.

**4. Train the model**
```bash
python train.py
```
Trains a Logistic Regression baseline and XGBoost main model, prints full evaluation, and saves `model_output/fraud_detector.joblib`.

**5. Start the API**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## API usage

Send a POST request to `/predict` with a listing in JSON:

```json
{
  "price": 35000,
  "bedrooms": 2,
  "bathrooms": 2,
  "size_sqm": 95,
  "floor_number": 3,
  "building_type": "Apartment Building",
  "area": "Bole",
  "furnished": 1,
  "has_generator": 1,
  "has_parking": 1,
  "has_security": 0,
  "has_elevator": 1,
  "listing_age_days": 5,
  "views": 10,
  "contact_clicks": 2
}
```

Response:
```json
{
  "is_fraud": true,
  "fraud_probability": 0.823,
  "confidence": "Very High",
  "verdict": "This listing shows strong signs of fraud and should be reviewed.",
  "threshold_used": 0.65
}
```

Other endpoints:
- `GET /` — API info
- `GET /health` — health check
- `GET /info` — model metadata and supported areas

---

## Model results

| Model | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.16 | 0.09 | 0.63 | 0.76 |
| XGBoost | **0.33** | **0.46** | **0.25** | **0.74** |

Detection rate by fraud type:
- Engagement manipulation — 100%
- Amenity inflation — ~35%
- Moderate price inflation — ~40%
- Size deception — ~15%
- Location fraud — ~10%

Location fraud is the hardest — when a listing just claims a wrong neighbourhood, the price difference is subtle and there's no way to verify it without a live database of comparable listings. That's a known limitation and something we'd fix with real labelled data.

---

## Fraud types the model looks for

- **Moderate inflation** — price 2x–4x above what's typical for that area and building type
- **Amenity inflation** — claiming generator, elevator, or parking that doesn't exist, with inflated price
- **Size deception** — overstating property size to justify a higher price
- **Location fraud** — listing a budget-area property under a premium neighbourhood name
- **Engagement manipulation** — artificially inflated views and contact clicks on a new listing

---

## Dataset

The dataset is synthetic but calibrated from real sources:
- **Bet Afri Telegram channel** — 644 scraped listings (Sep 2025 – May 2026), primary reference for budget/mid pricing, studio prevalence, and amenity rates
- **Jiji.com.et** — price distribution and condo market data
- **Engocha.com** — mid-tier apartment pricing
- **The Africanvestor** — H1 2026 market reports for the premium segment

36 areas across Addis Ababa in three tiers (high-end, mid, budget). Studios are ~22% of listings, consistent with the real market.

---

## Tech stack

- Python 3.10+
- XGBoost, scikit-learn
- FastAPI + Uvicorn
- pandas, numpy, matplotlib, seaborn
- SHAP for model explainability
- joblib for model persistence

---

## Built by

**Natanim Mengistu Sisay** — Electrical and Computer Engineering graduate from Hawassa University, focused on applied AI and full-stack development. Built and trained the fraud detection model for AddisNest. Also the developer behind Wallet Buddy, a production Telegram bot for personal finance tracking, and has run a profitable Telegram advertising business doing $3,000+ in revenue. Works across Python, JavaScript/React, and SQL.

📩 natanimxo@gmail.com | 💬 @natanimxo
