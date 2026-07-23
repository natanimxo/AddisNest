"""
===================================================
MODEL TRAINING — Addis Ababa Rental Fraud Detection
===================================================
Runs in order:
  1. Load train/val/test splits
  2. Preprocessing pipeline
  3. Baseline — Logistic Regression
  4. Main model — XGBoost (tuned on val set)
  5. Final evaluation on test set (true_is_fraud)
  6. Feature importance + SHAP

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve,
                                       precision_recall_curve, average_precision_score,
                                       f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

os.makedirs("model_output", exist_ok=True)

# ── COLOUR CONFIG ─────────────────────────────────────────────────────────────
C_LEGIT = "#2196F3"
C_FRAUD = "#F44336"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

def save(name):
    path = os.path.join("model_output", name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FRAUD DETECTION — MODEL TRAINING")
print("="*60)

TRAIN = "addis_rental_fraud_v5_train.csv"
VAL   = "addis_rental_fraud_v5_val.csv"
TEST  = "addis_rental_fraud_v5_test.csv"

for f in [TRAIN, VAL, TEST]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"'{f}' not found.\n"
            "Run  python addis_rental_fraud_v5_final.py  first."
        )

train_df = pd.read_csv(TRAIN)
val_df   = pd.read_csv(VAL)
test_df  = pd.read_csv(TEST)

# Drop rows where label is missing
train_df = train_df.dropna(subset=['is_fraud'])
val_df   = val_df.dropna(subset=['is_fraud'])
test_df  = test_df.dropna(subset=['true_is_fraud'])

# Fill NaN in numeric columns with 0 (safe default for all feature types)
train_df = train_df.fillna(0)
val_df   = val_df.fillna(0)
test_df  = test_df.fillna(0)

print(f"\nTrain: {len(train_df):,} rows")
print(f"Val:   {len(val_df):,} rows")
print(f"Test:  {len(test_df):,} rows")
print(f"Train fraud rate: {train_df['is_fraud'].mean()*100:.2f}%  (noisy label)")
print(f"Test  fraud rate: {test_df['true_is_fraud'].mean()*100:.2f}%  (ground truth)")

# ── DEFINE FEATURES ───────────────────────────────────────────────────────────
# Safe training features only (no oracle scores, no true labels)
NUMERIC_FEATURES = [
    'price', 'bedrooms', 'bathrooms', 'size_sqm', 'floor_number',
    'furnished', 'has_generator', 'has_parking', 'has_security', 'has_elevator',
    'listing_age_days', 'views', 'contact_clicks',
    'price_per_sqm', 'price_per_bedroom', 'size_per_bedroom',
    'area_tier_encoded', 'engagement_rate', 'engagement_velocity',
    'is_condo', 'is_villa', 'is_townhouse', 'is_apartment', 'amenity_count',
    'prod_generator_anomaly', 'prod_elevator_anomaly',
    'prod_price_anomaly', 'prod_engagement_anomaly',
    'prod_studio_premium_anomaly', 'price_position',
]

CATEGORICAL_FEATURES = ['building_type', 'area']

# Keep only columns that exist in both train and test
NUMERIC_FEATURES     = [c for c in NUMERIC_FEATURES     if c in train_df.columns]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print(f"\nFeatures: {len(NUMERIC_FEATURES)} numeric + "
      f"{len(CATEGORICAL_FEATURES)} categorical = {len(ALL_FEATURES)} total")

# ── PREPARE SPLITS ────────────────────────────────────────────────────────────
X_train = train_df[ALL_FEATURES]
y_train = train_df['is_fraud']          # noisy label for training

X_val   = val_df[ALL_FEATURES]
y_val   = val_df['is_fraud']            # noisy label for val tuning

X_test  = test_df[ALL_FEATURES]
y_test  = test_df['true_is_fraud']      # GROUND TRUTH for final evaluation

# ── PREPROCESSING PIPELINE ────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(),                       NUMERIC_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
], remainder='drop')

# Convert labels to int to avoid float mismatch in Colab/different environments
y_train = y_train.astype(int)
y_val   = y_val.astype(int)
y_test  = y_test.astype(int)

# Class weights to handle imbalance (~5% fraud)
classes   = np.unique(y_train)
cw        = compute_class_weight('balanced', classes=classes, y=y_train)
cw_dict   = {int(c): w for c, w in zip(classes, cw)}
print(f"\nClass weights: legit={cw_dict[0]:.2f}  fraud={cw_dict[1]:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. BASELINE — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  BASELINE: Logistic Regression")
print("─"*60)

baseline_pipe = Pipeline([
    ('pre',   preprocessor),
    ('model', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

baseline_pipe.fit(X_train, y_train)

# Evaluate on val set (noisy label — for comparison only)
y_val_pred_base   = baseline_pipe.predict(X_val)
y_val_proba_base  = baseline_pipe.predict_proba(X_val)[:, 1]

# Evaluate on test set (ground truth)
y_test_pred_base  = baseline_pipe.predict(X_test)
y_test_proba_base = baseline_pipe.predict_proba(X_test)[:, 1]

base_f1        = f1_score(y_test, y_test_pred_base)
base_precision = precision_score(y_test, y_test_pred_base)
base_recall    = recall_score(y_test, y_test_pred_base)
base_auc       = roc_auc_score(y_test, y_test_proba_base)

print(f"\n  [TEST SET — ground truth labels]")
print(f"  F1 Score:   {base_f1:.4f}")
print(f"  Precision:  {base_precision:.4f}")
print(f"  Recall:     {base_recall:.4f}")
print(f"  AUC-ROC:    {base_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_test_pred_base,
                             target_names=['Legitimate','Fraud'],
                             digits=4))

# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN MODEL — XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  MAIN MODEL: XGBoost")
print("─"*60)

# scale_pos_weight handles class imbalance natively in XGBoost
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
spw       = neg_count / pos_count
print(f"\n  scale_pos_weight = {spw:.2f}  (neg/pos ratio)")

# ── HYPERPARAMETER SEARCH ON VAL SET ─────────────────────────────────────────
print("\n  Tuning hyperparameters on validation set...")

# Preprocess once for faster search
X_train_pre = preprocessor.fit_transform(X_train)
X_val_pre   = preprocessor.transform(X_val)
X_test_pre  = preprocessor.transform(X_test)

param_grid = [
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,  "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,  "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.10,  "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03,  "subsample": 0.8, "colsample_bytree": 0.7},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,  "subsample": 0.7, "colsample_bytree": 0.8},
]

best_f1     = -1
best_params = None
best_model  = None

for i, params in enumerate(param_grid):
    m = xgb.XGBClassifier(
        **params,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    m.fit(X_train_pre, y_train,
          eval_set=[(X_val_pre, y_val)],
          verbose=False)

    y_val_pred = m.predict(X_val_pre)
    f1 = f1_score(y_val, y_val_pred)

    print(f"  [{i+1}/{len(param_grid)}] depth={params['max_depth']}  "
          f"lr={params['learning_rate']}  trees={params['n_estimators']}  "
          f"→ val F1={f1:.4f}")

    if f1 > best_f1:
        best_f1     = f1
        best_params = params
        best_model  = m

print(f"\n  ✓ Best val F1: {best_f1:.4f}")
print(f"  Best params:  {best_params}")

# ── FINAL EVALUATION ON TEST SET ─────────────────────────────────────────────
print(f"\n  [TEST SET — ground truth labels]")

y_test_pred_xgb  = best_model.predict(X_test_pre)
y_test_proba_xgb = best_model.predict_proba(X_test_pre)[:, 1]

xgb_f1        = f1_score(y_test, y_test_pred_xgb)
xgb_precision = precision_score(y_test, y_test_pred_xgb)
xgb_recall    = recall_score(y_test, y_test_pred_xgb)
xgb_auc       = roc_auc_score(y_test, y_test_proba_xgb)
xgb_ap        = average_precision_score(y_test, y_test_proba_xgb)

print(f"  F1 Score:            {xgb_f1:.4f}")
print(f"  Precision:           {xgb_precision:.4f}")
print(f"  Recall:              {xgb_recall:.4f}")
print(f"  AUC-ROC:             {xgb_auc:.4f}")
print(f"  Avg Precision (AP):  {xgb_ap:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_test_pred_xgb,
                             target_names=['Legitimate','Fraud'],
                             digits=4))

# ── COMPARISON TABLE ──────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("  MODEL COMPARISON (test set, ground truth)")
print("─"*60)
print(f"  {'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC-ROC':>9}")
print(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*8} {'─'*9}")
print(f"  {'Logistic Regression':<25} {base_f1:>8.4f} {base_precision:>10.4f} "
      f"{base_recall:>8.4f} {base_auc:>9.4f}")
print(f"  {'XGBoost':<25} {xgb_f1:>8.4f} {xgb_precision:>10.4f} "
      f"{xgb_recall:>8.4f} {xgb_auc:>9.4f}")

improvement = (xgb_f1 - base_f1) / base_f1 * 100
print(f"\n  XGBoost F1 improvement over baseline: +{improvement:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  GENERATING PLOTS")
print("─"*60)

# ── 4a. Confusion matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, pred, title in zip(axes,
    [y_test_pred_base, y_test_pred_xgb],
    ['Logistic Regression', 'XGBoost (Best)']):
    cm = confusion_matrix(y_test, pred)
    sns_colors = [[C_LEGIT if i==j else C_FRAUD
                   for j in range(2)] for i in range(2)]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Legit','Pred Fraud'])
    ax.set_yticklabels(['True Legit','True Fraud'])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.suptitle("Confusion Matrices — Test Set (Ground Truth)", fontsize=13)
plt.tight_layout()
save("confusion_matrices.png")

# ── 4b. ROC curves ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC
ax = axes[0]
for proba, label, color, auc in [
    (y_test_proba_base, 'Logistic Regression', '#FF9800', base_auc),
    (y_test_proba_xgb,  'XGBoost',             C_FRAUD,  xgb_auc),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label}  (AUC={auc:.3f})")
ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(fontsize=9)

# Precision-Recall
ax = axes[1]
for proba, label, color, ap in [
    (y_test_proba_base, 'Logistic Regression', '#FF9800', average_precision_score(y_test, y_test_proba_base)),
    (y_test_proba_xgb,  'XGBoost',             C_FRAUD,  xgb_ap),
]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, color=color, lw=2, label=f"{label}  (AP={ap:.3f})")
ax.axhline(y=y_test.mean(), color='k', linestyle='--', lw=1,
           label=f'Random  ({y_test.mean():.3f})')
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend(fontsize=9)

plt.suptitle("Model Performance Curves — Test Set", fontsize=13)
plt.tight_layout()
save("roc_pr_curves.png")

# ── 4c. Feature importance ────────────────────────────────────────────────────
# Get feature names after one-hot encoding
cat_feature_names = (preprocessor.named_transformers_['cat']
                     .get_feature_names_out(CATEGORICAL_FEATURES).tolist())
all_feature_names = NUMERIC_FEATURES + cat_feature_names

importances = best_model.feature_importances_
feat_imp    = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
top20       = feat_imp.head(20)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(top20.index[::-1], top20.values[::-1], color=C_FRAUD, alpha=0.82)
ax.set_xlabel("Feature Importance (XGBoost)")
ax.set_title("Top 20 Most Important Features")
plt.tight_layout()
save("feature_importance.png")

print(f"\n  Top 10 features:")
for feat, imp in feat_imp.head(10).items():
    bar = "█" * int(imp * 300)
    print(f"    {feat:<35} {imp:.4f}  {bar}")

# ── 4d. Score distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
legit_scores = y_test_proba_xgb[y_test == 0]
fraud_scores = y_test_proba_xgb[y_test == 1]
ax.hist(legit_scores, bins=50, alpha=0.65, color=C_LEGIT, label="Legitimate", density=True)
ax.hist(fraud_scores, bins=50, alpha=0.65, color=C_FRAUD, label="Fraud",      density=True)
ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label="Threshold = 0.5")
ax.set_xlabel("Predicted Fraud Probability")
ax.set_ylabel("Density")
ax.set_title("XGBoost — Predicted Score Distribution (Test Set)")
ax.legend()
plt.tight_layout()
save("score_distribution.png")

# ── 4e. Fraud type performance ────────────────────────────────────────────────
print("\n  Performance by fraud type:")

test_with_preds = test_df.copy()
test_with_preds['xgb_pred']  = y_test_pred_xgb
test_with_preds['xgb_proba'] = y_test_proba_xgb

fraud_test = test_with_preds[test_with_preds['true_is_fraud'] == 1]

fig, ax = plt.subplots(figsize=(10, 5))
fraud_types_perf = []

for ft in fraud_test['true_fraud_type'].unique():
    if ft in ('none', 'false_report'): continue
    subset      = fraud_test[fraud_test['true_fraud_type'] == ft]
    detected    = (subset['xgb_pred'] == 1).sum()
    total       = len(subset)
    detect_rate = detected / total * 100
    fraud_types_perf.append({
        'type': ft, 'total': total,
        'detected': detected, 'rate': detect_rate
    })
    print(f"    {ft:<28}  detected {detected}/{total}  ({detect_rate:.1f}%)")

perf_df = pd.DataFrame(fraud_types_perf).sort_values('rate')
colors  = [C_FRAUD if r < 50 else '#FF9800' if r < 75 else '#4CAF50'
           for r in perf_df['rate']]
ax.barh(perf_df['type'], perf_df['rate'], color=colors, alpha=0.85)
ax.axvline(x=50, color='black', linestyle='--', lw=1)
ax.set_xlabel("Detection Rate (%)")
ax.set_title("XGBoost — Detection Rate by Fraud Type\n"
             "(red <50%  orange <75%  green ≥75%)")
for i, (_, row) in enumerate(perf_df.iterrows()):
    ax.text(row['rate'] + 0.5, i,
            f"{row['rate']:.1f}%  (n={row['total']})",
            va='center', fontsize=9)
ax.set_xlim(0, 115)
plt.tight_layout()
save("fraud_type_detection.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SHAP (optional — install with: pip install shap)
# ══════════════════════════════════════════════════════════════════════════════
try:
    import shap
    print("\n" + "─"*60)
    print("  SHAP — Feature Explanation")
    print("─"*60)

    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_pre[:500])  # sample for speed

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_pre[:500],
                      feature_names=all_feature_names,
                      show=False, max_display=15)
    plt.title("SHAP Summary — XGBoost", fontsize=13)
    plt.tight_layout()
    save("shap_summary.png")
    print("  SHAP summary plot saved.")

except ImportError:
    print("\n  SHAP not installed — skipping.")
    print("  Install with:  pip install shap")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
import joblib

print("\n" + "─"*60)
print("  SAVING MODEL")
print("─"*60)

# Save everything needed to make predictions on new listings
model_bundle = {
    "preprocessor":    preprocessor,       # fitted ColumnTransformer
    "model":           best_model,         # best XGBoost
    "feature_names":   ALL_FEATURES,       # list of expected input columns
    "numeric_features":  NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "best_params":     best_params,
    "metrics": {
        "f1":        xgb_f1,
        "precision": xgb_precision,
        "recall":    xgb_recall,
        "auc_roc":   xgb_auc,
        "avg_precision": xgb_ap,
    }
}

joblib.dump(model_bundle, "model_output/fraud_detector.joblib")
print("  Saved → model_output/fraud_detector.joblib")
print(f"  Size  → {os.path.getsize('model_output/fraud_detector.joblib') / 1024:.1f} KB")

# Also save baseline for comparison
baseline_bundle = {
    "preprocessor": preprocessor,
    "model":        baseline_pipe.named_steps['model'],
    "metrics": {"f1": base_f1, "precision": base_precision,
                "recall": base_recall, "auc_roc": base_auc}
}
joblib.dump(baseline_bundle, "model_output/baseline_logistic.joblib")
print("  Saved → model_output/baseline_logistic.joblib")

print("""
  To load and use the model later:

    import joblib
    bundle = joblib.load("model_output/fraud_detector.joblib")
    preprocessor = bundle["preprocessor"]
    model        = bundle["model"]

    X_new_pre = preprocessor.transform(new_listings_df[bundle["feature_names"]])
    predictions  = model.predict(X_new_pre)
    probabilities = model.predict_proba(X_new_pre)[:, 1]
""")


print("\n" + "="*60)
print("  TRAINING COMPLETE")
print("="*60)
print(f"""
  RESULTS SUMMARY (test set, ground truth):
  ──────────────────────────────────────────
  Baseline (Logistic Regression):
    F1={base_f1:.4f}  Precision={base_precision:.4f}  Recall={base_recall:.4f}  AUC={base_auc:.4f}

  Main model (XGBoost):
    F1={xgb_f1:.4f}  Precision={xgb_precision:.4f}  Recall={xgb_recall:.4f}  AUC={xgb_auc:.4f}

  Improvement: +{improvement:.1f}% F1 over baseline

  Output files in model_output/:
    confusion_matrices.png
    roc_pr_curves.png
    feature_importance.png
    score_distribution.png
    fraud_type_detection.png
""")
