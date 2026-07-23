"""
===================================================
EDA — Addis Ababa Rental Fraud Detection Dataset
===================================================
Run this after generating your dataset:
    python addis_rental_fraud_v5_final.py   ← generates the CSVs
    python eda.py                            ← runs this script

Outputs:
    eda_output/  ← folder of saved plots
    console      ← printed summary report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_FILE   = "addis_rental_fraud_v5_final.csv"   # change if your filename differs
OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette
C_LEGIT = "#2196F3"   # blue  — legitimate
C_FRAUD = "#F44336"   # red   — fraud
C_DARK  = "#1a1a2e"
PALETTE = [C_LEGIT, C_FRAUD]

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
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

def etb(x, _=None):
    """Format axis tick labels as ETB amounts."""
    if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if x >= 1_000:     return f"{x/1_000:.0f}K"
    return str(int(x))

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ADDIS ABABA RENTAL FRAUD — EDA")
print("="*60)

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"\n'{CSV_FILE}' not found.\n"
        "Run  python addis_rental_fraud_v5_final.py  first to generate it."
    )

df = pd.read_csv(CSV_FILE)
print(f"\nLoaded: {CSV_FILE}")
print(f"Shape:  {df.shape[0]:,} rows × {df.shape[1]} columns")

# Use ground-truth label for evaluation
df["label"]      = df["true_is_fraud"]
df["label_name"] = df["label"].map({0: "Legitimate", 1: "Fraud"})

legit = df[df["label"] == 0]
fraud = df[df["label"] == 1]

# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1. DATASET OVERVIEW ──────────────────────────────────")
print(f"  Total listings:    {len(df):,}")
print(f"  Legitimate:        {len(legit):,}  ({len(legit)/len(df)*100:.1f}%)")
print(f"  Fraud (true):      {len(fraud):,}  ({len(fraud)/len(df)*100:.1f}%)")
print(f"  Noisy label rate:  {df['is_fraud'].mean()*100:.1f}%")
print(f"\n  Missing values:")
miss = df.isnull().sum()
miss = miss[miss > 0]
if len(miss) == 0:
    print("    None — dataset is complete ✓")
else:
    for col, n in miss.items():
        print(f"    {col}: {n}")

print(f"\n  Price range (ETB/month):")
print(f"    Min:    {df['price'].min():>10,.0f}")
print(f"    Median: {df['price'].median():>10,.0f}")
print(f"    Mean:   {df['price'].mean():>10,.0f}")
print(f"    Max:    {df['price'].max():>10,.0f}")

print(f"\n  Bedroom distribution:")
for b, cnt in df['bedrooms'].value_counts().sort_index().items():
    label = "Studio" if b == 0 else f"{b}BR"
    bar   = "█" * int(cnt / len(df) * 50)
    print(f"    {label:>6}  {bar}  {cnt:,} ({cnt/len(df)*100:.1f}%)")

print(f"\n  Building type distribution:")
for bt, cnt in df['building_type'].value_counts().items():
    print(f"    {bt:<25} {cnt:,} ({cnt/len(df)*100:.1f}%)")

print(f"\n  Area tier distribution:")
for t, cnt in df['area_tier'].value_counts().items():
    print(f"    {t:<10} {cnt:,} ({cnt/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FRAUD RATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2. FRAUD ANALYSIS ────────────────────────────────────")
print("\nGenerating fraud analysis plots...")

# 2a. Fraud type breakdown
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fraud_types = fraud['true_fraud_type'].value_counts()
axes[0].barh(fraud_types.index, fraud_types.values, color=C_FRAUD, alpha=0.85)
axes[0].set_title("Fraud Types (True Labels)")
axes[0].set_xlabel("Count")
for i, v in enumerate(fraud_types.values):
    axes[0].text(v + 2, i, f"{v}  ({v/len(fraud)*100:.1f}%)", va='center', fontsize=9)

# 2b. Fraud rate by building type
fraud_by_bt = df.groupby('building_type')['label'].mean().sort_values(ascending=False)
axes[1].bar(fraud_by_bt.index, fraud_by_bt.values * 100, color=C_FRAUD, alpha=0.85)
axes[1].set_title("Fraud Rate by Building Type")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(fraud_by_bt.values):
    axes[1].text(i, v * 100 + 0.1, f"{v*100:.1f}%", ha='center', fontsize=9)

plt.tight_layout()
save("02_fraud_analysis.png")

# 2c. Fraud rate by area tier
fig, ax = plt.subplots(figsize=(7, 4))
fraud_by_tier = df.groupby('area_tier')['label'].agg(['mean', 'sum', 'count']).reset_index()
fraud_by_tier['rate'] = fraud_by_tier['mean'] * 100
colors = {"budget": "#FF9800", "mid": "#4CAF50", "high_end": "#2196F3"}
bars = ax.bar(fraud_by_tier['area_tier'],
              fraud_by_tier['rate'],
              color=[colors[t] for t in fraud_by_tier['area_tier']],
              alpha=0.88)
ax.set_title("Fraud Rate by Area Tier")
ax.set_ylabel("Fraud Rate (%)")
for bar, (_, row) in zip(bars, fraud_by_tier.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{row['rate']:.1f}%\n(n={int(row['sum'])})", ha='center', fontsize=9)
plt.tight_layout()
save("02b_fraud_by_tier.png")

for bt in df['building_type'].unique():
    rate = df[df['building_type']==bt]['label'].mean()*100
    print(f"    {bt:<25} fraud rate: {rate:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. PRICE ANALYSIS ────────────────────────────────────")
print("\nGenerating price plots...")

# 3a. Price distribution — fraud vs legit
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cap = df['price'].quantile(0.97)
for ax, log in zip(axes, [False, True]):
    ax.hist(legit['price'].clip(upper=cap), bins=60, alpha=0.65,
            color=C_LEGIT, label="Legitimate", density=True)
    ax.hist(fraud['price'].clip(upper=cap), bins=60, alpha=0.65,
            color=C_FRAUD, label="Fraud", density=True)
    ax.set_xlabel("Monthly Rent (ETB)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(etb))
    if log:
        ax.set_yscale('log')
        ax.set_title("Price Distribution (log scale)")
    else:
        ax.set_title("Price Distribution")

plt.tight_layout()
save("03a_price_distribution.png")

# 3b. Median price by building type and tier
fig, ax = plt.subplots(figsize=(12, 5))
pivot = df[df['label']==0].groupby(['building_type','area_tier'])['price'].median().unstack()
pivot = pivot[['budget','mid','high_end']] if 'high_end' in pivot.columns else pivot
pivot.plot(kind='bar', ax=ax, alpha=0.85,
           color=['#FF9800','#4CAF50','#2196F3'])
ax.set_title("Median Monthly Rent by Building Type & Area Tier (Legitimate Only)")
ax.set_ylabel("ETB/month")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(etb))
ax.legend(title="Tier")
plt.tight_layout()
save("03b_price_by_type_tier.png")

# 3c. Price vs size scatter
fig, ax = plt.subplots(figsize=(9, 5))
sample = df.sample(min(3000, len(df)), random_state=42)
for lbl, grp, col in [(0, sample[sample['label']==0], C_LEGIT),
                       (1, sample[sample['label']==1], C_FRAUD)]:
    ax.scatter(grp['size_sqm'], grp['price'], c=col, alpha=0.25, s=12,
               label="Legitimate" if lbl==0 else "Fraud")
ax.set_xlabel("Size (sqm)")
ax.set_ylabel("Monthly Rent (ETB)")
ax.set_title("Price vs Size — Fraud vs Legitimate")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(etb))
ax.set_xlim(0, df['size_sqm'].quantile(0.98))
ax.set_ylim(0, df['price'].quantile(0.97))
ax.legend()
plt.tight_layout()
save("03c_price_vs_size.png")

print(f"  Legit median price:  {legit['price'].median():,.0f} ETB")
print(f"  Fraud median price:  {fraud['price'].median():,.0f} ETB")
print(f"  Fraud premium:       {(fraud['price'].median()/legit['price'].median()-1)*100:.1f}% higher")

# ══════════════════════════════════════════════════════════════════════════════
# 4. BEDROOM & SIZE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 4. PROPERTY FEATURES ─────────────────────────────────")
print("\nGenerating feature distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 4a. Bedrooms
ax = axes[0, 0]
bed_fraud = fraud['bedrooms'].value_counts().sort_index()
bed_legit = legit['bedrooms'].value_counts().sort_index()
all_beds  = sorted(set(bed_fraud.index) | set(bed_legit.index))
x = np.arange(len(all_beds))
w = 0.35
ax.bar(x - w/2, [bed_legit.get(b,0) for b in all_beds], w, label="Legit", color=C_LEGIT, alpha=0.8)
ax.bar(x + w/2, [bed_fraud.get(b,0) for b in all_beds], w, label="Fraud", color=C_FRAUD, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["Studio" if b==0 else f"{b}BR" for b in all_beds])
ax.set_title("Bedrooms — Fraud vs Legitimate")
ax.set_ylabel("Count")
ax.legend()

# 4b. Bathrooms
ax = axes[0, 1]
bath_counts = df.groupby(['bathrooms','label_name']).size().unstack(fill_value=0)
bath_counts.plot(kind='bar', ax=ax, color=PALETTE, alpha=0.85)
ax.set_title("Bathroom Count — Fraud vs Legitimate")
ax.set_xlabel("Bathrooms")
ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=0)
ax.legend(title="")

# 4c. Size distribution
ax = axes[1, 0]
cap_size = df['size_sqm'].quantile(0.97)
ax.hist(legit['size_sqm'].clip(upper=cap_size), bins=50, alpha=0.65,
        color=C_LEGIT, label="Legitimate", density=True)
ax.hist(fraud['size_sqm'].clip(upper=cap_size), bins=50, alpha=0.65,
        color=C_FRAUD, label="Fraud", density=True)
ax.set_xlabel("Size (sqm)")
ax.set_title("Size Distribution")
ax.set_ylabel("Density")
ax.legend()

# 4d. Floor distribution
ax = axes[1, 1]
floor_l = legit['floor_number'].value_counts().sort_index().head(12)
floor_f = fraud['floor_number'].value_counts().sort_index().head(12)
all_floors = sorted(set(floor_l.index) | set(floor_f.index))
x = np.arange(len(all_floors))
ax.bar(x - w/2, [floor_l.get(f,0) for f in all_floors], w, label="Legit", color=C_LEGIT, alpha=0.8)
ax.bar(x + w/2, [floor_f.get(f,0) for f in all_floors], w, label="Fraud", color=C_FRAUD, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(all_floors)
ax.set_title("Floor Number — Fraud vs Legitimate")
ax.set_xlabel("Floor")
ax.set_ylabel("Count")
ax.legend()

plt.suptitle("Property Feature Distributions", fontsize=14, y=1.01)
plt.tight_layout()
save("04_property_features.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. AMENITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 5. AMENITY ANALYSIS ──────────────────────────────────")
print("\nGenerating amenity plots...")

amenities = ['has_generator', 'has_security', 'has_parking', 'has_elevator', 'furnished']
amen_labels = ['Generator', 'Security', 'Parking', 'Elevator', 'Furnished']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 5a. Amenity rate — fraud vs legit
rates_legit = [legit[a].mean() * 100 for a in amenities]
rates_fraud  = [fraud[a].mean() * 100 for a in amenities]
x = np.arange(len(amenities))
w = 0.35
axes[0].bar(x - w/2, rates_legit, w, label="Legitimate", color=C_LEGIT, alpha=0.85)
axes[0].bar(x + w/2, rates_fraud,  w, label="Fraud",      color=C_FRAUD, alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(amen_labels)
axes[0].set_ylabel("Rate (%)")
axes[0].set_title("Amenity Rates — Fraud vs Legitimate")
axes[0].legend()

# 5b. Amenity count distribution
axes[1].hist(legit['amenity_count'], bins=[-0.5,0.5,1.5,2.5,3.5,4.5],
             alpha=0.7, color=C_LEGIT, label="Legitimate", density=True)
axes[1].hist(fraud['amenity_count'], bins=[-0.5,0.5,1.5,2.5,3.5,4.5],
             alpha=0.7, color=C_FRAUD, label="Fraud",      density=True)
axes[1].set_xlabel("Number of Amenities")
axes[1].set_title("Amenity Count Distribution")
axes[1].set_ylabel("Density")
axes[1].set_xticks([0,1,2,3,4])
axes[1].legend()

plt.tight_layout()
save("05_amenity_analysis.png")

print(f"\n  Amenity rates (Legit / Fraud):")
for a, lbl in zip(amenities, amen_labels):
    l_rate = legit[a].mean() * 100
    f_rate = fraud[a].mean() * 100
    print(f"    {lbl:<12}  legit={l_rate:5.1f}%   fraud={f_rate:5.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ANOMALY SCORES
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 6. ANOMALY SCORES ────────────────────────────────────")
print("\nGenerating anomaly score plots...")

prod_scores = [c for c in df.columns if c.startswith('prod_')]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(prod_scores):
    ax = axes[i]
    ax.hist(legit[col].dropna(), bins=40, alpha=0.65,
            color=C_LEGIT, label="Legitimate", density=True)
    ax.hist(fraud[col].dropna(), bins=40, alpha=0.65,
            color=C_FRAUD, label="Fraud", density=True)
    ax.set_title(col.replace('prod_','').replace('_',' ').title())
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

# Hide unused subplot if any
for j in range(len(prod_scores), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Production Anomaly Score Distributions", fontsize=14)
plt.tight_layout()
save("06_anomaly_scores.png")

print(f"\n  Anomaly score means (Legit / Fraud):")
for col in prod_scores:
    l_mean = legit[col].mean()
    f_mean = fraud[col].mean()
    diff   = f_mean - l_mean
    print(f"    {col.replace('prod_',''):<28}  legit={l_mean:.3f}  fraud={f_mean:.3f}  Δ={diff:+.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 7. CORRELATION ANALYSIS ──────────────────────────────")
print("\nGenerating correlation heatmap...")

feature_cols = [
    'price', 'bedrooms', 'bathrooms', 'size_sqm', 'floor_number',
    'furnished', 'has_generator', 'has_parking', 'has_security', 'has_elevator',
    'listing_age_days', 'views', 'contact_clicks',
    'price_per_sqm', 'engagement_rate', 'amenity_count',
    'prod_price_anomaly', 'prod_generator_anomaly',
    'prod_elevator_anomaly', 'prod_engagement_anomaly',
    'area_tier_encoded', 'true_is_fraud'
]
feature_cols = [c for c in feature_cols if c in df.columns]

corr = df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.4,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14, pad=12)
plt.tight_layout()
save("07_correlation_heatmap.png")

# Top correlations with fraud
fraud_corr = corr['true_is_fraud'].drop('true_is_fraud').sort_values(key=abs, ascending=False)
print(f"\n  Top 10 correlations with fraud label:")
for col, val in fraud_corr.head(10).items():
    bar = "█" * int(abs(val) * 20)
    sign = "+" if val > 0 else "-"
    print(f"    {col:<35}  {sign}{abs(val):.3f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. AREA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 8. AREA ANALYSIS ─────────────────────────────────────")
print("\nGenerating area plots...")

# 8a. Median price by area (top 20 areas, legit only)
area_stats = legit.groupby('area').agg(
    median_price=('price', 'median'),
    count=('price', 'count'),
    tier=('area_tier', 'first')
).reset_index().sort_values('median_price', ascending=True)
top_areas = area_stats[area_stats['count'] >= 50].tail(22)

tier_colors = {"budget": "#FF9800", "mid": "#4CAF50", "high_end": "#2196F3"}
bar_colors  = [tier_colors.get(t, "grey") for t in top_areas['tier']]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top_areas['area'], top_areas['median_price'],
               color=bar_colors, alpha=0.88)
ax.set_xlabel("Median Monthly Rent (ETB)")
ax.set_title("Median Rent by Area — Legitimate Listings\n"
             "(orange=budget  green=mid  blue=high_end)", fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(etb))
for bar, (_, row) in zip(bars, top_areas.iterrows()):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            etb(row['median_price']), va='center', fontsize=8)
plt.tight_layout()
save("08a_price_by_area.png")

# 8b. Fraud rate by area (areas with 80+ listings)
area_fraud = df.groupby('area').agg(
    fraud_rate=('label', 'mean'),
    count=('label', 'count')
).reset_index()
area_fraud = area_fraud[area_fraud['count'] >= 80].sort_values('fraud_rate', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(area_fraud['area'], area_fraud['fraud_rate'] * 100, color=C_FRAUD, alpha=0.82)
ax.set_xlabel("Fraud Rate (%)")
ax.set_title("Fraud Rate by Area (areas with 80+ listings)")
for i, (_, row) in enumerate(area_fraud.iterrows()):
    ax.text(row['fraud_rate']*100 + 0.1, i, f"{row['fraud_rate']*100:.1f}%",
            va='center', fontsize=8)
plt.tight_layout()
save("08b_fraud_by_area.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. ENGAGEMENT METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 9. ENGAGEMENT METRICS ────────────────────────────────")
print("\nGenerating engagement plots...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, col, title in zip(axes,
    ['views', 'contact_clicks', 'engagement_rate'],
    ['Views', 'Contact Clicks', 'Engagement Rate (clicks/views)']):
    cap = df[col].quantile(0.97)
    ax.hist(legit[col].clip(upper=cap), bins=45, alpha=0.65,
            color=C_LEGIT, label="Legit", density=True)
    ax.hist(fraud[col].clip(upper=cap), bins=45, alpha=0.65,
            color=C_FRAUD, label="Fraud", density=True)
    ax.set_title(title)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.suptitle("Engagement Metric Distributions", fontsize=13)
plt.tight_layout()
save("09_engagement_metrics.png")

for col in ['views', 'contact_clicks', 'engagement_rate', 'listing_age_days']:
    print(f"    {col:<22}  legit median={legit[col].median():.2f}   "
          f"fraud median={fraud[col].median():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. FRAUD TYPE DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 10. FRAUD TYPE DEEP DIVE ─────────────────────────────")
print("\nGenerating fraud type comparison plots...")

fraud_typed = fraud[fraud['true_fraud_type'] != 'none'].copy()
fraud_types_list = fraud_typed['true_fraud_type'].unique()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

metrics = ['price', 'size_sqm', 'prod_price_anomaly',
           'prod_generator_anomaly', 'engagement_rate', 'amenity_count']
titles  = ['Price (ETB)', 'Size (sqm)', 'Price Anomaly Score',
           'Generator Anomaly', 'Engagement Rate', 'Amenity Count']

for ax, metric, title in zip(axes, metrics, titles):
    data = [fraud_typed[fraud_typed['true_fraud_type']==ft][metric].dropna()
            for ft in fraud_types_list]
    ax.boxplot(data, labels=[ft.replace('_','\n') for ft in fraud_types_list],
               patch_artist=True,
               boxprops=dict(facecolor=C_FRAUD, alpha=0.5),
               medianprops=dict(color='black', linewidth=2))
    ax.set_title(title)
    ax.tick_params(axis='x', labelsize=7)

plt.suptitle("Feature Distributions by Fraud Type", fontsize=14)
plt.tight_layout()
save("10_fraud_type_deep_dive.png")

print(f"\n  Key stats by fraud type:")
for ft in fraud_typed['true_fraud_type'].value_counts().index:
    subset = fraud_typed[fraud_typed['true_fraud_type']==ft]
    print(f"\n    [{ft}]  n={len(subset)}")
    print(f"      Price:           median {subset['price'].median():>8,.0f} ETB  "
          f"(legit: {legit['price'].median():,.0f})")
    print(f"      Price anomaly:   mean   {subset['prod_price_anomaly'].mean():.3f}  "
          f"(legit: {legit['prod_price_anomaly'].mean():.3f})")
    print(f"      Size inflate:    median {subset['size_sqm'].median():>5.0f} sqm  "
          f"(legit: {legit['size_sqm'].median():.0f})")

# ══════════════════════════════════════════════════════════════════════════════
# 11. LABEL NOISE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 11. LABEL NOISE ANALYSIS ─────────────────────────────")

missed_fraud   = ((df['true_is_fraud']==1) & (df['is_fraud']==0)).sum()
false_reports  = ((df['true_is_fraud']==0) & (df['is_fraud']==1)).sum()
total_fraud_true = df['true_is_fraud'].sum()

print(f"  True fraud cases:       {total_fraud_true}")
print(f"  Missed (labeled legit): {missed_fraud}  "
      f"({missed_fraud/total_fraud_true*100:.1f}% of true fraud)")
print(f"  False reports:          {false_reports}  "
      f"({false_reports/(len(df)-total_fraud_true)*100:.1f}% of true legit)")
print(f"\n  → Train on 'is_fraud' (noisy label)")
print(f"  → Evaluate final model on 'true_is_fraud' (ground truth)")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  EDA COMPLETE")
print("="*60)
print(f"\n  All plots saved to:  ./{OUTPUT_DIR}/")
print(f"\n  Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"    {f}")

print(f"""
  KEY FINDINGS FOR YOUR REPORT:
  ─────────────────────────────
  • Dataset: {len(df):,} listings, {len(fraud)/len(df)*100:.1f}% fraud (true label)
  • Most common fraud: {fraud['true_fraud_type'].value_counts().index[0]}
  • Fraud raises median price by {(fraud['price'].median()/legit['price'].median()-1)*100:.0f}%
  • Strongest fraud signal: {fraud_corr.head(1).index[0]}
    (corr = {fraud_corr.iloc[0]:.3f} with true_is_fraud)
  • Hardest fraud to detect: subtle_mixed
    (small, probabilistic, multi-signal deception)
""")
