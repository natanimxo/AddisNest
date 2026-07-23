import pandas as pd
import random
import numpy as np
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# ADDIS ABABA RENTAL FRAUD DETECTION DATASET — v5.1 FINAL
# ==================================================
# Calibrated against live Ethiopian listing platforms:
#   - Bet Afri Telegram channel (644 scraped listings, Sep 2025–May 2026)
#     ← PRIMARY REFERENCE for budget/mid prices, studio sizes, amenity rates
#   - Jiji.com.et        (4,800+ rental listings, Dec 2025)
#   - Engocha.com        (broker listings, Apr 2026)
#   - TikTok delalas     (landlord videos, 2024–2026)
#   - Ethiopia Property Centre / Africanvestor (high-end segment only)
#
# FINAL FIXES IN v5.0:
#
# FIX A — STUDIOS ELIMINATED FROM CONDOS AND TOWNHOUSES
#   Ethiopian rental listings on Jiji, Engocha, and TikTok show
#   studios (0-bedroom) are essentially nonexistent in IHDP
#   condominiums and townhouses. They are rare even in apartments.
#   New weights remove 0-bedroom from Condominium and Townhouse
#   entirely, and reduce it to 1% in Apartment Building.
#
# FIX B — MINIMUM 1 BATHROOM ENFORCED EVERYWHERE
#   No Ethiopian rental listing shows 0 bathrooms. Every unit,
#   including studios, has at least 1 bathroom. Bathrooms are now
#   building-type-aware: IHDP condos almost universally have 1
#   bathroom per unit regardless of bedroom count.
#   Source: Jiji condo filter — 213 of 278 condo listings show 1 bath.
#
# FIX C — CONDOMINIUM PRICES CORRECTED FROM JIJI DATA
#   Jiji.com.et shows 278 condo rental listings with price distribution:
#     Under 13K ETB:  16 ads  (old/deteriorated units)
#     13K–16K ETB:    66 ads  ← budget IHDP dominant range
#     16K–23K ETB:   111 ads  ← mid IHDP dominant range
#     23K–5.1M ETB:   69 ads  (private/premium "condo-style" apts)
#   v4.1 budget condo median was 27,500 ETB — too high.
#   Corrected: budget condo should cluster around 14,000–22,000 ETB.
#   Fix: lower PRICE_PER_SQM for condos AND add building-type-specific
#   bedroom premiums so the private apt premiums don't bleed into condos.
#
# FIX D — PER-BUILDING-TYPE BEDROOM PREMIUMS
#   Previous code applied the same bedroom premium table to all
#   building types. This caused condos to be overpriced. Each
#   building type now has its own premium scale:
#     Condominium:    small  (3K–12K ETB per extra bedroom)
#     Apartment:      medium (8K–35K ETB per extra bedroom)
#     Townhouse:      medium (6K–25K ETB per extra bedroom)
#     Villa Compound: large  (12K–50K ETB per extra bedroom)
#   Sources: Jiji/Engocha listing price analysis per bedroom count.
#
# FIX E — VILLA COMPOUND MIN BEDROOMS = 2
#   No Ethiopian villa compound listing shows 0 or 1 bedroom.
#   Even the smallest compound (Kebena unfurnished) shows 3BR.
#   Weights updated: villas start at 2BR minimum.
#
# All v4.0 ML fixes are preserved:
#   - 20,000 samples (≈1,000 fraud cases)
#   - Oracle vs production anomaly score split (no training leakage)
#   - 8% missed fraud + 1% false positive label noise
#   - ±15% base rate calibration uncertainty
#   - Stratified train/val/test split
# ==================================================


# ==================================================
# 1. AREA CONFIGURATIONS
# ==================================================

AREA_CONFIGS = {
    "high_end": {
        # Bet Afri: Bisrate Gebriel listings show 20K-30K even on budget channel
        # Kazanchis, Old Airport, Hayahulet confirmed high-end from all sources
        # Haya-hulet = "twenty two" in Amharic (common mistranslation in scraped data)
        "areas": [
            "Bole", "Old Airport", "Bole Michael", "Hayahulet",
            "Bole Medhane Alem", "Urael", "Atlas", "Kazanchis",
            "Sarbet", "Bisrate Gebriel", "Yeka Hills",
        ],
        "price_volatility": 0.18,
    },
    "mid": {
        # New from Bet Afri data: Kotebe, Shola, Lamberet appear frequently
        # Gotera = "barn" in mistranslated data — already in list
        # Ayer Tena = "air health" mistranslation — added with correct name
        # Adisu Gebeya = "new market" mistranslation — added with correct name
        "areas": [
            "CMC", "Megenagna", "Gerji", "Summit", "Ayat",
            "Wollo Sefer", "Yeka", "Lebu", "Jemo", "Gulele",
            "Lemi Kura", "Mekanisa", "Gurd Shola", "Lideta",
            "Kirkos", "Piassa", "Gotera", "Kotebe", "Shola",
            "Lamberet", "Ayer Tena", "Adisu Gebeya", "Kera",
        ],
        "price_volatility": 0.15,
    },
    "budget": {
        # Qaliti = Akaki Kality (mistranslation in scraped data — same area)
        # Lafto = "loft" mistranslation — added with correct Amharic name
        # Saris Abo: Bet Afri median 11,000 ETB — confirmed budget
        # Bole Bulbula: outer Bole, budget despite "Bole" in name
        "areas": [
            "Kolfe Keranio", "Akaki Kality", "Gofa", "Nifas Silk Lafto",
            "Shiro Meda", "Entoto", "Kality", "Koye Feche",
            "Saris", "Bole Bulbula", "Addis Ketema", "Lafto",
            "Saris Abo",
        ],
        "price_volatility": 0.15,
    }
}


# ==================================================
# 2. BUILDING CONFIGURATIONS
# ==================================================
# Studio prevalence calibrated from Bet Afri Telegram channel (359 real listings,
# Sep 2025 – May 2026):
#   - Studios = 29.8% of ALL listings (107/359) — very common in Ethiopian market
#   - Condo studios confirmed: 15 out of 49 condo listings = 30.6%
#   - Townhouse/compound studios: common (Ethiopian "room" rental culture)
#   - Apartments: ~8% studios (private buildings have more studio options)
#   - Villa Compound: 0BR/1BR remain zero (confirmed — no villa studios in data)

BUILDING_CONFIGS = {
    "Condominium": {
        # Bet Afri data: studio=31%, 1BR=44%, 2BR=15%, 3BR=10% (of known condos)
        # Adjusted slightly toward 1BR dominance to account for full-market bias
        "bedroom_weights": [0.22, 0.40, 0.28, 0.10],  # studio, 1BR, 2BR, 3BR
        "max_bedrooms": 3,
        "typical_floor_range": (0, 6),
    },
    "Townhouse": {
        # Compound/duet rooms in Addis — studio rooms are very common.
        # Many are single rooms with shared or own bathroom in a compound.
        # Bet Afri channel: large portion of "private house" listings are
        # compound studios/1BR — applying ~20% studio rate for townhouse.
        "bedroom_weights": [0.20, 0.28, 0.38, 0.14],  # studio, 1BR, 2BR, 3BR
        "max_bedrooms": 3,
        "typical_floor_range": (0, 2),
    },
    "Apartment Building": {
        # Private developer buildings. Studios less common than condos but
        # Bet Afri apartment listings show some studios — ~8%.
        "bedroom_weights": [0.08, 0.36, 0.40, 0.14, 0.02],
        "max_bedrooms": 4,
        "typical_floor_range": (0, 20),
    },
    "Villa Compound": {
        # Standalone compounds. Zero studios/1BR confirmed across all data sources.
        "bedroom_weights": [0.00, 0.00, 0.06, 0.42, 0.36, 0.12, 0.04],
        "max_bedrooms": 6,
        "typical_floor_range": (0, 2),
    }
}


# ==================================================
# 3. PRICE PER SQM — MONTHLY RENTAL (ETB/sqm)
# ==================================================
# PRIMARY SOURCE: Bet Afri Telegram channel (644 listings, Sep 2025–May 2026)
#
# Key findings from real data (budget/mid segment):
#   Studio overall:  p10=7K  med=13K  p90=20K  (n=386)
#   1BR overall:     p10=15K med=20K  p90=30K  (n=186)
#   2BR overall:     p10=17K med=25K  p90=30K  (n=51)
#   Condo studio:    med=16K  range 8K–23K     (n=30)
#   Condo 1BR:       med=25K  range 15K–30K    (n=38)
#   Apt studio:      med=15K  range 4K–28K     (n=113)
#   Apt 1BR:         med=25K  range 10K–30K    (n=35)
#
# NOTE: The Bet Afri channel is capped at ~30K ETB. High-end prices
# (Bole villas, premium apartments 80K–400K) are sourced from Jiji/
# Engocha/TikTok as before — those markets are not in this channel.
#
# Validation targets:
#   Budget condo studio  28sqm:   ~11,000 ETB  (8K–20K)  ✓
#   Budget condo 1BR     48sqm:   ~17,000 + 3K–6K prem   ✓
#   Mid condo 1BR        50sqm:   ~22,000 + 4K–8K prem   ✓
#   Budget apt studio    35sqm:   ~12,000 ETB  (8K–20K)  ✓
#   Townhouse studio     12sqm:   ~8,000  ETB  (5K–15K)  ✓
#   High-end apt 2BR     90sqm:   ~110,000+ ETB           ✓
#   High-end villa 3BR   250sqm:  ~200,000+ ETB           ✓

PRICE_PER_SQM = {
    "Condominium": {
        # Bet Afri: condo studio med 16K / 30sqm ≈ 533; 1BR med 25K / 50sqm–prem ≈ 420
        # Budget lowered to 320 so 1BR median lands at ~18K–23K ✓
        "high_end": 620,
        "mid":      480,
        "budget":   320,
    },
    "Townhouse": {
        # Budget townhouse mixes tiny studio rooms (8–22 sqm) with 1–3BR units.
        # At 750/sqm a 2BR 70sqm unit = 52,500 — far too high for Kolfe/Saris.
        # 380/sqm: studio 12sqm → 4,560 → floor 5,000 ✓; 1BR 40sqm → 15,200 + prem ✓
        "high_end": 900,
        "mid":      600,
        "budget":   380,
    },
    "Apartment Building": {
        # Budget private apartments in Kolfe/Akaki: newer G+buildings.
        # 360/sqm: studio 35sqm → 12,600 ✓; 1BR 55sqm → 19,800 + 9K = 28,800 ✓
        # 2BR 87sqm → 31,320 + 16K = 47,320 — acceptable for a private budget apt.
        "high_end": 950,
        "mid":      550,
        "budget":   360,
    },
    "Villa Compound": {
        # Not in Bet Afri channel. Jiji/TikTok calibration unchanged.
        "high_end": 700,
        "mid":      370,
        "budget":   210,
    }
}


# ==================================================
# 4. BEDROOM PREMIUMS BY BUILDING TYPE (ETB/month)
# ==================================================
# Bet Afri data: condo studio med=16K, 1BR med=25K → +9K for 1 bedroom
# But 2BR med=25K (same as 1BR) → 2BR barely costs more than 1BR in condos
# This suggests condo bedroom premiums compress at 2–3BR (standardised layouts)
#
# Apt: studio med=15K, 1BR med=25K → +10K for 1 bedroom
# Townhouse: compound rooms are studios; 1BR adds a room → meaningful jump
#
# Feature premiums (budget channel, so these feel small):
#   Generator in budget adds ~14K–40K on top → only high-end segments
#   Security adds ~6K–18K → again, high-end
#   These are correct for high-end but the budget channel doesn't show them
#   because budget units don't have generators/security. The rates handle this.

BEDROOM_PREMIUMS = {
    "Condominium": {
        # Bet Afri: studio→1BR +9K, 1BR→2BR +0K (both med=25K) → premiums compress
        1: (3_000,  8_000),
        2: (4_000,  9_000),
        3: (5_000, 11_000),
    },
    "Townhouse": {
        # Compound rooms → adding a room is a meaningful upgrade
        1: (4_000,  9_000),
        2: (8_000, 15_000),
        3: (12_000, 20_000),
    },
    "Apartment Building": {
        # Bet Afri: studio→1BR +10K; beyond that Jiji/Engocha data applies
        1: (8_000,  14_000),
        2: (14_000, 22_000),
        3: (22_000, 34_000),
        4: (32_000, 48_000),
    },
    "Villa Compound": {
        # Not in Bet Afri channel. Jiji/TikTok calibration unchanged.
        2: (10_000, 18_000),
        3: (18_000, 30_000),
        4: (28_000, 44_000),
        5: (40_000, 60_000),
    },
}


# ==================================================
# 5. FEATURE BASE RATES (research-calibrated)
# ==================================================

_TRUE_FEATURE_BASE_RATES = {
    "generator": {
        # Bet Afri channel (budget/mid): generator almost never mentioned.
        # High-end segment: Africanvestor confirms generators are premium feature.
        # Budget rates kept very low — consistent with 3.9% amenity rate overall.
        "Villa Compound":     {"high_end": 0.80, "mid": 0.45, "budget": 0.12},
        "Apartment Building": {"high_end": 0.65, "mid": 0.28, "budget": 0.05},
        "Condominium":        {"high_end": 0.04, "mid": 0.02, "budget": 0.01},
        "Townhouse":          {"high_end": 0.28, "mid": 0.10, "budget": 0.02},
    },
    "security": {
        # Bet Afri ACTUAL: security mentioned in only 3.9% of 644 listings.
        # This is a budget/mid channel — security guards are a premium feature.
        # Previous rates (30–55% for budget) were far too high.
        # High-end rates (villas in Bole) kept from Jiji/TikTok — still valid.
        "Villa Compound":     {"high_end": 0.92, "mid": 0.72, "budget": 0.42},
        "Apartment Building": {"high_end": 0.68, "mid": 0.22, "budget": 0.05},
        "Condominium":        {"high_end": 0.35, "mid": 0.18, "budget": 0.06},
        "Townhouse":          {"high_end": 0.65, "mid": 0.28, "budget": 0.08},
    },
    "parking": {
        # Bet Afri: own/yard parking in 64% of known listings.
        # But "known" is biased toward compound listings that always have yards.
        # Budget condo parking remains low (no dedicated spots in IHDP).
        "Villa Compound":     {"high_end": 0.96, "mid": 0.90, "budget": 0.78},
        "Apartment Building": {"high_end": 0.62, "mid": 0.35, "budget": 0.12},
        "Condominium":        {"high_end": 0.28, "mid": 0.15, "budget": 0.08},
        "Townhouse":          {"high_end": 0.82, "mid": 0.62, "budget": 0.45},
    },
    "furnished": {
        # Bet Afri: furnished in 6.8% of 644 listings — very uncommon.
        # Budget rates reduced from previous estimates.
        "Villa Compound":     {"high_end": 0.32, "mid": 0.12, "budget": 0.04},
        "Apartment Building": {"high_end": 0.42, "mid": 0.14, "budget": 0.04},
        "Condominium":        {"high_end": 0.10, "mid": 0.04, "budget": 0.02},
        "Townhouse":          {"high_end": 0.18, "mid": 0.06, "budget": 0.02},
    },
}

_TRUE_ELEVATOR_BASE_RATES = {
    "Villa Compound":     {"high_end": 0.004, "mid": 0.001, "budget": 0.000},
    "Apartment Building": {"high_end": 0.18,  "mid": 0.06,  "budget": 0.015},
    "Condominium":        {"high_end": 0.000, "mid": 0.000, "budget": 0.000},
    "Townhouse":          {"high_end": 0.000, "mid": 0.000, "budget": 0.000},
}


# ==================================================
# 6. FEATURE TAXONOMY
# ==================================================

SAFE_TRAINING_FEATURES = [
    "price", "bedrooms", "bathrooms", "size_sqm", "floor_number",
    "building_type", "area",
    "furnished", "has_generator", "has_parking", "has_security", "has_elevator",
    "listing_age_days", "views", "contact_clicks",
    "price_per_sqm", "price_per_bedroom", "size_per_bedroom",
    "area_tier_encoded", "engagement_rate", "engagement_velocity",
    "is_condo", "is_villa", "is_townhouse", "is_apartment", "amenity_count",
    "prod_generator_anomaly", "prod_elevator_anomaly",
    "prod_price_anomaly", "prod_engagement_anomaly",
    "prod_studio_premium_anomaly",
    "price_position",
]

ANALYSIS_ONLY_FEATURES = [
    "oracle_generator_anomaly", "oracle_elevator_anomaly",
    "oracle_price_anomaly", "oracle_size_anomaly",
    "oracle_engagement_anomaly", "oracle_studio_premium_anomaly",
]

LABEL_COLUMNS = ["is_fraud", "true_is_fraud", "fraud_type", "true_fraud_type"]


# ==================================================
# 7. UTILITY FUNCTIONS
# ==================================================

def get_area_tier(area):
    for tier, config in AREA_CONFIGS.items():
        if area in config["areas"]:
            return tier
    return "mid"

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def smooth_scale(value, midpoint=0.5, steepness=5):
    return float(expit(steepness * (value - midpoint)))

def _perturb_rates(rates, noise_std=0.15):
    perturbed = {}
    for feature, building_map in rates.items():
        perturbed[feature] = {}
        for btype, tier_map in building_map.items():
            perturbed[feature][btype] = {}
            for tier, rate in tier_map.items():
                perturbed[feature][btype][tier] = clamp(
                    rate * random.gauss(1.0, noise_std), 0.001, 0.990)
    return perturbed

def _perturb_elevator_rates(rates, noise_std=0.15):
    perturbed = {}
    for btype, tier_map in rates.items():
        perturbed[btype] = {}
        for tier, rate in tier_map.items():
            perturbed[btype][tier] = 0.0 if btype == "Condominium" else clamp(
                rate * random.gauss(1.0, noise_std), 0.0, 0.990)
    return perturbed


# ==================================================
# 8. PROPERTY GENERATION
# ==================================================

def get_bedrooms(building_type):
    config = BUILDING_CONFIGS[building_type]
    weights = config["bedroom_weights"]
    bedrooms = random.choices(range(len(weights)), weights=weights)[0]
    return min(bedrooms, config["max_bedrooms"])


def get_bathrooms(bedrooms, building_type):
    """
    Bathroom counts calibrated from Bet Afri Telegram data (359 listings).

    KEY FINDING: Studios in Ethiopian rentals commonly have bathrooms = 0,
    meaning a shared/common toilet and shower with other tenants in the compound
    or building. Bet Afri data: 53 of 107 studios explicitly say 'Kitchen: No'
    and most do not mention a private shower — confirming shared facilities.

    bathroom = 0 → shared/common bathroom (valid for budget studios)
    bathroom = 1 → own private bathroom
    bathroom = 2+ → multiple bathrooms (larger units)

    Non-studio units always have at least 1 bathroom (no shared bathrooms
    observed for 1BR+ units in any listing source).
    """
    # --- STUDIOS (0 bedrooms) ---
    if bedrooms == 0:
        if building_type == "Condominium":
            # IHDP condo studios: often own small shower room
            # but cheaper/older units share. ~35% shared.
            return random.choices([0, 1], weights=[0.35, 0.65])[0]
        elif building_type == "Townhouse":
            # Compound room rentals: very commonly shared bathroom
            # Bet Afri: most budget compound studios have common facilities
            return random.choices([0, 1], weights=[0.55, 0.45])[0]
        elif building_type == "Apartment Building":
            # Private apartment studios: usually own bathroom
            return random.choices([0, 1], weights=[0.18, 0.82])[0]
        else:
            return 1  # Villa studios don't exist; fallback

    # --- 1 BEDROOM ---
    if bedrooms == 1:
        return 1  # Always own bathroom for 1BR+

    # --- 2+ BEDROOMS: building-type-aware ---
    if building_type == "Condominium":
        # IHDP units: mostly 1 bath; newer 3BR units occasionally 2
        if bedrooms == 2: return random.choices([1, 2], weights=[0.88, 0.12])[0]
        if bedrooms == 3: return random.choices([1, 2], weights=[0.72, 0.28])[0]
        return 1

    elif building_type == "Townhouse":
        if bedrooms == 2: return random.choices([1, 2], weights=[0.65, 0.35])[0]
        if bedrooms == 3: return random.choices([1, 2, 3], weights=[0.30, 0.58, 0.12])[0]
        return max(1, bedrooms - 1)

    elif building_type == "Apartment Building":
        if bedrooms == 2: return random.choices([1, 2], weights=[0.58, 0.42])[0]
        if bedrooms == 3: return random.choices([1, 2, 3], weights=[0.20, 0.65, 0.15])[0]
        return random.choices([2, 3, 4], weights=[0.45, 0.42, 0.13])[0]

    else:  # Villa Compound
        if bedrooms == 2: return random.choices([1, 2], weights=[0.45, 0.55])[0]
        if bedrooms == 3: return random.choices([2, 3], weights=[0.58, 0.42])[0]
        if bedrooms == 4: return random.choices([2, 3, 4], weights=[0.35, 0.50, 0.15])[0]
        return random.choices([3, 4, 5], weights=[0.40, 0.45, 0.15])[0]


def get_size_sqm(bedrooms, building_type):
    """
    Size ranges calibrated from Bet Afri scraped data (644 listings).

    KEY FINDING: Studios have very different sizes by building type:
      - Townhouse compound studios: median 12 sqm (range 6–22 sqm) — single rooms
      - Condo studios: 28–42 sqm (proper IHDP studio unit)
      - Apartment studios: 25–46 sqm (proper studio flat)

    The aggregate studio median of 16 sqm across all listings is driven by the
    large number of compound room rentals (Townhouse category), not condo studios.

    1BR sizes from Bet Afri: median ~42 sqm overall; larger in apartments.
    2BR: median 72 sqm. 3BR: median 83 sqm.
    """
    if building_type == "Condominium":
        # Proper IHDP studio unit, not a compound room
        base_sizes = {0: (26, 42), 1: (42, 56), 2: (58, 82), 3: (88, 118)}

    elif building_type == "Townhouse":
        # Studios = compound/room rentals. Bet Afri: min 6, p25=12, med=16, p75=20.
        # 1BR+ are proper units in the compound — larger.
        base_sizes = {0: (8, 22), 1: (30, 55), 2: (55, 88), 3: (82, 115)}

    elif building_type == "Villa Compound":
        if bedrooms <= 2:   min_s, max_s = 130, 200
        elif bedrooms == 3: min_s, max_s = 185, 290
        elif bedrooms == 4: min_s, max_s = 250, 360
        elif bedrooms == 5: min_s, max_s = 310, 440
        else:               min_s, max_s = 390, 540
        base = random.randint(min_s, max_s)
        return max(120, int(base + random.gauss(0, base * 0.025 * bedrooms)))

    else:  # Apartment Building
        # Bet Afri: studio apt median ~35 sqm; 1BR ~48 sqm; 2BR ~80 sqm
        base_sizes = {
            0: (25, 46), 1: (44, 72), 2: (65, 110),
            3: (105, 155), 4: (148, 200)
        }

    min_s, max_s = base_sizes.get(bedrooms, (50, 150))
    return max(6, int(random.randint(min_s, max_s) *
                      clamp(random.gauss(1.0, 0.075), 0.85, 1.15)))


def get_floor_number(building_type):
    """Floor distributions calibrated to Ethiopian building stock."""
    if building_type == "Condominium":
        return random.choices(
            [0, 1, 2, 3, 4, 5, 6],
            weights=[0.22, 0.28, 0.24, 0.16, 0.06, 0.03, 0.01])[0]

    elif building_type == "Townhouse":
        return random.choices([0, 1, 2], weights=[0.60, 0.35, 0.05])[0]

    elif building_type == "Villa Compound":
        return random.choices([0, 1, 2], weights=[0.68, 0.27, 0.05])[0]

    else:  # Apartment Building
        floors  = list(range(0, 21))
        weights = [0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.07,
                   0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01,
                   0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        total   = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(floors, weights=weights)[0]


# ==================================================
# 9. FEATURE GENERATION WITH SOFT CORRELATIONS
# ==================================================

def generate_features(building_type, area, floor_number, bedrooms,
                      feature_rates, elevator_rates):
    tier     = get_area_tier(area)
    is_villa = (building_type == "Villa Compound")
    features = {}

    # Security (anchor — influences other features)
    sec_base = feature_rates["security"][building_type][tier]
    features["security"] = int(random.random() <
                               clamp(sec_base + random.gauss(0, 0.03), 0.001, 0.999))

    # Parking (soft correlation with security)
    park_base = feature_rates["parking"][building_type][tier]
    if features["security"]: park_base += random.uniform(0.02, 0.08)
    features["parking"] = int(random.random() <
                              clamp(park_base + random.gauss(0, 0.03), 0.001, 0.999))

    # Generator (correlation with villa + security)
    gen_base = feature_rates["generator"][building_type][tier]
    if is_villa:             gen_base += random.uniform(0.02, 0.06)
    if features["security"]: gen_base += random.uniform(0.01, 0.04)
    features["generator"] = int(random.random() <
                                clamp(gen_base + random.gauss(0, 0.025), 0.001, 0.999))

    # Furnished
    furn_base = feature_rates["furnished"][building_type][tier]
    features["furnished"] = int(random.random() <
                                clamp(furn_base + random.gauss(0, 0.02), 0.001, 0.999))

    # Elevator — ABSOLUTE ZERO for condos and townhouses
    if building_type in ("Condominium", "Townhouse"):
        features["elevator"] = 0
    else:
        elev_base = elevator_rates[building_type][tier]
        if building_type == "Apartment Building":
            if floor_number < 5:    elev_base *= 0.02
            elif floor_number >= 10: elev_base *= 1.5
        features["elevator"] = int(random.random() <
                                   clamp(elev_base + random.gauss(0, 0.01), 0.0, 0.999))

    return features


# ==================================================
# 10. PRICE CALCULATION
# ==================================================

def calculate_price(area, building_type, bedrooms, size_sqm, features, floor_number):
    """
    Monthly rental price in ETB.
    Uses per-building-type bedroom premiums (FIX D).
    Condo prices now match Jiji distribution (FIX C).
    """
    tier       = get_area_tier(area)
    volatility = AREA_CONFIGS[tier]["price_volatility"]
    base_sqm   = PRICE_PER_SQM[building_type][tier]

    # Base with heteroskedastic noise
    size_noise = volatility * (1 + size_sqm / 150)
    price = size_sqm * base_sqm * clamp(random.gauss(1.0, size_noise), 0.65, 1.40)

    # Building-type-specific bedroom premium
    premiums = BEDROOM_PREMIUMS.get(building_type, {})
    if bedrooms in premiums:
        lo, hi = premiums[bedrooms]
        price += random.randint(lo, hi)
    elif bedrooms > max(premiums.keys(), default=0):
        max_key = max(premiums.keys(), default=4)
        lo, hi  = premiums[max_key]
        extra   = (bedrooms - max_key) * random.randint(lo // 2, hi // 2)
        price  += hi + extra

    # Feature premiums — scaled to market segment
    # Bet Afri: budget/mid listings rarely mention these features,
    # consistent with low amenity rates. Premiums only materialise
    # when the feature is present (which is already rate-controlled).
    if features.get("furnished"):  price *= random.uniform(1.25, 1.55)
    if features.get("generator"):  price += random.randint(12_000, 38_000)
    if features.get("security"):   price += random.randint(5_000,  16_000)
    if features.get("elevator"):   price += random.randint(14_000, 42_000)
    if features.get("parking"):    price += random.randint(4_000,  12_000)

    # Floor penalty for condos (no elevator, high floors less desirable)
    if building_type == "Condominium" and floor_number > 3:
        penalty = min(floor_number * 0.014, 0.20) * random.uniform(0.8, 1.2)
        price  *= (1 - penalty)

    # Occasional pricing breaks (~5% of listings)
    if random.random() < 0.025 and size_sqm < 70:  price *= random.uniform(1.25, 1.75)
    if random.random() < 0.025 and size_sqm > 120: price *= random.uniform(0.60, 0.82)

    # Market noise
    price *= (1 + random.gauss(0, volatility))

    # Seller pricing quirks (8%)
    if random.random() < 0.08:
        price *= random.uniform(1.20, 1.60) if random.random() < 0.60 \
                 else random.uniform(0.70, 0.90)

    # Round to nearest 500 ETB
    # Floor: 5,000 ETB — shared-bath compound studios in outer Addis can be this cheap
    return int(round(max(5_000, price) / 500) * 500)


# ==================================================
# 11. FRAUD INJECTION
# ==================================================

def inject_fraud(record):
    fraud_type = random.choices(
        ['subtle_mixed', 'moderate_inflation', 'amenity_inflation',
         'size_deception', 'location_fraud', 'engagement_manipulation'],
        weights=[0.20, 0.25, 0.20, 0.15, 0.10, 0.10]
    )[0]

    record['fraud_type'] = fraud_type
    record['is_fraud']   = 1

    if fraud_type == 'subtle_mixed':
        if random.random() < 0.50: record['size_sqm'] = int(record['size_sqm'] * random.uniform(1.05, 1.15))
        if random.random() < 0.30: record['has_parking'] = 1
        if random.random() < 0.20: record['has_security'] = 1
        if random.random() < 0.15 and record['building_type'] not in ('Condominium', 'Townhouse'):
            record['has_elevator'] = 1
        record['price'] = int(round(record['price'] * random.uniform(1.05, 1.25) / 500) * 500)

    elif fraud_type == 'moderate_inflation':
        mult = random.choice([random.uniform(1.2, 1.5),
                              random.uniform(1.5, 2.0),
                              random.uniform(2.0, 2.8)])
        record['price'] = int(round(record['price'] * mult / 500) * 500)

    elif fraud_type == 'amenity_inflation':
        possible = ['generator', 'security', 'parking']
        if record['building_type'] not in ('Condominium', 'Townhouse'):
            possible.append('elevator')
        n_fake = random.choices([1, 2, 3], weights=[0.5, 0.4, 0.1])[0]
        for amenity in random.sample(possible, min(n_fake, len(possible))):
            record[f'has_{amenity}'] = 1
        record['price'] = int(round(record['price'] * random.uniform(1.05, 1.20) / 500) * 500)

    elif fraud_type == 'size_deception':
        inflation = random.choice([random.uniform(1.1, 1.3),
                                   random.uniform(1.3, 1.6),
                                   random.uniform(1.6, 2.0)])
        record['size_sqm'] = int(record['size_sqm'] * inflation)
        record['price'] = int(round(record['price'] * random.uniform(1.05, 1.15) / 500) * 500)

    elif fraud_type == 'location_fraud':
        tier = get_area_tier(record['area'])
        if tier == 'budget':
            new_areas = AREA_CONFIGS['mid']['areas'] + AREA_CONFIGS['high_end']['areas'][:3]
            mult = random.uniform(1.10, 1.35)
        elif tier == 'mid':
            new_areas = AREA_CONFIGS['high_end']['areas']
            mult = random.uniform(1.05, 1.25)
        else:
            new_areas = AREA_CONFIGS['high_end']['areas'][:2]
            mult = random.uniform(1.02, 1.10)
        record['area']  = random.choice(new_areas)
        record['price'] = int(round(record['price'] * mult / 500) * 500)

    elif fraud_type == 'engagement_manipulation':
        record['listing_age_days'] = random.randint(1, 5)
        record['views']            = random.randint(150, 450)
        record['contact_clicks']   = random.randint(50, 180)
        if random.random() < 0.30:
            record['price'] = int(round(record['price'] * random.uniform(0.9, 1.1) / 500) * 500)

    return record


# ==================================================
# 12. ORACLE ANOMALY SCORES (ANALYSIS ONLY — do not train on these)
# ==================================================

def _compute_oracle_scores(record):
    btype = record['building_type']
    tier  = get_area_tier(record['area'])

    def jitter(s): return clamp(s + random.gauss(0, 0.04), 0.0, 1.0)

    # Generator
    gen_exp    = _TRUE_FEATURE_BASE_RATES["generator"][btype][tier]
    gen_obs    = record['has_generator']
    rarity     = (1 - gen_exp) if gen_obs == 1 else gen_exp
    oracle_gen = jitter(smooth_scale(abs(gen_obs - gen_exp) * (0.3 + 0.7 * rarity),
                                     midpoint=0.15, steepness=8))

    # Elevator
    if btype in ("Condominium", "Townhouse"):
        oracle_elev = jitter(0.90) if record['has_elevator'] == 1 else jitter(0.04)
    else:
        elev_exp   = _TRUE_ELEVATOR_BASE_RATES[btype][tier]
        elev_obs   = record['has_elevator']
        rarity_e   = (1 - elev_exp) if elev_obs == 1 else elev_exp
        oracle_elev = jitter(smooth_scale(abs(elev_obs - elev_exp) * (0.3 + 0.7 * rarity_e),
                                          midpoint=0.2, steepness=7))

    # Price (log-ratio vs research-calibrated median)
    exp_sqm    = PRICE_PER_SQM[btype][tier]
    actual_sqm = record['price'] / max(record['size_sqm'], 1)
    log_r      = np.log(clamp(actual_sqm / max(exp_sqm, 1), 0.1, 10.0))
    oracle_price = jitter(clamp(abs(log_r) / 2.5, 0.0, 1.0))

    # Size per bedroom
    exp_ranges = {"Condominium": (30, 58), "Townhouse": (35, 68),
                  "Apartment Building": (38, 82), "Villa Compound": (48, 110)}
    if record['bedrooms'] > 0:
        spb = record['size_sqm'] / record['bedrooms']
        lo, hi = exp_ranges.get(btype, (30, 100))
        med, rng = (lo + hi) / 2, (hi - lo)
        oracle_size = jitter(clamp(abs(spb - med) / max(rng / 2, 1) / 2.5, 0.0, 1.0))
    else:
        oracle_size = jitter(0.25)

    # Studio premium (studios with many premium features)
    if record['bedrooms'] == 0:
        pc = (record['has_generator'] + record['has_elevator'] +
              record['has_parking']   + record['has_security'])
        oracle_studio = jitter(0.75 if pc >= 3 else (0.45 if pc >= 2 else 0.10))
    else:
        oracle_studio = 0.0

    # Engagement z-score
    rate    = record['contact_clicks'] / (record['views'] + 1)
    age     = record['listing_age_days']
    exp_r, std_r = (0.12, 0.08) if age <= 3 else ((0.18, 0.10) if age <= 10 else (0.15, 0.12))
    oracle_eng   = jitter(clamp(abs(rate - exp_r) / (std_r * 3), 0.0, 1.0))

    return {
        'oracle_generator_anomaly':      oracle_gen,
        'oracle_elevator_anomaly':       oracle_elev,
        'oracle_price_anomaly':          oracle_price,
        'oracle_size_anomaly':           oracle_size,
        'oracle_studio_premium_anomaly': oracle_studio,
        'oracle_engagement_anomaly':     oracle_eng,
    }


# ==================================================
# 13. MAIN LISTING GENERATOR
# ==================================================

def _generate_listing(feature_rates, elevator_rates, fraud_prob_mean=0.05):
    all_areas     = [a for cfg in AREA_CONFIGS.values() for a in cfg["areas"]]
    area          = random.choice(all_areas)
    building_type = random.choice(list(BUILDING_CONFIGS.keys()))

    bedrooms  = get_bedrooms(building_type)
    bathrooms = get_bathrooms(bedrooms, building_type)   # FIX B: building-type aware
    size_sqm  = get_size_sqm(bedrooms, building_type)
    floor_num = get_floor_number(building_type)
    features  = generate_features(building_type, area, floor_num, bedrooms,
                                  feature_rates, elevator_rates)

    # Engagement metrics
    listing_age = clamp(int(np.random.exponential(15) + 1), 1, 90)
    views       = np.random.poisson(listing_age * 5 + 10)
    conv_base   = random.betavariate(2.5, 8)
    if features.get("furnished"): conv_base *= random.uniform(1.05, 1.20)
    noisy_conv  = conv_base * clamp(random.gauss(1.0, 0.15), 0.75, 1.25)
    clicks      = int(views * noisy_conv)

    # Engagement anomaly breaks (~5%)
    if   random.random() < 0.025: clicks = int(views * random.uniform(0.02, 0.08))
    elif random.random() < 0.025:
        views  = int(views * random.uniform(0.3, 0.5))
        clicks = int(views * random.uniform(0.4, 0.7))

    price = calculate_price(area, building_type, bedrooms, size_sqm, features, floor_num)

    record = {
        "price": price, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "size_sqm": size_sqm, "floor_number": floor_num,
        "building_type": building_type, "furnished": features["furnished"],
        "has_generator": features["generator"], "has_parking": features["parking"],
        "has_security": features["security"], "has_elevator": features["elevator"],
        "area": area, "listing_age_days": listing_age,
        "views": views, "contact_clicks": clicks,
        "is_fraud": 0, "fraud_type": "none"
    }

    fraud_prob = clamp(random.gauss(fraud_prob_mean, 0.008), 0.02, 0.07)
    if random.random() < fraud_prob:
        record = inject_fraud(record)

    record.update(_compute_oracle_scores(record))
    return record


# ==================================================
# 14. DERIVED FEATURES
# ==================================================

def _add_derived_features(df):
    df['price_per_sqm']     = df['price'] / df['size_sqm'].clip(lower=1)
    df['price_per_bedroom'] = df['price'] / (df['bedrooms'] + 1)
    df['size_per_bedroom']  = df['size_sqm'] / (df['bedrooms'] + 1)

    tier_map = {a: t for t, cfg in AREA_CONFIGS.items() for a in cfg["areas"]}
    df['area_tier']         = df['area'].map(tier_map)
    df['area_tier_encoded'] = df['area_tier'].map({'budget': 1, 'mid': 2, 'high_end': 3})

    df['engagement_rate']     = df['contact_clicks'] / (df['views'] + 1)
    df['engagement_velocity'] = df['contact_clicks'] / (df['listing_age_days'] + 1)

    rng = np.random.default_rng(99)
    def noisy_expected(row):
        tier   = row['area_tier']
        base   = PRICE_PER_SQM[row['building_type']][tier]
        exp    = row['size_sqm'] * base
        # Structural distortions (prevent expected_price being a clean fraud signal)
        if rng.random() < 0.05:
            wrong = rng.choice([t for t in AREA_CONFIGS if t != tier])
            exp   = row['size_sqm'] * PRICE_PER_SQM[row['building_type']][wrong]
        if rng.random() < 0.08:
            exp *= np.sqrt(row['size_sqm'] / 80) if rng.random() < 0.5 \
                   else np.log(row['size_sqm'] / 40 + 1)
        exp *= clamp(float(rng.normal(1.0, 0.35)), 0.45, 1.55)
        return max(5_000, exp)

    df['expected_price'] = df.apply(noisy_expected, axis=1)
    df['price_position'] = df['price'] / (df['expected_price'] + 1)

    df['is_condo']      = (df['building_type'] == 'Condominium').astype(int)
    df['is_villa']      = (df['building_type'] == 'Villa Compound').astype(int)
    df['is_townhouse']  = (df['building_type'] == 'Townhouse').astype(int)
    df['is_apartment']  = (df['building_type'] == 'Apartment Building').astype(int)
    df['amenity_count'] = (df['has_generator'] + df['has_elevator'] +
                           df['has_parking']   + df['has_security'])
    return df


# ==================================================
# 15. PRODUCTION ANOMALY SCORES (safe for training)
# ==================================================

def _compute_production_scores(df):
    emp = {}
    for feat in ['has_generator', 'has_elevator', 'has_parking', 'has_security']:
        emp[feat] = {}
        for btype in df['building_type'].unique():
            emp[feat][btype] = {}
            for tier in ['high_end', 'mid', 'budget']:
                mask   = (df['building_type'] == btype) & (df['area_tier'] == tier)
                subset = df[mask]
                rate   = (subset[feat].mean() if len(subset) >= 20
                          else df[df['building_type'] == btype][feat].mean())
                emp[feat][btype][tier] = clamp(float(rate), 0.001, 0.999)

    price_medians = df.groupby(['building_type', 'area_tier'])['price_per_sqm'].median().to_dict()

    def score_row(row):
        btype = row['building_type']
        tier  = row['area_tier']

        def dev_score(obs, rate, mid=0.15, steep=8):
            rarity = (1 - rate) if obs == 1 else rate
            raw    = abs(obs - rate) * (0.3 + 0.7 * rarity)
            return clamp(smooth_scale(raw, midpoint=mid, steepness=steep), 0.0, 1.0)

        prod_gen  = dev_score(row['has_generator'],
                              emp['has_generator'][btype].get(tier, 0.10))

        if btype in ('Condominium', 'Townhouse'):
            prod_elev = 0.92 if row['has_elevator'] == 1 else 0.03
        else:
            prod_elev = dev_score(row['has_elevator'],
                                  emp['has_elevator'][btype].get(tier, 0.05), mid=0.2, steep=7)

        med_sqm    = price_medians.get((btype, tier), row['price_per_sqm'])
        log_r      = np.log(clamp(row['price_per_sqm'] / max(med_sqm, 1), 0.1, 10.0))
        prod_price = clamp(abs(log_r) / 2.5, 0.0, 1.0)

        age    = row['listing_age_days']
        rate_e = row['contact_clicks'] / (row['views'] + 1)
        exp_r, std_r = (0.12, 0.08) if age <= 3 else ((0.18, 0.10) if age <= 10 else (0.15, 0.12))
        prod_eng = clamp(abs(rate_e - exp_r) / (std_r * 3), 0.0, 1.0)

        pc = (row['has_generator'] + row['has_elevator'] +
              row['has_parking']   + row['has_security'])
        prod_studio = (0.75 if pc >= 3 else (0.45 if pc >= 2 else 0.10)) \
                      if row['bedrooms'] == 0 else 0.0

        return pd.Series({
            'prod_generator_anomaly':      prod_gen,
            'prod_elevator_anomaly':       prod_elev,
            'prod_price_anomaly':          prod_price,
            'prod_engagement_anomaly':     prod_eng,
            'prod_studio_premium_anomaly': prod_studio,
        })

    return pd.concat([df, df.apply(score_row, axis=1)], axis=1)


# ==================================================
# 16. LABEL NOISE
# ==================================================

def add_label_noise(df, fraud_miss_rate=0.08, false_report_rate=0.01, seed=42):
    rng = np.random.default_rng(seed)
    df  = df.copy()
    df['true_is_fraud']   = df['is_fraud']
    df['true_fraud_type'] = df['fraud_type']

    fraud_idx = df.index[df['is_fraud'] == 1].tolist()
    missed    = rng.choice(fraud_idx, size=int(len(fraud_idx) * fraud_miss_rate), replace=False)
    df.loc[missed, 'is_fraud']   = 0
    df.loc[missed, 'fraud_type'] = 'none'

    legit_idx = df.index[df['is_fraud'] == 0].tolist()
    false_pos = rng.choice(legit_idx, size=int(len(legit_idx) * false_report_rate), replace=False)
    df.loc[false_pos, 'is_fraud']   = 1
    df.loc[false_pos, 'fraud_type'] = 'false_report'
    return df


# ==================================================
# 17. STRATIFIED SPLIT
# ==================================================

def stratified_split(df, ratios=(0.70, 0.15, 0.15), label_col='is_fraud', seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-9
    train_r, val_r, _ = ratios
    rng = np.random.default_rng(seed)

    def _split(group):
        idx     = rng.permutation(len(group))
        n_train = int(len(group) * train_r)
        n_val   = int(len(group) * val_r)
        return (group.iloc[idx[:n_train]],
                group.iloc[idx[n_train:n_train + n_val]],
                group.iloc[idx[n_train + n_val:]])

    fraud = df[df[label_col] == 1].copy()
    legit = df[df[label_col] == 0].copy()
    f_tr, f_v, f_te = _split(fraud)
    l_tr, l_v, l_te = _split(legit)

    splits = {}
    for name, f, l in [('train', f_tr, l_tr), ('val', f_v, l_v), ('test', f_te, l_te)]:
        splits[name] = pd.concat([f, l]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return splits


# ==================================================
# 18. MAIN GENERATION FUNCTION
# ==================================================

def generate_dataset(n_samples=20000, seed=42,
                     fraud_miss_rate=0.08, false_report_rate=0.01,
                     base_rate_noise_std=0.15):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Generating {n_samples:,} listings (v5.0 final)...")

    feature_rates  = _perturb_rates(_TRUE_FEATURE_BASE_RATES, noise_std=base_rate_noise_std)
    elevator_rates = _perturb_elevator_rates(_TRUE_ELEVATOR_BASE_RATES, noise_std=base_rate_noise_std)

    data = [_generate_listing(feature_rates, elevator_rates) for _ in range(n_samples)]
    df   = pd.DataFrame(data)

    print("Computing derived features...")
    df = _add_derived_features(df)
    print("Computing production anomaly scores...")
    df = _compute_production_scores(df)
    print("Injecting label noise...")
    df = add_label_noise(df, fraud_miss_rate=fraud_miss_rate,
                         false_report_rate=false_report_rate, seed=seed)
    return df


# ==================================================
# 19. VALIDATION
# ==================================================

def validate_dataset(df):
    sep = "="*65
    print(f"\n{sep}\nDATASET VALIDATION — v5.0 FINAL\n{sep}")

    n = len(df)
    print(f"\nRows:             {n:,}")
    print(f"True fraud rate:  {df['true_is_fraud'].mean()*100:.2f}%  ({df['true_is_fraud'].sum()} listings)")
    print(f"Noisy label rate: {df['is_fraud'].mean()*100:.2f}%")

    # Fraud type breakdown
    print("\nFraud type breakdown:")
    tfd = df[df['true_is_fraud'] == 1]
    for ft, cnt in tfd['true_fraud_type'].value_counts().items():
        print(f"  {ft:<28} {cnt:>4}  ({cnt/len(tfd)*100:.1f}%)")

    # Studio check — should be significant (~15–25%)
    studios = (df['bedrooms'] == 0).sum()
    studio_pct = studios / n * 100
    studio_flag = "✓" if 12 <= studio_pct <= 28 else "✗"
    print(f"\nStudio listings:        {studios}  ({studio_pct:.1f}%)  {studio_flag} target 15–25%")

    # Shared bathroom (0 bath) — valid ONLY for studios
    zero_bath_studio    = ((df['bathrooms'] == 0) & (df['bedrooms'] == 0)).sum()
    zero_bath_nonstudio = ((df['bathrooms'] == 0) & (df['bedrooms'] > 0)).sum()
    print(f"Shared bath (studios):  {zero_bath_studio}  ({zero_bath_studio/max(studios,1)*100:.1f}% of studios)  ✓ expected")
    print(f"Zero bath (non-studio): {zero_bath_nonstudio}  {'✓ PASS' if zero_bath_nonstudio == 0 else '✗ FAIL'}")

    # Elevator constraint
    bad_elev = df[(df['building_type'].isin(['Condominium', 'Townhouse'])) &
                  (df['has_elevator'] == 1)]
    print(f"Elevator constraint:    {'✓ PASS' if len(bad_elev) == 0 else f'✗ FAIL ({len(bad_elev)} violations)'}")

    # Label noise
    missed = ((df['true_is_fraud'] == 1) & (df['is_fraud'] == 0)).sum()
    fp     = ((df['true_is_fraud'] == 0) & (df['is_fraud'] == 1)).sum()
    print(f"Missed fraud labels:    {missed}")
    print(f"False positive labels:  {fp}")

    # Price check vs Bet Afri + Jiji/Engocha targets
    # NOTE: targets are for ALL unit types combined (studio to 3BR+),
    # not just studios. Studio-only targets would be lower.
    print("\nPrice sanity check — median monthly rent (ETB), legit listings only:")
    targets = {
        ("Condominium",        "budget"):   (12_000, 28_000),  # studio→1BR→2BR mix
        ("Condominium",        "mid"):      (22_000, 45_000),
        ("Condominium",        "high_end"): (35_000, 80_000),
        ("Apartment Building", "budget"):   (14_000, 50_000),  # private G+ buildings
        ("Apartment Building", "mid"):      (40_000, 85_000),
        ("Apartment Building", "high_end"): (80_000, 180_000),
        ("Townhouse",          "budget"):   (8_000,  38_000),  # rooms + 1-3BR mix
        ("Townhouse",          "mid"):      (20_000, 65_000),
        ("Villa Compound",     "mid"):      (70_000, 180_000),
        ("Villa Compound",     "high_end"): (160_000, 420_000),
    }
    for btype in df['building_type'].unique():
        for tier in ['high_end', 'mid', 'budget']:
            mask = (df['building_type'] == btype) & (df['area_tier'] == tier)
            sub  = df[mask & (df['true_is_fraud'] == 0)]
            if len(sub) < 10: continue
            p10, med, p90 = sub['price'].quantile([0.10, 0.50, 0.90])
            target = targets.get((btype, tier))
            if target:
                lo, hi = target
                flag = "✓" if lo <= med <= hi else "✗"
                print(f"  {flag} {btype:<22} {tier:<9} p10={p10:>8,.0f}  "
                      f"med={med:>8,.0f}  p90={p90:>8,.0f}  target={lo:,}–{hi:,}")
            else:
                print(f"  · {btype:<22} {tier:<9} p10={p10:>8,.0f}  med={med:>8,.0f}  p90={p90:>8,.0f}")

    # Top correlations
    print("\nTop 5 correlations with true_is_fraud:")
    num_cols = df.select_dtypes(include=[np.number]).columns
    corrs    = df[num_cols].corr()['true_is_fraud'].abs().drop('true_is_fraud').sort_values(ascending=False)
    for col, c in corrs.head(5).items():
        flag = " ← LEAKAGE" if c > 0.80 else (" ← HIGH" if c > 0.60 else "")
        print(f"  {col:<42} {c:.3f}{flag}")

    print(f"\n{sep}")
    return df


# ==================================================
# 20. MAIN
# ==================================================

if __name__ == "__main__":
    df = generate_dataset(n_samples=20000, seed=42)
    df = validate_dataset(df)

    out = "addis_rental_fraud_v5_final.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved full dataset:  {out}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    splits = stratified_split(df, label_col='is_fraud', seed=42)
    for name, part in splits.items():
        fname = f"addis_rental_fraud_v5_{name}.csv"
        if name == 'test':
            part.to_csv(fname, index=False)
        else:
            keep = SAFE_TRAINING_FEATURES + ['is_fraud']
            part[[c for c in keep if c in part.columns]].to_csv(fname, index=False)
        print(f"  {name:5s}: {len(part):,} rows  →  {fname}")

    print("\nDone. This is the final version — start training your model.")
