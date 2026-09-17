"""
AirGuard - AQI Machine Learning & Atmospheric Data Science Engine
Final-Year BCA Data Science Project
Author: AirGuard Team
Description:
    Computes CPCB-standard Sub-index AQI and trains a scikit-learn
    Random Forest Regressor to predict Air Quality Index based on
    ambient concentrations of PM2.5, PM10, NO2, SO2, CO, and O3.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'aqi_rf_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

# CPCB (Central Pollution Control Board, India) Sub-Index Breakpoints Table
# Format: (C_low, C_high, I_low, I_high)
CPCB_BREAKPOINTS = {
    'pm25': [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500)
    ],
    'pm10': [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500)
    ],
    'no2': [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 600, 401, 500)
    ],
    'so2': [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500)
    ],
    'co': [
        (0.0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500)
    ],
    'o3': [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500)
    ]
}

def compute_linear_subindex(conc, pollutant_key):
    """
    Computes sub-index for a specific pollutant using CPCB linear formula:
    I = I_low + ((I_high - I_low) / (B_high - B_low)) * (conc - B_low)
    """
    if conc is None or conc < 0:
        return 0.0

    breakpoints = CPCB_BREAKPOINTS.get(pollutant_key, [])
    for b_low, b_high, i_low, i_high in breakpoints:
        if b_low <= conc <= b_high:
            if b_high == b_low:
                return float(i_low)
            return round(i_low + ((i_high - i_low) / (b_high - b_low)) * (conc - b_low), 1)

    if conc > breakpoints[-1][1]:
        return 500.0
    return 0.0

def compute_official_aqi(pm25, pm10, no2=20, so2=15, co=1.0, o3=30):
    """
    Computes overall CPCB AQI (maximum of individual sub-indices).
    """
    sub_indices = {
        'PM2.5': compute_linear_subindex(pm25, 'pm25'),
        'PM10': compute_linear_subindex(pm10, 'pm10'),
        'NO2': compute_linear_subindex(no2, 'no2'),
        'SO2': compute_linear_subindex(so2, 'so2'),
        'CO': compute_linear_subindex(co, 'co'),
        'O3': compute_linear_subindex(o3, 'o3')
    }
    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    max_aqi = round(sub_indices[dominant_pollutant])
    return max_aqi, dominant_pollutant, sub_indices

def get_aqi_category(aqi_val):
    """
    Returns citizen-friendly category, badge colors, and clear health recommendations.
    """
    if aqi_val <= 50:
        return {
            'level': 'Good',
            'range': '0 - 50',
            'color': '#10B981',
            'bg_class': 'bg-emerald-500',
            'text_class': 'text-emerald-700 dark:text-emerald-400',
            'badge_class': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
            'border_class': 'border-emerald-500',
            'summary': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'citizen_action': 'Enjoy your normal outdoor activities and open windows for fresh ventilation.',
            'exercise': 'Safe for all outdoor activities & jogging',
            'ventilation': 'Open windows freely for natural air',
            'mask': 'No mask needed',
            'purifier': 'Not needed'
        }
    elif aqi_val <= 100:
        return {
            'level': 'Satisfactory',
            'range': '51 - 100',
            'color': '#84CC16',
            'bg_class': 'bg-lime-500',
            'text_class': 'text-lime-700 dark:text-lime-400',
            'badge_class': 'bg-lime-100 text-lime-800 dark:bg-lime-900/40 dark:text-lime-300',
            'border_class': 'border-lime-500',
            'summary': 'Air quality is acceptable. Minor breathing discomfort for unusually sensitive people.',
            'citizen_action': 'Safe for normal outdoor exercise. Sensitive individuals should monitor for throat irritation.',
            'exercise': 'Safe for most citizens',
            'ventilation': 'Good to ventilate during daytime',
            'mask': 'Optional for unusually sensitive people',
            'purifier': 'Optional'
        }
    elif aqi_val <= 200:
        return {
            'level': 'Moderate',
            'range': '101 - 200',
            'color': '#F59E0B',
            'bg_class': 'bg-amber-500',
            'text_class': 'text-amber-700 dark:text-amber-400',
            'badge_class': 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
            'border_class': 'border-amber-500',
            'summary': 'May cause breathing discomfort to children, elderly, and people with lungs, asthma, or heart conditions.',
            'citizen_action': 'Children and seniors should reduce prolonged outdoor exertion. Sensitive groups keep inhalers ready.',
            'exercise': 'Reduce intense outdoor workouts',
            'ventilation': 'Close windows during morning/evening rush hours',
            'mask': 'Recommended near busy roads',
            'purifier': 'Recommended for sensitive individuals'
        }
    elif aqi_val <= 300:
        return {
            'level': 'Poor',
            'range': '201 - 300',
            'color': '#F97316',
            'bg_class': 'bg-orange-500',
            'text_class': 'text-orange-700 dark:text-orange-400',
            'badge_class': 'bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300',
            'border_class': 'border-orange-500',
            'summary': 'Causes breathing discomfort to most people on prolonged exposure. May trigger acute respiratory symptoms.',
            'citizen_action': 'Wear an N95 mask outdoors. Avoid morning runs. Keep windows closed during peak traffic hours.',
            'exercise': 'Avoid outdoor jogging and sports',
            'ventilation': 'Keep windows closed',
            'mask': 'N95 / N99 mask strongly advised outdoors',
            'purifier': 'Run air purifiers indoors'
        }
    elif aqi_val <= 400:
        return {
            'level': 'Very Poor',
            'range': '301 - 400',
            'color': '#EF4444',
            'bg_class': 'bg-red-500',
            'text_class': 'text-red-700 dark:text-red-400',
            'badge_class': 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
            'border_class': 'border-red-500',
            'summary': 'Causes respiratory illness to people on prolonged exposure. Pronounced effect on healthy individuals.',
            'citizen_action': 'Strictly avoid outdoor exercise. Children, pregnant women, and elderly must remain indoors. Use air purifiers.',
            'exercise': 'Strictly avoid outdoor activities',
            'ventilation': 'Seal windows and exterior vents',
            'mask': 'N95 mask mandatory outdoors',
            'purifier': 'Run air purifier continuously'
        }
    else:
        return {
            'level': 'Severe',
            'range': '401 - 500+',
            'color': '#7F1D1D',
            'bg_class': 'bg-red-900',
            'text_class': 'text-red-900 dark:text-red-300',
            'badge_class': 'bg-red-200 text-red-950 dark:bg-red-950 dark:text-red-200 border border-red-500',
            'border_class': 'border-red-900',
            'summary': 'Health Emergency! Affects healthy people and seriously impacts those with existing medical conditions.',
            'citizen_action': 'Health emergency! Stay completely indoors. Seal doors and windows. Wear high-filtration N95/N99 masks if transit is unavoidable.',
            'exercise': 'Complete ban on outdoor exertion',
            'ventilation': 'Never open windows; seal gaps',
            'mask': 'High-filtration N95/N99 mandatory',
            'purifier': 'Maximum HEPA filtration needed'
        }

def generate_atmospheric_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Clean air regime ~ 30%
    n_clean = int(n_samples * 0.30)
    pm25_clean = np.random.gamma(shape=3.0, scale=8.0, size=n_clean)
    pm10_clean = pm25_clean * np.random.uniform(1.4, 2.2, size=n_clean)
    no2_clean = np.random.uniform(5, 35, size=n_clean)
    so2_clean = np.random.uniform(2, 25, size=n_clean)
    co_clean = np.random.uniform(0.2, 0.9, size=n_clean)
    o3_clean = np.random.uniform(10, 45, size=n_clean)

    # 2. Moderate urban regime ~ 45%
    n_mod = int(n_samples * 0.45)
    pm25_mod = np.random.normal(loc=75.0, scale=25.0, size=n_mod).clip(30, 150)
    pm10_mod = pm25_mod * np.random.uniform(1.6, 2.5, size=n_mod)
    no2_mod = np.random.normal(loc=60.0, scale=20.0, size=n_mod).clip(20, 120)
    so2_mod = np.random.normal(loc=35.0, scale=15.0, size=n_mod).clip(10, 80)
    co_mod = np.random.normal(loc=1.8, scale=0.8, size=n_mod).clip(0.6, 3.5)
    o3_mod = np.random.normal(loc=65.0, scale=25.0, size=n_mod).clip(20, 140)

    # 3. High pollution / winter smog / industrial regime ~ 25%
    n_severe = n_samples - n_clean - n_mod
    pm25_severe = np.random.uniform(140, 450, size=n_severe)
    pm10_severe = pm25_severe * np.random.uniform(1.5, 2.8, size=n_severe)
    no2_severe = np.random.uniform(90, 240, size=n_severe)
    so2_severe = np.random.uniform(50, 160, size=n_severe)
    co_severe = np.random.uniform(2.5, 12.0, size=n_severe)
    o3_severe = np.random.uniform(40, 180, size=n_severe)

    pm25_all = np.concatenate([pm25_clean, pm25_mod, pm25_severe])
    pm10_all = np.concatenate([pm10_clean, pm10_mod, pm10_severe])
    no2_all = np.concatenate([no2_clean, no2_mod, no2_severe])
    so2_all = np.concatenate([so2_clean, so2_mod, so2_severe])
    co_all = np.concatenate([co_clean, co_mod, co_severe])
    o3_all = np.concatenate([o3_clean, o3_mod, o3_severe])

    aqi_list = []
    dominant_list = []
    for i in range(len(pm25_all)):
        aqi, dominant, _ = compute_official_aqi(
            pm25_all[i], pm10_all[i], no2_all[i], so2_all[i], co_all[i], o3_all[i]
        )
        aqi_list.append(aqi)
        dominant_list.append(dominant)

    df = pd.DataFrame({
        'pm25': np.round(pm25_all, 1),
        'pm10': np.round(pm10_all, 1),
        'no2': np.round(no2_all, 1),
        'so2': np.round(so2_all, 1),
        'co': np.round(co_all, 2),
        'o3': np.round(o3_all, 1),
        'aqi': aqi_list,
        'dominant_pollutant': dominant_list
    })
    return df

def train_aqi_ml_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("-> Generating atmospheric training dataset...")
    df = generate_atmospheric_dataset(n_samples=5000, random_state=42)
    features = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
    X = df[features]
    y = df['aqi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("-> Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    importances = rf_model.feature_importances_
    feat_importance_dict = {feat: round(float(imp) * 100, 2) for feat, imp in zip(features, importances)}

    joblib.dump(rf_model, MODEL_PATH)
    print(f"-> Model saved to {MODEL_PATH}")

    metrics = {
        'model_name': 'Random Forest Regressor (Ensemble of 100 Trees)',
        'dataset_size': len(df),
        'training_samples': len(X_train),
        'testing_samples': len(X_test),
        'r2_score': round(r2, 4),
        'accuracy_percentage': round(r2 * 100, 2),
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'feature_importances': feat_importance_dict,
        'features': features,
        'training_timestamp': pd.Timestamp.now().isoformat()
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"-> Metrics: R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    return metrics

_LOADED_MODEL = None

def get_ml_model():
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        if not os.path.exists(MODEL_PATH):
            train_aqi_ml_models()
        _LOADED_MODEL = joblib.load(MODEL_PATH)
    return _LOADED_MODEL

def predict_aqi(pm25, pm10, no2=20.0, so2=15.0, co=1.0, o3=30.0):
    model = get_ml_model()
    input_data = np.array([[float(pm25), float(pm10), float(no2), float(so2), float(co), float(o3)]])
    ml_pred = float(model.predict(input_data)[0])
    ml_pred = max(0, min(500, round(ml_pred)))
    cpcb_aqi, dominant, sub_indices = compute_official_aqi(pm25, pm10, no2, so2, co, o3)
    category_info = get_aqi_category(ml_pred)
    return {
        'predicted_aqi': ml_pred,
        'official_cpcb_aqi': cpcb_aqi,
        'dominant_pollutant': dominant,
        'sub_indices': sub_indices,
        'category': category_info,
        'inputs': {'pm25': pm25, 'pm10': pm10, 'no2': no2, 'so2': so2, 'co': co, 'o3': o3}
    }

def get_model_metrics():
    if not os.path.exists(METRICS_PATH):
        return train_aqi_ml_models()
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    train_aqi_ml_models()
    test_sample = predict_aqi(pm25=120, pm10=220, no2=65, so2=25, co=2.1, o3=45)
    print("Test Prediction AQI:", test_sample['predicted_aqi'], test_sample['category']['level'])

