"""
AirGuard - AQI Awareness & Citizen Safety System
Final-Year BCA Data Science Project
Main Flask Application Server
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from data_provider import CITIES_DB, fetch_live_aqi, get_all_cities_for_map, get_7day_forecast, get_city_info
from ml_engine import predict_aqi, get_model_metrics, get_aqi_category, compute_official_aqi

app = Flask(__name__)
app.config['SECRET_KEY'] = 'airguard-bca-datascience-2026'

# -----------------------------------------------------------------------------
# Web Page Routes
# -----------------------------------------------------------------------------
@app.route('/')
def home():
    featured_cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata', 'Chennai', 'Hyderabad']
    city_cards = []
    for c_name in featured_cities:
        try:
            info = fetch_live_aqi(c_name)
            city_cards.append({
                'name': info['city'],
                'state': info['state'],
                'aqi': info['aqi'],
                'level': info['category']['level'],
                'color': info['category']['color'],
                'badge_class': info['category']['badge_class'],
                'dominant': info['dominant_pollutant'],
                'summary': info['category']['summary'],
                'action': info['category']['citizen_action']
            })
        except Exception:
            pass
    return render_template('index.html', city_cards=city_cards)


@app.route('/dashboard')
def dashboard():
    city_query = request.args.get('city', 'Delhi')
    data = fetch_live_aqi(city_query)
    popular_cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata', 'Chennai', 'Hyderabad', 'Pune', 'Lucknow', 'Ahmedabad', 'Patna']
    return render_template('dashboard.html', data=data, current_city=data['city'], popular_cities=popular_cities)


@app.route('/categories')
def categories():
    return render_template('categories.html')


@app.route('/safety')
def safety():
    return render_template('safety.html')


@app.route('/map')
def aqi_map():
    return render_template('map.html')


@app.route('/analytics')
def analytics():
    city_query = request.args.get('city', 'Delhi')
    forecast = get_7day_forecast(city_query)
    metrics = get_model_metrics()
    return render_template('analytics.html', current_city=city_query, forecast=forecast, metrics=metrics)


@app.route('/risk-calculator')
def risk_calculator():
    return render_template('risk_calculator.html')


@app.route('/advisories')
def advisories():
    city_query = request.args.get('city', 'Delhi')
    data = fetch_live_aqi(city_query)
    return render_template('advisories.html', data=data)


@app.route('/about')
def about():
    metrics = get_model_metrics()
    return render_template('about.html', metrics=metrics)


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.route('/api/cities')
def api_cities():
    cities_list = [
        {'id': k, 'name': v['name'], 'state': v['state'], 'country': v['country']}
        for k, v in CITIES_DB.items()
    ]
    return jsonify(cities_list)


@app.route('/api/aqi')
def api_aqi():
    city = request.args.get('city', 'Delhi')
    data = fetch_live_aqi(city)
    return jsonify(data)


@app.route('/api/map-data')
def api_map_data():
    data = get_all_cities_for_map()
    return jsonify(data)


@app.route('/api/forecast')
def api_forecast():
    city = request.args.get('city', 'Delhi')
    forecast = get_7day_forecast(city)
    return jsonify(forecast)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    pm25 = float(payload.get('pm25', 50))
    pm10 = float(payload.get('pm10', 90))
    no2 = float(payload.get('no2', 25))
    so2 = float(payload.get('so2', 15))
    co = float(payload.get('co', 1.0))
    o3 = float(payload.get('o3', 35))

    result = predict_aqi(pm25, pm10, no2, so2, co, o3)
    return jsonify(result)


@app.route('/api/risk-score', methods=['POST'])
def api_risk_score():
    """
    Computes a personalized citizen health risk assessment score (0 - 100)
    based on demographic vulnerability, preexisting condition, exposure duration,
    and current ambient AQI.
    """
    payload = request.get_json(force=True, silent=True) or {}
    aqi = float(payload.get('aqi', 150))
    age_group = payload.get('age_group', 'adult')  # child, adult, senior
    condition = payload.get('condition', 'none')    # none, asthma, copd, heart, pregnancy
    duration_hrs = float(payload.get('duration_hrs', 1.0))
    activity = payload.get('activity', 'moderate')  # rest, light, heavy

    # Base risk factor from AQI (0 to 50 scale)
    base_risk = min(50.0, (aqi / 500.0) * 50.0)

    # Demographic multipliers
    age_multipliers = {'child': 1.35, 'adult': 1.0, 'senior': 1.45}
    condition_multipliers = {
        'none': 1.0,
        'asthma': 1.6,
        'copd': 1.7,
        'heart': 1.65,
        'pregnancy': 1.4
    }
    activity_multipliers = {'rest': 0.8, 'light': 1.0, 'moderate': 1.25, 'heavy': 1.55}

    age_factor = age_multipliers.get(age_group, 1.0)
    cond_factor = condition_multipliers.get(condition, 1.0)
    act_factor = activity_multipliers.get(activity, 1.0)
    time_factor = min(1.8, 1.0 + (duration_hrs - 1.0) * 0.15) if duration_hrs > 1.0 else 0.9

    total_risk_score = round(min(100.0, base_risk * age_factor * cond_factor * act_factor * time_factor), 1)

    if total_risk_score <= 25:
        risk_level = 'Low Risk'
        badge = 'bg-emerald-100 text-emerald-800'
        color = '#10B981'
        advice = 'Normal daily routines are safe. Maintain basic health hydration.'
        mask_need = 'Not needed'
    elif total_risk_score <= 50:
        risk_level = 'Moderate Risk'
        badge = 'bg-amber-100 text-amber-800'
        color = '#F59E0B'
        advice = 'Reduce prolonged intense physical exertion outdoors. Take regular indoor breathers.'
        mask_need = 'Optional for sensitive individuals'
    elif total_risk_score <= 75:
        risk_level = 'High Risk'
        badge = 'bg-orange-100 text-orange-800'
        color = '#F97316'
        advice = 'Avoid all non-essential outdoor travel. Keep rescue inhalers / prescribed medicines accessible.'
        mask_need = 'N95 Respirator strongly advised'
    else:
        risk_level = 'Critical Risk'
        badge = 'bg-red-100 text-red-800'
        color = '#DC2626'
        advice = 'Danger to health! Remain strictly indoors with HEPA air purifier active. Consult a doctor if chest discomfort occurs.'
        mask_need = 'N95 / N99 mask mandatory outdoors'

    return jsonify({
        'risk_score': total_risk_score,
        'risk_level': risk_level,
        'badge': badge,
        'color': color,
        'advice': advice,
        'mask_need': mask_need,
        'breakdown': {
            'aqi_component': round(base_risk, 1),
            'vulnerability_factor': round(age_factor * cond_factor, 2),
            'activity_exposure_factor': round(act_factor * time_factor, 2)
        }
    })


@app.route('/api/model-metrics')
def api_model_metrics():
    metrics = get_model_metrics()
    return jsonify(metrics)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"-> AirGuard System launching on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
