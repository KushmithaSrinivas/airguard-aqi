"""
AirGuard - Real-time Air Quality Data Provider & Fallback Engine
Final-Year BCA Data Science Project
"""
import math
import time
import requests
from datetime import datetime, timedelta
from ml_engine import compute_official_aqi, get_aqi_category

# Curated catalog of 40+ Indian & Global cities with coordinates and realistic pollution baselines
CITIES_DB = {
    'delhi': {'name': 'Delhi', 'state': 'Delhi NCT', 'country': 'India', 'lat': 28.6139, 'lon': 77.2090, 'base_pm25': 145, 'base_pm10': 230, 'base_no2': 55, 'base_so2': 22, 'base_co': 2.4, 'base_o3': 48},
    'mumbai': {'name': 'Mumbai', 'state': 'Maharashtra', 'country': 'India', 'lat': 19.0760, 'lon': 72.8777, 'base_pm25': 82, 'base_pm10': 130, 'base_no2': 42, 'base_so2': 18, 'base_co': 1.6, 'base_o3': 38},
    'bengaluru': {'name': 'Bengaluru', 'state': 'Karnataka', 'country': 'India', 'lat': 12.9716, 'lon': 77.5946, 'base_pm25': 46, 'base_pm10': 78, 'base_no2': 28, 'base_so2': 12, 'base_co': 1.1, 'base_o3': 32},
    'kolkata': {'name': 'Kolkata', 'state': 'West Bengal', 'country': 'India', 'lat': 22.5726, 'lon': 88.3639, 'base_pm25': 118, 'base_pm10': 185, 'base_no2': 48, 'base_so2': 20, 'base_co': 2.0, 'base_o3': 42},
    'chennai': {'name': 'Chennai', 'state': 'Tamil Nadu', 'country': 'India', 'lat': 13.0827, 'lon': 80.2707, 'base_pm25': 52, 'base_pm10': 86, 'base_no2': 25, 'base_so2': 14, 'base_co': 1.2, 'base_o3': 35},
    'hyderabad': {'name': 'Hyderabad', 'state': 'Telangana', 'country': 'India', 'lat': 17.3850, 'lon': 78.4867, 'base_pm25': 74, 'base_pm10': 118, 'base_no2': 38, 'base_so2': 16, 'base_co': 1.5, 'base_o3': 39},
    'pune': {'name': 'Pune', 'state': 'Maharashtra', 'country': 'India', 'lat': 18.5204, 'lon': 73.8567, 'base_pm25': 68, 'base_pm10': 105, 'base_no2': 34, 'base_so2': 15, 'base_co': 1.3, 'base_o3': 36},
    'ahmedabad': {'name': 'Ahmedabad', 'state': 'Gujarat', 'country': 'India', 'lat': 23.0225, 'lon': 72.5714, 'base_pm25': 96, 'base_pm10': 152, 'base_no2': 44, 'base_so2': 24, 'base_co': 1.8, 'base_o3': 44},
    'lucknow': {'name': 'Lucknow', 'state': 'Uttar Pradesh', 'country': 'India', 'lat': 26.8467, 'lon': 80.9462, 'base_pm25': 132, 'base_pm10': 205, 'base_no2': 50, 'base_so2': 21, 'base_co': 2.2, 'base_o3': 45},
    'patna': {'name': 'Patna', 'state': 'Bihar', 'country': 'India', 'lat': 25.5941, 'lon': 85.1376, 'base_pm25': 155, 'base_pm10': 240, 'base_no2': 58, 'base_so2': 25, 'base_co': 2.6, 'base_o3': 50},
    'jaipur': {'name': 'Jaipur', 'state': 'Rajasthan', 'country': 'India', 'lat': 26.9124, 'lon': 75.7873, 'base_pm25': 102, 'base_pm10': 165, 'base_no2': 40, 'base_so2': 19, 'base_co': 1.7, 'base_o3': 41},
    'chandigarh': {'name': 'Chandigarh', 'state': 'Punjab/Haryana', 'country': 'India', 'lat': 30.7333, 'lon': 76.7794, 'base_pm25': 78, 'base_pm10': 125, 'base_no2': 32, 'base_so2': 14, 'base_co': 1.4, 'base_o3': 37},
    'varanasi': {'name': 'Varanasi', 'state': 'Uttar Pradesh', 'country': 'India', 'lat': 25.3176, 'lon': 82.9739, 'base_pm25': 128, 'base_pm10': 198, 'base_no2': 46, 'base_so2': 20, 'base_co': 2.1, 'base_o3': 43},
    'bhopal': {'name': 'Bhopal', 'state': 'Madhya Pradesh', 'country': 'India', 'lat': 23.2599, 'lon': 77.4126, 'base_pm25': 70, 'base_pm10': 110, 'base_no2': 30, 'base_so2': 13, 'base_co': 1.3, 'base_o3': 34},
    'indore': {'name': 'Indore', 'state': 'Madhya Pradesh', 'country': 'India', 'lat': 22.7196, 'lon': 75.8577, 'base_pm25': 65, 'base_pm10': 102, 'base_no2': 28, 'base_so2': 14, 'base_co': 1.2, 'base_o3': 33},
    'kanpur': {'name': 'Kanpur', 'state': 'Uttar Pradesh', 'country': 'India', 'lat': 26.4499, 'lon': 80.3319, 'base_pm25': 142, 'base_pm10': 220, 'base_no2': 52, 'base_so2': 23, 'base_co': 2.3, 'base_o3': 47},
    'agra': {'name': 'Agra', 'state': 'Uttar Pradesh', 'country': 'India', 'lat': 27.1767, 'lon': 78.0081, 'base_pm25': 120, 'base_pm10': 190, 'base_no2': 45, 'base_so2': 20, 'base_co': 2.0, 'base_o3': 44},
    'surat': {'name': 'Surat', 'state': 'Gujarat', 'country': 'India', 'lat': 21.1702, 'lon': 72.8311, 'base_pm25': 78, 'base_pm10': 122, 'base_no2': 36, 'base_so2': 20, 'base_co': 1.5, 'base_o3': 37},
    'visakhapatnam': {'name': 'Visakhapatnam', 'state': 'Andhra Pradesh', 'country': 'India', 'lat': 17.6868, 'lon': 83.2185, 'base_pm25': 54, 'base_pm10': 88, 'base_no2': 26, 'base_so2': 15, 'base_co': 1.1, 'base_o3': 34},
    'guwahati': {'name': 'Guwahati', 'state': 'Assam', 'country': 'India', 'lat': 26.1445, 'lon': 91.7362, 'base_pm25': 58, 'base_pm10': 92, 'base_no2': 24, 'base_so2': 11, 'base_co': 1.0, 'base_o3': 31},
    'kochi': {'name': 'Kochi', 'state': 'Kerala', 'country': 'India', 'lat': 9.9312, 'lon': 76.2673, 'base_pm25': 38, 'base_pm10': 62, 'base_no2': 20, 'base_so2': 9, 'base_co': 0.8, 'base_o3': 28},
    'srinagar': {'name': 'Srinagar', 'state': 'Jammu & Kashmir', 'country': 'India', 'lat': 34.0837, 'lon': 74.7973, 'base_pm25': 42, 'base_pm10': 68, 'base_no2': 18, 'base_so2': 8, 'base_co': 0.9, 'base_o3': 30},
    'amritsar': {'name': 'Amritsar', 'state': 'Punjab', 'country': 'India', 'lat': 31.6340, 'lon': 74.8723, 'base_pm25': 110, 'base_pm10': 170, 'base_no2': 42, 'base_so2': 18, 'base_co': 1.8, 'base_o3': 40},
    'dehradun': {'name': 'Dehradun', 'state': 'Uttarakhand', 'country': 'India', 'lat': 30.3165, 'lon': 78.0322, 'base_pm25': 48, 'base_pm10': 78, 'base_no2': 22, 'base_so2': 10, 'base_co': 1.0, 'base_o3': 32},
    'bhubaneswar': {'name': 'Bhubaneswar', 'state': 'Odisha', 'country': 'India', 'lat': 20.2961, 'lon': 85.8245, 'base_pm25': 62, 'base_pm10': 98, 'base_no2': 27, 'base_so2': 14, 'base_co': 1.2, 'base_o3': 33},
    'london': {'name': 'London', 'state': 'England', 'country': 'United Kingdom', 'lat': 51.5074, 'lon': -0.1278, 'base_pm25': 18, 'base_pm10': 32, 'base_no2': 22, 'base_so2': 6, 'base_co': 0.4, 'base_o3': 36},
    'new york': {'name': 'New York', 'state': 'NY', 'country': 'USA', 'lat': 40.7128, 'lon': -74.0060, 'base_pm25': 16, 'base_pm10': 28, 'base_no2': 20, 'base_so2': 5, 'base_co': 0.5, 'base_o3': 38},
    'tokyo': {'name': 'Tokyo', 'state': 'Kanto', 'country': 'Japan', 'lat': 35.6762, 'lon': 139.6503, 'base_pm25': 15, 'base_pm10': 26, 'base_no2': 18, 'base_so2': 4, 'base_co': 0.4, 'base_o3': 34},
    'dubai': {'name': 'Dubai', 'state': 'Dubai', 'country': 'UAE', 'lat': 25.2048, 'lon': 55.2708, 'base_pm25': 65, 'base_pm10': 150, 'base_no2': 32, 'base_so2': 16, 'base_co': 1.2, 'base_o3': 44},
    'singapore': {'name': 'Singapore', 'state': 'Singapore', 'country': 'Singapore', 'lat': 1.3521, 'lon': 103.8198, 'base_pm25': 22, 'base_pm10': 38, 'base_no2': 16, 'base_so2': 8, 'base_co': 0.6, 'base_o3': 26}
}

def get_diurnal_factor(hour_of_day):
    """Calculates diurnal cycle factor (morning/evening rush hour peaks)."""
    val = 1.0 + 0.35 * math.sin((hour_of_day - 4) * math.pi / 12) + 0.2 * math.sin((hour_of_day - 8) * math.pi / 6)
    return max(0.65, min(1.45, val))

def get_city_info(query):
    if not query:
        return CITIES_DB['delhi']
    query_clean = query.strip().lower()
    if query_clean in CITIES_DB:
        return CITIES_DB[query_clean]
    for key, val in CITIES_DB.items():
        if query_clean in key or query_clean in val['name'].lower() or query_clean in val['state'].lower():
            return val
    return CITIES_DB['delhi']

def fetch_live_aqi(city_name='Delhi'):
    city = get_city_info(city_name)
    lat, lon = city['lat'], city['lon']
    now = datetime.now()
    current_hour = now.hour

    api_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&"
        f"current=us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&"
        f"hourly=pm10,pm2_5,us_aqi&forecast_days=3"
    )

    data_source = 'Open-Meteo Live Global Air Quality API'
    pm25, pm10, no2, so2, co, o3 = None, None, None, None, None, None
    hourly_history = []
    hourly_forecast = []

    try:
        resp = requests.get(api_url, timeout=3.5)
        if resp.status_code == 200:
            api_data = resp.json()
            curr = api_data.get('current', {})
            pm25 = curr.get('pm2_5')
            pm10 = curr.get('pm10')
            no2 = curr.get('nitrogen_dioxide')
            so2 = curr.get('sulphur_dioxide')
            raw_co = curr.get('carbon_monoxide')
            co = (raw_co / 1000.0) if raw_co is not None else None
            o3 = curr.get('ozone')

            hourly = api_data.get('hourly', {})
            times = hourly.get('time', [])
            h_pm25 = hourly.get('pm2_5', [])
            h_pm10 = hourly.get('pm10', [])

            curr_iso = now.strftime('%Y-%m-%dT%H:00')
            matched_idx = 0
            for i, t in enumerate(times):
                if t >= curr_iso:
                    matched_idx = i
                    break

            start_idx = max(0, matched_idx - 12)
            for j in range(start_idx, min(len(times), start_idx + 24)):
                t_str = times[j]
                dt = datetime.fromisoformat(t_str)
                time_label = dt.strftime('%I %p')
                p25_val = round(h_pm25[j], 1) if j < len(h_pm25) and h_pm25[j] is not None else city['base_pm25']
                p10_val = round(h_pm10[j], 1) if j < len(h_pm10) and h_pm10[j] is not None else city['base_pm10']
                est_aqi, _, _ = compute_official_aqi(p25_val, p10_val)
                item = {
                    'time': time_label,
                    'is_past': j < matched_idx,
                    'is_current': j == matched_idx,
                    'pm25': p25_val,
                    'pm10': p10_val,
                    'aqi': est_aqi
                }
                if j <= matched_idx:
                    hourly_history.append(item)
                else:
                    hourly_forecast.append(item)
    except Exception:
        data_source = 'AirGuard Atmospheric Model (Offline Fallback)'

    # Diurnal fallback if API values unavailable
    if pm25 is None or pm10 is None:
        data_source = 'AirGuard Atmospheric Model (Offline Fallback)'
        diurnal = get_diurnal_factor(current_hour)
        pm25 = round(city['base_pm25'] * diurnal, 1)
        pm10 = round(city['base_pm10'] * diurnal, 1)
        no2 = round(city['base_no2'] * diurnal, 1)
        so2 = round(city['base_so2'] * diurnal * 0.9, 1)
        co = round(city['base_co'] * diurnal, 2)
        o3 = round(city['base_o3'] * (1.6 - diurnal * 0.5), 1)

        hourly_history = []
        hourly_forecast = []
        for offset in range(-12, 13):
            target_time = now + timedelta(hours=offset)
            h_val = target_time.hour
            d_fact = get_diurnal_factor(h_val)
            h_pm25 = round(city['base_pm25'] * d_fact, 1)
            h_pm10 = round(city['base_pm10'] * d_fact, 1)
            h_aqi, _, _ = compute_official_aqi(h_pm25, h_pm10)
            item = {
                'time': target_time.strftime('%I %p'),
                'is_past': offset < 0,
                'is_current': offset == 0,
                'pm25': h_pm25,
                'pm10': h_pm10,
                'aqi': h_aqi
            }
            if offset <= 0:
                hourly_history.append(item)
            else:
                hourly_forecast.append(item)

    if no2 is None: no2 = round(city['base_no2'], 1)
    if so2 is None: so2 = round(city['base_so2'], 1)
    if co is None: co = round(city['base_co'], 2)
    if o3 is None: o3 = round(city['base_o3'], 1)

    aqi, dominant, sub_indices = compute_official_aqi(pm25, pm10, no2, so2, co, o3)
    category = get_aqi_category(aqi)

    pollutants = {
        'pm25': {
            'code': 'PM2.5',
            'name': 'Fine Particulate Matter (<2.5 µm)',
            'val': round(pm25, 1),
            'unit': 'µg/m³',
            'who_limit': 15.0,
            'ratio': round(pm25 / 15.0, 1),
            'subindex': sub_indices.get('PM2.5', 0),
            'desc': 'Microscopic airborne particles from engine exhaust, smoke, and industrial burning that penetrate deep into alveolar lung tissues.'
        },
        'pm10': {
            'code': 'PM10',
            'name': 'Inhalable Particulate Matter (<10 µm)',
            'val': round(pm10, 1),
            'unit': 'µg/m³',
            'who_limit': 45.0,
            'ratio': round(pm10 / 45.0, 1),
            'subindex': sub_indices.get('PM10', 0),
            'desc': 'Coarse dust, pollen, and debris from construction, traffic, and dry roads causing eye and throat irritation.'
        },
        'no2': {
            'code': 'NO₂',
            'name': 'Nitrogen Dioxide',
            'val': round(no2, 1),
            'unit': 'µg/m³',
            'who_limit': 25.0,
            'ratio': round(no2 / 25.0, 1),
            'subindex': sub_indices.get('NO2', 0),
            'desc': 'Pungent reddish-brown gas from vehicle exhausts and thermal stations; triggers inflammation of the airways.'
        },
        'so2': {
            'code': 'SO₂',
            'name': 'Sulphur Dioxide',
            'val': round(so2, 1),
            'unit': 'µg/m³',
            'who_limit': 40.0,
            'ratio': round(so2 / 40.0, 1),
            'subindex': sub_indices.get('SO2', 0),
            'desc': 'Acidic gas produced by coal combustion, diesel engines, and smelters; causes coughing and wheezing.'
        },
        'co': {
            'code': 'CO',
            'name': 'Carbon Monoxide',
            'val': round(co, 2),
            'unit': 'mg/m³',
            'who_limit': 4.0,
            'ratio': round(co / 4.0, 1),
            'subindex': sub_indices.get('CO', 0),
            'desc': 'Invisible, odorless toxic gas from incomplete engine combustion; restricts oxygen transport in the bloodstream.'
        },
        'o3': {
            'code': 'O₃',
            'name': 'Surface Ozone',
            'val': round(o3, 1),
            'unit': 'µg/m³',
            'who_limit': 100.0,
            'ratio': round(o3 / 100.0, 1),
            'subindex': sub_indices.get('O3', 0),
            'desc': 'Ground-level pollutant created when sunlight bakes vehicular NOx and volatile organic vapors on sunny afternoons.'
        }
    }

    return {
        'city': city['name'],
        'state': city['state'],
        'country': city['country'],
        'lat': city['lat'],
        'lon': city['lon'],
        'timestamp': now.strftime('%d %b %Y, %I:%M %p'),
        'data_source': data_source,
        'aqi': aqi,
        'category': category,
        'dominant_pollutant': dominant,
        'pollutants': pollutants,
        'hourly_history': hourly_history,
        'hourly_forecast': hourly_forecast,
        'guidance': {
            'exercise': category['exercise'],
            'ventilation': category['ventilation'],
            'mask': category['mask'],
            'purifier': category['purifier']
        }
    }

def get_all_cities_for_map():
    results = []
    now = datetime.now()
    hour = now.hour
    diurnal = get_diurnal_factor(hour)

    for key, city in CITIES_DB.items():
        pm25 = round(city['base_pm25'] * diurnal, 1)
        pm10 = round(city['base_pm10'] * diurnal, 1)
        aqi, dominant, _ = compute_official_aqi(pm25, pm10)
        cat = get_aqi_category(aqi)
        results.append({
            'name': city['name'],
            'state': city['state'],
            'country': city['country'],
            'lat': city['lat'],
            'lon': city['lon'],
            'aqi': aqi,
            'pm25': pm25,
            'pm10': pm10,
            'dominant': dominant,
            'level': cat['level'],
            'color': cat['color'],
            'summary': cat['summary']
        })
    return results

def get_7day_forecast(city_name='Delhi'):
    city = get_city_info(city_name)
    now = datetime.now()
    forecast = []
    variances = [0.0, 0.05, -0.12, 0.08, -0.04, 0.15, -0.08]
    day_names = ['Today', 'Tomorrow'] + [(now + timedelta(days=i)).strftime('%A') for i in range(2, 7)]

    for i in range(7):
        var = variances[i]
        target_date = now + timedelta(days=i)
        p25 = round(city['base_pm25'] * (1.0 + var), 1)
        p10 = round(city['base_pm10'] * (1.0 + var), 1)
        aqi, dominant, _ = compute_official_aqi(p25, p10)
        cat = get_aqi_category(aqi)
        forecast.append({
            'day': day_names[i],
            'date': target_date.strftime('%b %d'),
            'aqi': aqi,
            'level': cat['level'],
            'color': cat['color'],
            'pm25': p25,
            'pm10': p10,
            'dominant': dominant,
            'summary': cat['summary']
        })
    return forecast
