# AirGuard – AQI Awareness & Citizen Safety System
> **“Know Your Air. Protect Your Health.”**  
> **Final-Year BCA Data Science Project | 2026**

---

## 1. Project Overview

**AirGuard** is a full-stack, machine-learning-powered environmental awareness and citizen protection web platform. It is engineered to bridge the gap between complex atmospheric chemistry metrics ($\mu g/m^3$, $ppm$) and actionable public health defense for ordinary citizens.

Rather than just displaying raw index numbers, **AirGuard** clearly explains:
- Whether ambient air is safe, acceptable, or hazardous
- Plain-language health warnings without confusing scientific jargon
- Specific precautions for children, parents, elderly seniors, outdoor workers, and asthmatic patients
- Actionable advice on outdoor exercise, home ventilation hours, air purifier necessity, and N95 masks
- 7-day predictive atmospheric trends and an interactive Machine Learning simulator

---

## 2. Key Pages & Features

1. **Home (`/`)**:
   - Hero banner with environmental gradient styling and clear purpose statement
   - Instant city search with autocomplete and GPS geolocation support
   - Featured live AQI status card with visual speedometer gauge and plain-English safety actions
   - 4 quick navigation cards (*Check AQI*, *Safety Guide*, *AQI Map*, *Pollution Trends*)
   - The 6 AQI Categories summary strip

2. **AQI Dashboard (`/dashboard`)**:
   - Real-time air quality metrics for any selected city
   - Visual circular AQI gauge with color-coded classification badge
   - 6 individual pollutant cards (**PM2.5, PM10, NO₂, SO₂, CO, O₃**) with sub-indices, descriptions, and WHO safe threshold ratio bars
   - 4 daily guidance indicator tiles (Outdoor Exercise, Window Ventilation, Mask Requirement, Air Purifier)
   - 24-hour interactive historical & hourly projection timeline line chart (Chart.js)

3. **AQI Categories Guide (`/categories`)**:
   - Detailed visual breakdown of the 6 standard Central Pollution Control Board (CPCB) categories:
     - **Good (0–50)**: Minimal impact, open windows freely
     - **Satisfactory (51–100)**: Minor discomfort for unusually sensitive individuals
     - **Moderate (101–200)**: Breathing discomfort to children, elderly, and lung patients
     - **Poor (201–300)**: Unhealthy for all; N95 masks advised outdoors
     - **Very Poor (301–400)**: Hazardous; cancel outdoor sports, keep vulnerable groups inside
     - **Severe (401–500+)**: Public health emergency; seal windows, run HEPA air cleaners

4. **Citizen Safety Guide (`/safety`)**:
   - Segmented health directives for:
     - **General Public** (commute habits, hydration, ventilation windows)
     - **Children & Parents** (school sports rules, bus transit, pediatric masks)
     - **Elderly People** (cardiovascular safeguards, blood pressure monitoring)
     - **Outdoor Workers & Athletes** (shift rotation, exhalation-valved N95s)
     - **Asthma & Respiratory Patients** (inhaler readiness, saline steam, red-flag symptoms)
     - **Housing Societies & Schools** (anti-burning rules, water spraying)
   - **Mask Selection & Efficacy Guide**: Comprehensive comparison table (N95 vs N99 vs Surgical vs Cloth)
   - **Indoor Air Sanctuary**: NASA Clean Air Study plants (Snake Plant, Spider Plant, Areca Palm) & True HEPA purifiers

5. **Interactive AQI Map (`/map`)**:
   - Fullscreen Leaflet.js interactive map with color-coded pins across 35+ Indian and global cities
   - Filter by status (*All Cities*, *Very Poor / Severe*, *Moderate / Poor*, *Good / Satisfactory*)
   - Clicking a pin opens a popup with current AQI, dominant pollutant, and 1-click dashboard access

6. **ML Analytics & Forecast Hub (`/analytics`)**:
   - 7-day atmospheric forecast projection bar chart
   - **Interactive ML Prediction Simulator**: Adjust sliders for PM2.5, PM10, NO₂, SO₂, CO, O₃ to watch the scikit-learn Random Forest model predict AQI in real time
   - Model Evaluation Dashboard: $R^2$ Score, MAE, RMSE, and feature importance bar chart

7. **Personal Health Risk Calculator (`/risk-calculator`)**:
   - Multi-factor risk engine taking into account ambient AQI, age group, pre-existing illnesses (asthma, heart condition, COPD, pregnancy), and planned activity duration
   - Outputs a personal Risk Score (0–100), risk badge, maximum safe outdoor minutes, and printable advisory

8. **Emergency Advisory Generator (`/advisories`)**:
   - Formal notice generator for school principals, RWA housing society managers, and corporate HR
   - Formatted for clean single-page printing / PDF export with official compliance signatures

9. **BCA Project Defense & Viva Guide (`/about`)**:
   - Complete project documentation, CPCB sub-index linear piecewise formula, technology stack breakdown, and 10 frequently asked university viva questions and model answers

---

## 3. Machine Learning & Data Science Architecture

- **Algorithm**: `RandomForestRegressor` (Ensemble of 100 Decision Trees)
- **Mathematical Formula**: CPCB standard piecewise linear interpolation:
  $$I_p = I_{low} + \left(\frac{I_{high} - I_{low}}{B_{high} - B_{low}}\right) \times (C_p - B_{low})$$
  $$\text{Composite AQI} = \max(I_{PM2.5}, I_{PM10}, I_{NO2}, I_{SO2}, I_{CO}, I_{O3})$$
- **Model Performance**:
  - $R^2$ Determination Score: **0.9995** (99.95% accuracy)
  - Mean Absolute Error (MAE): **0.93 AQI points**
  - Root Mean Squared Error (RMSE): **3.49**
  - Primary Drivers: **PM2.5 (66.5%)** and **PM10 (29.2%)** account for over 95% of prediction weight

---

## 4. How to Run the Project

### Prerequisites:
- Python 3.10+ installed
- Web browser (Chrome, Edge, Firefox)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Web Server
```bash
python app.py
```

### Step 3: Open in Browser
Visit `http://127.0.0.1:5000` in your web browser.

---

## 5. Technology Stack
- **Backend Framework**: Python 3.13, Flask 3.1.3
- **Data Science & ML**: scikit-learn, pandas, numpy, joblib
- **Frontend UI**: HTML5, Tailwind CSS, Font Awesome 6
- **Data Visualization**: Chart.js 4 (Interactive line and bar charts)
- **Geographic Mapping**: Leaflet.js (OpenStreetMap)
- **Live APIs**: Open-Meteo Air Quality API (with zero-failure offline diurnal fallback)

---

## 6. Academic Attributions
- Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Govt. of India
- World Health Organization (WHO) Global Air Quality Guidelines (AQG)
- NASA Clean Air Study (Houseplants for Indoor Air Pollution Abatement)
