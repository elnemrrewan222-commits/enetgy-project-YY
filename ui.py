# Step 0: Automatic Dependency Verification & Installation
import sys
import subprocess

required_packages = ["gradio", "sklearn", "pandas", "numpy", "matplotlib", "joblib"]
missing_packages = []

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append("scikit-learn" if pkg == "sklearn" else pkg)

if missing_packages:
    print(f"Missing libraries detected: {missing_packages}")
    print("Installing dependencies automatically into current environment. Please wait...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
    print("All required packages installed successfully!")
else:
    print("All required libraries are installed and ready to go.")



# Step 1: Environment Setup & Cross-Version Unpickling Compatibility
import os
import sys
import pickle
import warnings
import io
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

# Suppress sklearn unpickle version warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Cross-version compatibility patch for scikit-learn (e.g. 1.6 -> 1.8+)
import sklearn.compose._column_transformer as ct
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

import sklearn.linear_model
if not hasattr(sklearn.linear_model.LinearRegression, 'tol'):
    sklearn.linear_model.LinearRegression.tol = None

print("Libraries imported and runtime compatibility patches applied.")



# Step 2: Load the Trained Production Pipeline
MODEL_PATH = pathlib.Path("models/best_model_pipeline.pkl")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

# Ensure estimator compatibility
regressor = pipeline.named_steps.get("model", pipeline.steps[-1][1])
if not hasattr(regressor, 'tol'):
    regressor.tol = None

preprocessor = pipeline.named_steps.get("preprocess", pipeline.steps[0][1])
transformed_feature_names = list(preprocessor.get_feature_names_out())
coefs = regressor.coef_
intercept = float(regressor.intercept_)

print("Model Pipeline successfully loaded.")
print(f" - Fitted Intercept: {intercept:.4f} kWh")
print(f" - Transformed Features: {len(transformed_feature_names)}")

# Feature mapping and physical engineering descriptions
FACTOR_DESCRIPTIONS = {
    "num__has_solar": ("Rooftop Solar PV", "Clean Energy", "Reduces net grid draw via onsite photovoltaic generation"),
    "num__has_ev": ("Electric Vehicle Charger", "High-Draw Asset", "Adds daily battery replenishment energy consumption"),
    "cat__heating_type_electric": ("Heating: Electric Resistance", "Space Heating", "High electricity draw for thermal heat generation"),
    "num__home_sqft": ("Living Area (sqft)", "Dwelling Geometry", "Larger floor spaces increase baseline heating/cooling demand"),
    "cat__heating_type_gas": ("Heating: Natural Gas", "Space Heating", "Shifts heating fuel to natural gas, reducing electric grid load"),
    "num__has_pool": ("Private Swimming Pool", "High-Draw Asset", "Adds continuous pump circulation and optional water heating"),
    "num__temp_avg_c": ("Outdoor Avg Temperature", "Meteorological", "Colder temperatures increase heating system duty cycles"),
    "num__num_residents": ("Number of Occupants", "Occupancy", "More occupants drive water heating, cooking, and appliance use"),
    "num__is_weekend": ("Weekend Activity", "Temporal Context", "Higher home occupancy and daytime appliance usage on weekends"),
    "num__prior_week_avg_kwh": ("7-Day Historical Baseline", "Telemetry", "Captures household persistent base load and occupancy habits"),
    "num__solar_index": ("Solar Irradiance Index", "Meteorological", "High solar irradiance promotes daylighting and solar heat gain"),
    "num__hvac_age_years": ("HVAC Equipment Age", "Equipment Health", "Older heating/cooling systems operate at lower seasonal efficiency"),
    "cat__heating_type_heat_pump": ("Heating: Heat Pump", "Space Heating", "High seasonal COP efficiency compared to resistance heat"),
    "num__is_holiday": ("Holiday Occupancy", "Temporal Context", "Occupancy changes during observed official holidays"),
    "num__temp_min_c": ("Outdoor Min Temperature", "Meteorological", "Overnight low temperature driving nighttime heating demand"),
    "num__prior_day_kwh": ("Prior Day Baseline", "Telemetry", "Immediate preceding day consumption for persistence auto-correlation"),
    "num__temp_max_c": ("Outdoor Max Temperature", "Meteorological", "Peak daytime temperature driving cooling load when warm"),
    "num__day_of_week": ("Day of Week Index", "Temporal Context", "Day cycle variation (Monday=0 to Sunday=6)"),
    "num__wind_kph": ("Wind Speed (km/h)", "Meteorological", "Wind infiltration increasing building envelope thermal loss"),
    "num__humidity_pct": ("Relative Humidity (%)", "Meteorological", "Moisture impact on perceived temperature and latent cooling load"),
    "num__precip_mm": ("Precipitation (mm)", "Meteorological", "Rainfall or snow impact on outdoor activities and indoor lighting")
}

# Construct coefficient inspection DataFrame
coef_rows = []
for name, c in zip(transformed_feature_names, coefs):
    meta = FACTOR_DESCRIPTIONS.get(name, (name, "Other", "Calibrated coefficient"))
    coef_rows.append({
        "Transformed Feature": name,
        "Friendly Name": meta[0],
        "Category": meta[1],
        "Standardized Weight": round(c, 4),
        "Absolute Magnitude": round(abs(c), 4),
        "Impact Direction": "Increases Demand" if c > 0 else "Reduces Demand",
        "Physical Engineering Interpretation": meta[2]
    })

coef_registry_df = pd.DataFrame(coef_rows).sort_values(by="Absolute Magnitude", ascending=False).reset_index(drop=True)



# Step 3: Core Prediction Engine & Analytical Functions

DAY_NAME_TO_INT = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

INT_TO_DAY_NAME = {v: k for k, v in DAY_NAME_TO_INT.items()}

FEATURE_COLUMNS = [
    'num_residents', 'home_sqft', 'has_ev', 'has_solar', 'has_pool',
    'heating_type', 'hvac_age_years', 'temp_avg_c', 'temp_min_c', 'temp_max_c',
    'humidity_pct', 'wind_kph', 'precip_mm', 'solar_index', 'is_weekend',
    'is_holiday', 'prior_day_kwh', 'prior_week_avg_kwh', 'day_of_week'
]

def predict_single_household(
    num_residents, home_sqft, has_ev, has_solar, has_pool,
    heating_type, hvac_age_years, temp_avg_c, temp_min_c, temp_max_c,
    humidity_pct, wind_kph, precip_mm, solar_index, day_name,
    is_weekend_override, is_holiday, prior_day_kwh, prior_week_avg_kwh, tariff_rate
):
    """Predicts daily kWh for a single household and decomposes contributing drivers."""
    day_of_week = DAY_NAME_TO_INT.get(day_name, 2)
    # Determine weekend
    if is_weekend_override is not None:
        is_weekend = 1 if is_weekend_override else 0
    else:
        is_weekend = 1 if day_of_week in [5, 6] else 0

    heating_map = {
        "Natural Gas Furnace": "gas",
        "Electric Resistance": "electric",
        "Air-Source Heat Pump": "heat_pump",
        "gas": "gas", "electric": "electric", "heat_pump": "heat_pump"
    }
    heating_clean = heating_map.get(heating_type, "gas")

    input_dict = {
        'num_residents': int(num_residents),
        'home_sqft': float(home_sqft),
        'has_ev': 1 if has_ev else 0,
        'has_solar': 1 if has_solar else 0,
        'has_pool': 1 if has_pool else 0,
        'heating_type': heating_clean,
        'hvac_age_years': float(hvac_age_years),
        'temp_avg_c': float(temp_avg_c),
        'temp_min_c': float(temp_min_c),
        'temp_max_c': float(temp_max_c),
        'humidity_pct': float(humidity_pct),
        'wind_kph': float(wind_kph),
        'precip_mm': float(precip_mm),
        'solar_index': float(solar_index),
        'is_weekend': int(is_weekend),
        'is_holiday': 1 if is_holiday else 0,
        'prior_day_kwh': float(prior_day_kwh),
        'prior_week_avg_kwh': float(prior_week_avg_kwh),
        'day_of_week': int(day_of_week)
    }

    df_in = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    raw_pred = float(pipeline.predict(df_in)[0])
    pred_kwh = max(0.1, round(raw_pred, 2))

    # Economics & Emissions
    cost_daily = round(pred_kwh * tariff_rate, 2)
    cost_monthly = round(cost_daily * 30.0, 2)
    co2_daily = round(pred_kwh * 0.385, 2)

    # Variances
    diff_prior_day = round(pred_kwh - prior_day_kwh, 2)
    pct_prior_day = round((diff_prior_day / prior_day_kwh * 100), 1) if prior_day_kwh > 0 else 0.0

    diff_prior_week = round(pred_kwh - prior_week_avg_kwh, 2)
    pct_prior_week = round((diff_prior_week / prior_week_avg_kwh * 100), 1) if prior_week_avg_kwh > 0 else 0.0

    # Demand Tier Classification
    if pred_kwh < 20.0:
        tier_label = "LOW CONSUMPTION"
        tier_bg = "#ecfdf5"
        tier_color = "#047857"
        tier_desc = "Optimal efficiency or active solar offset. Grid draw is well below regional benchmark."
    elif pred_kwh <= 32.0:
        tier_label = "STANDARD BASELINE"
        tier_bg = "#f0fdf4"
        tier_color = "#15803d"
        tier_desc = "Within typical residential baseline bounds (mean benchmark: 26.1 kWh)."
    elif pred_kwh <= 45.0:
        tier_label = "ELEVATED DEMAND"
        tier_bg = "#fffbeb"
        tier_color = "#b45309"
        tier_desc = "Noticeable heating/cooling load, EV charging, or multiple residents."
    else:
        tier_label = "PEAK HIGH DEMAND"
        tier_bg = "#fef2f2"
        tier_color = "#b91c1c"
        tier_desc = "Heavy consumption tier driven by extreme weather, pool pumps, or electric resistance heating."

    # Mathematical Contribution Breakdown
    X_trans = preprocessor.transform(df_in)
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    contributions = X_trans[0] * coefs

    contrib_list = []
    for f_name, c_val in zip(transformed_feature_names, contributions):
        meta = FACTOR_DESCRIPTIONS.get(f_name, (f_name, "Other", ""))
        contrib_list.append({
            "Factor": meta[0],
            "Category": meta[1],
            "Net Impact (kWh)": c_val,
            "Abs Impact": abs(c_val),
            "Direction": "Increases Load" if c_val > 0 else "Reduces Load"
        })

    contrib_df = pd.DataFrame(contrib_list)
    top_drivers = contrib_df.sort_values(by="Abs Impact", ascending=False).head(7).copy()
    top_drivers_display = pd.DataFrame({
        "Attribute": top_drivers["Factor"],
        "Category": top_drivers["Category"],
        "Effect": top_drivers["Direction"],
        "Impact Shift": top_drivers["Net Impact (kWh)"].map(lambda x: f"{x:+.2f} kWh")
    })

    # KPI Banner HTML
    kpi_banner_html = f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px 24px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04); margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase;">Forecasted Daily Consumption</div>
                <div style="font-size: 40px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; line-height: 1.1; margin: 6px 0 4px 0;">
                    {pred_kwh:.2f} <span style="font-size: 17px; font-weight: 600; color: #0d9488;">kWh</span>
                </div>
                <div style="font-size: 13px; color: #475569;">{tier_desc}</div>
            </div>
            <div style="background: {tier_bg}; color: {tier_color}; border: 1px solid {tier_color}33; padding: 6px 14px; border-radius: 9999px; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;">
                {tier_label}
            </div>
        </div>
    </div>
    """

    # Secondary Metrics HTML
    sign_day = "+" if diff_prior_day >= 0 else ""
    sign_week = "+" if diff_prior_week >= 0 else ""
    color_day = "#b45309" if diff_prior_day > 0 else "#047857"
    color_week = "#b45309" if diff_prior_week > 0 else "#047857"

    secondary_metrics_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px;">
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Est. Daily Cost</div>
            <div style="font-size: 19px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">${cost_daily:.2f}</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">@ ${tariff_rate:.2f}/kWh</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">30-Day Extrapolated</div>
            <div style="font-size: 19px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">${cost_monthly:.2f}</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Monthly projection</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Carbon Footprint</div>
            <div style="font-size: 19px; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">{co2_daily:.1f} <span style="font-size: 11px;">kg</span></div>
            <div style="font-size: 11px; color: #64748b; margin-top: 2px;">CO₂ emissions</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">vs. Prior Day</div>
            <div style="font-size: 17px; font-weight: 700; color: {color_day}; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">{sign_day}{diff_prior_day:.2f} kWh</div>
            <div style="font-size: 11px; color: {color_day}; margin-top: 2px;">{sign_day}{pct_prior_day:.1f}% shift</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">vs. 7-Day Average</div>
            <div style="font-size: 17px; font-weight: 700; color: {color_week}; font-family: 'JetBrains Mono', monospace; margin-top: 3px;">{sign_week}{diff_prior_week:.2f} kWh</div>
            <div style="font-size: 11px; color: {color_week}; margin-top: 2px;">{sign_week}{pct_prior_week:.1f}% shift</div>
        </div>
    </div>
    """

    # Benchmark Comparison Chart
    fig, ax = plt.subplots(figsize=(6.5, 2.6), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    cat_labels = ['Prior Day', '7-Day Avg', 'Regional Median', 'Forecast Today']
    cat_vals = [prior_day_kwh, prior_week_avg_kwh, 25.26, pred_kwh]
    cat_colors = ['#94a3b8', '#64748b', '#cbd5e1', '#0d9488']

    bars = ax.barh(cat_labels, cat_vals, color=cat_colors, height=0.52, edgecolor='none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#475569', labelsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='#94a3b8')

    max_axis = max(cat_vals) * 1.20
    ax.set_xlim(0, max_axis)
    ax.set_xlabel("Daily Consumption (kWh)", fontsize=9, color='#475569', labelpad=6)
    ax.set_title("Consumption Telemetry & Benchmark Comparison", fontsize=10.5, fontweight='bold', color='#0f172a', pad=10, loc='left')

    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(w + (max_axis * 0.02), y, f"{w:.1f} kWh", va='center', ha='left', fontsize=8.5, fontweight='bold', color='#0f172a', fontfamily='monospace')

    plt.tight_layout()

    return kpi_banner_html, secondary_metrics_html, top_drivers_display, fig

def predict_batch_csv(file_obj, tariff_rate=0.16):
    """Processes an uploaded CSV dataset and outputs batch predictions."""
    if file_obj is None:
        return None, "Upload a valid CSV file to begin batch processing.", None

    try:
        df_raw = pd.read_csv(file_obj.name)
    except Exception as e:
        return None, f"Error reading CSV file: {str(e)}", None

    df_proc = df_raw.copy()

    # Parse date if day_of_week is not present
    if 'day_of_week' not in df_proc.columns and 'date' in df_proc.columns:
        df_proc['date_parsed'] = pd.to_datetime(df_proc['date'], errors='coerce')
        df_proc['day_of_week'] = df_proc['date_parsed'].dt.dayofweek.fillna(2).astype(int)

    # Impute missing feature columns with default baseline medians
    for col in FEATURE_COLUMNS:
        if col not in df_proc.columns:
            if col == 'is_weekend' and 'day_of_week' in df_proc.columns:
                df_proc['is_weekend'] = df_proc['day_of_week'].isin([5, 6]).astype(int)
            elif col in ['has_ev', 'has_solar', 'has_pool', 'is_holiday']:
                df_proc[col] = 0
            elif col == 'heating_type':
                df_proc[col] = 'gas'
            elif col == 'home_sqft':
                df_proc[col] = 2500.0
            elif col == 'num_residents':
                df_proc[col] = 3
            elif col == 'prior_day_kwh':
                df_proc[col] = 25.0
            elif col == 'prior_week_avg_kwh':
                df_proc[col] = 25.5
            else:
                df_proc[col] = 0.0

    X_in = df_proc[FEATURE_COLUMNS]
    preds = pipeline.predict(X_in)
    preds = np.clip(preds, 0.1, None)

    df_results = df_raw.copy()
    df_results['predicted_kwh'] = np.round(preds, 2)
    df_results['est_cost_usd'] = np.round(preds * tariff_rate, 2)
    df_results['est_co2_kg'] = np.round(preds * 0.385, 2)

    total_kwh = float(df_results['predicted_kwh'].sum())
    mean_kwh = float(df_results['predicted_kwh'].mean())
    min_kwh = float(df_results['predicted_kwh'].min())
    max_kwh = float(df_results['predicted_kwh'].max())
    rows_count = len(df_results)

    summary_md = f"""
    ### Batch Evaluation Telemetry
    * **Records Evaluated:** {rows_count:,} observations
    * **Cumulative Grid Demand:** **{total_kwh:,.2f} kWh**
    * **Mean Daily Consumption:** **{mean_kwh:.2f} kWh**
    * **Forecast Range:** {min_kwh:.2f} kWh (minimum) — {max_kwh:.2f} kWh (peak)
    * **Total Estimated Energy Cost:** **${total_kwh * tariff_rate:,.2f}** (@ ${tariff_rate:.2f}/kWh)
    """

    out_csv_path = "batch_forecast_results.csv"
    df_results.to_csv(out_csv_path, index=False)

    return df_results.head(50), summary_md, out_csv_path

def generate_sample_batch_csv():
    """Generates a sample batch CSV for quick user testing."""
    sample_data = [
        {"num_residents": 3, "home_sqft": 2400, "has_ev": 0, "has_solar": 0, "has_pool": 0, "heating_type": "gas", "hvac_age_years": 8, "temp_avg_c": 12.0, "temp_min_c": 8.0, "temp_max_c": 16.0, "humidity_pct": 55, "wind_kph": 10.0, "precip_mm": 0.0, "solar_index": 6.5, "day_of_week": 2, "is_weekend": 0, "is_holiday": 0, "prior_day_kwh": 24.5, "prior_week_avg_kwh": 25.0},
        {"num_residents": 4, "home_sqft": 3100, "has_ev": 1, "has_solar": 1, "has_pool": 0, "heating_type": "heat_pump", "hvac_age_years": 4, "temp_avg_c": 15.0, "temp_min_c": 11.0, "temp_max_c": 20.0, "humidity_pct": 50, "wind_kph": 8.0, "precip_mm": 0.0, "solar_index": 8.0, "day_of_week": 6, "is_weekend": 1, "is_holiday": 0, "prior_day_kwh": 27.2, "prior_week_avg_kwh": 26.5},
        {"num_residents": 2, "home_sqft": 1400, "has_ev": 0, "has_solar": 0, "has_pool": 0, "heating_type": "gas", "hvac_age_years": 12, "temp_avg_c": 5.0, "temp_min_c": 1.0, "temp_max_c": 9.0, "humidity_pct": 70, "wind_kph": 14.0, "precip_mm": 2.0, "solar_index": 4.0, "day_of_week": 1, "is_weekend": 0, "is_holiday": 0, "prior_day_kwh": 18.2, "prior_week_avg_kwh": 19.1},
        {"num_residents": 5, "home_sqft": 4200, "has_ev": 1, "has_solar": 0, "has_pool": 1, "heating_type": "electric", "hvac_age_years": 14, "temp_avg_c": -5.0, "temp_min_c": -11.0, "temp_max_c": 0.0, "humidity_pct": 78, "wind_kph": 22.0, "precip_mm": 3.5, "solar_index": 2.5, "day_of_week": 5, "is_weekend": 1, "is_holiday": 0, "prior_day_kwh": 55.0, "prior_week_avg_kwh": 52.0},
        {"num_residents": 3, "home_sqft": 2600, "has_ev": 0, "has_solar": 1, "has_pool": 0, "heating_type": "gas", "hvac_age_years": 6, "temp_avg_c": 18.0, "temp_min_c": 13.0, "temp_max_c": 23.0, "humidity_pct": 48, "wind_kph": 9.0, "precip_mm": 0.0, "solar_index": 9.0, "day_of_week": 4, "is_weekend": 0, "is_holiday": 0, "prior_day_kwh": 21.0, "prior_week_avg_kwh": 22.3}
    ]
    sample_df = pd.DataFrame(sample_data)
    sample_path = "sample_household_batch.csv"
    sample_df.to_csv(sample_path, index=False)
    return sample_path

# Presets mapping
PRESETS = {
    "Suburban Baseline (Standard)": {
        "num_residents": 3, "home_sqft": 2500, "has_ev": False, "has_solar": False, "has_pool": False,
        "heating_type": "Natural Gas Furnace", "hvac_age_years": 10,
        "temp_avg_c": 12.0, "temp_min_c": 7.0, "temp_max_c": 17.0,
        "humidity_pct": 55, "wind_kph": 10.0, "precip_mm": 0.0, "solar_index": 6.5,
        "day_name": "Wednesday", "is_weekend": False, "is_holiday": False,
        "prior_day_kwh": 24.5, "prior_week_avg_kwh": 25.0
    },
    "Eco Residence (Solar + Heat Pump)": {
        "num_residents": 4, "home_sqft": 2800, "has_ev": True, "has_solar": True, "has_pool": False,
        "heating_type": "Air-Source Heat Pump", "hvac_age_years": 4,
        "temp_avg_c": 16.0, "temp_min_c": 12.0, "temp_max_c": 21.0,
        "humidity_pct": 48, "wind_kph": 8.0, "precip_mm": 0.0, "solar_index": 8.5,
        "day_name": "Sunday", "is_weekend": True, "is_holiday": False,
        "prior_day_kwh": 25.5, "prior_week_avg_kwh": 25.0
    },
    "Winter Cold Snap (Electric Heat)": {
        "num_residents": 4, "home_sqft": 3400, "has_ev": True, "has_solar": False, "has_pool": False,
        "heating_type": "Electric Resistance", "hvac_age_years": 16,
        "temp_avg_c": -8.0, "temp_min_c": -14.0, "temp_max_c": -2.0,
        "humidity_pct": 75, "wind_kph": 24.0, "precip_mm": 4.5, "solar_index": 2.0,
        "day_name": "Monday", "is_weekend": False, "is_holiday": False,
        "prior_day_kwh": 48.0, "prior_week_avg_kwh": 45.0
    },
    "Estate with Pool & High Demand": {
        "num_residents": 5, "home_sqft": 4100, "has_ev": True, "has_solar": False, "has_pool": True,
        "heating_type": "Electric Resistance", "hvac_age_years": 12,
        "temp_avg_c": 17.0, "temp_min_c": 12.0, "temp_max_c": 22.0,
        "humidity_pct": 60, "wind_kph": 12.0, "precip_mm": 0.0, "solar_index": 7.0,
        "day_name": "Saturday", "is_weekend": True, "is_holiday": True,
        "prior_day_kwh": 52.0, "prior_week_avg_kwh": 50.0
    }
}

def load_preset(name):
    p = PRESETS.get(name, PRESETS["Suburban Baseline (Standard)"])
    return (
        p["num_residents"], p["home_sqft"], p["has_ev"], p["has_solar"], p["has_pool"],
        p["heating_type"], p["hvac_age_years"], p["temp_avg_c"], p["temp_min_c"], p["temp_max_c"],
        p["humidity_pct"], p["wind_kph"], p["precip_mm"], p["solar_index"], p["day_name"],
        p["is_weekend"], p["is_holiday"], p["prior_day_kwh"], p["prior_week_avg_kwh"]
    )



# Step 4: Interface Styling and Gradio Application Layout

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

.mono-metric {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums;
}

.header-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.header-badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 6px;
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
    margin-bottom: 10px;
}

.header-title {
    font-size: 26px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
}

.header-subtitle {
    font-size: 14px;
    color: #64748b;
    margin: 0;
    line-height: 1.5;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #0f766e;
    background: #f0fdfa;
    border: 1px solid #ccfbf1;
    padding: 4px 12px;
    border-radius: 9999px;
}

.card-panel {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
}

.card-title {
    font-size: 12.5px !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    margin-bottom: 12px !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-title::before {
    content: "";
    display: inline-block;
    width: 3px;
    height: 14px;
    background-color: #0d9488;
    border-radius: 2px;
}

.btn-primary {
    background: #0f766e !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 11px 22px !important;
    transition: background 0.15s ease-in-out !important;
}

.btn-primary:hover {
    background: #0d9488 !important;
}

.btn-secondary {
    background: #ffffff !important;
    color: #475569 !important;
    font-weight: 600 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}

.btn-secondary:hover {
    background: #f8fafc !important;
    color: #0f172a !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Plus Jakarta Sans", "Inter", "sans-serif"],
    font_mono=["JetBrains Mono", "monospace"]
)

# Benchmark comparison table across ML workflow models
BENCHMARK_MODELS_DF = pd.DataFrame([
    {"Model Name": "Linear Regression (Pipeline)", "MAE (kWh)": 3.2337, "RMSE (kWh)": 4.3258, "R² Score": 0.7494, "Status": "Selected Production Model"},
    {"Model Name": "Random Forest Regressor", "MAE (kWh)": 3.2067, "RMSE (kWh)": 4.3752, "R² Score": 0.7436, "Status": "Alternative Ensemble"},
    {"Model Name": "Support Vector Regression (RBF)", "MAE (kWh)": 3.2609, "RMSE (kWh)": 4.5000, "R² Score": 0.7288, "Status": "Non-Linear Baseline"},
    {"Model Name": "K-Nearest Neighbors Regressor", "MAE (kWh)": 3.3396, "RMSE (kWh)": 4.5466, "R² Score": 0.7231, "Status": "Distance-Based Baseline"}
])

with gr.Blocks(title="GridPulse | Electricity Demand Forecaster", theme=theme, css=CUSTOM_CSS) as demo:
    # Header Banner
    gr.HTML("""
    <div class="header-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
            <div>
                <span class="header-badge">ENGINEERING ANALYTICS // MODEL PIPELINE v1.0</span>
                <h1 class="header-title">GridPulse | Household Electricity Demand Forecaster</h1>
                <p class="header-subtitle">
                    Production prediction interface for residential daily energy consumption (kWh) based on building physics, climatic parameters, and meter telemetry.
                </p>
            </div>
            <div style="text-align: right;">
                <span class="status-pill">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background-color: #0d9488; display: inline-block;"></span>
                    Calibrated Pipeline Active
                </span>
                <div style="font-size: 11px; color: #64748b; margin-top: 6px;">Target: Daily kWh | Train Samples: 88,500</div>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: Single Property Forecaster
        with gr.Tab("Single Household Forecaster"):
            # Scenario Presets Toolbar
            with gr.Row():
                with gr.Column(scale=4):
                    preset_dropdown = gr.Dropdown(
                        label="Quick Scenario Presets",
                        choices=list(PRESETS.keys()),
                        value="Suburban Baseline (Standard)",
                        interactive=True,
                        info="Load representative residential configurations in one click."
                    )
                with gr.Column(scale=1):
                    apply_preset_btn = gr.Button("Load Preset", elem_classes=["btn-secondary"])

            with gr.Row(equal_height=False):
                # LEFT COLUMN: Structured Input Cards
                with gr.Column(scale=5):
                    # Card 1: Dwelling & Structural Characteristics
                    with gr.Group(elem_classes=["card-panel"]):
                        gr.HTML('<div class="card-title">Dwelling & Heating Infrastructure</div>')
                        with gr.Row():
                            home_sqft_in = gr.Slider(minimum=700, maximum=4500, value=2500, step=50, label="Living Area (sq ft)", info="Total conditioned indoor space.")
                            num_residents_in = gr.Slider(minimum=1, maximum=6, value=3, step=1, label="Household Occupants", info="Number of permanent residents.")
                        with gr.Row():
                            heating_type_in = gr.Dropdown(
                                choices=["Natural Gas Furnace", "Air-Source Heat Pump", "Electric Resistance"],
                                value="Natural Gas Furnace",
                                label="Primary Heating System",
                                info="Main space heating technology."
                            )
                            hvac_age_in = gr.Slider(minimum=1, maximum=25, value=10, step=1, label="HVAC System Age (Years)", info="Operating age of heating/cooling unit.")

                    # Card 2: High-Draw & Renewable Assets
                    with gr.Group(elem_classes=["card-panel"]):
                        gr.HTML('<div class="card-title">High-Draw Assets & Solar Photovoltaics</div>')
                        with gr.Row():
                            has_solar_in = gr.Checkbox(value=False, label="Rooftop Solar PV Installed", info="Generates local power (-3.24 kWh offset).")
                            has_ev_in = gr.Checkbox(value=False, label="Electric Vehicle (EV Charger)", info="Adds daily charging load (+3.15 kWh).")
                            has_pool_in = gr.Checkbox(value=False, label="Private Swimming Pool", info="Pool pump & circulation (+2.37 kWh).")

                    # Card 3: Meteorological Variables
                    with gr.Group(elem_classes=["card-panel"]):
                        gr.HTML('<div class="card-title">Meteorological & Ambient Climate</div>')
                        with gr.Row():
                            temp_avg_in = gr.Slider(minimum=-20.0, maximum=25.0, value=12.0, step=0.5, label="Mean Outdoor Temp (°C)", info="24-hour mean temperature.")
                            temp_min_in = gr.Slider(minimum=-25.0, maximum=20.0, value=7.0, step=0.5, label="Min Temp (°C)", info="Nighttime minimum.")
                            temp_max_in = gr.Slider(minimum=-18.0, maximum=30.0, value=17.0, step=0.5, label="Max Temp (°C)", info="Daytime peak.")
                        with gr.Row():
                            humidity_in = gr.Slider(minimum=10, maximum=100, value=55, step=1, label="Relative Humidity (%)")
                            wind_in = gr.Slider(minimum=0.0, maximum=60.0, value=10.0, step=1.0, label="Wind Speed (km/h)")
                        with gr.Row():
                            solar_idx_in = gr.Slider(minimum=0.0, maximum=10.0, value=6.5, step=0.5, label="Solar Irradiance Index (0-10)", info="0 = Overcast, 10 = Clear Sky.")
                            precip_in = gr.Slider(minimum=0.0, maximum=50.0, value=0.0, step=0.5, label="Daily Precipitation (mm)")

                    # Card 4: Baseline Telemetry & Temporal Context
                    with gr.Group(elem_classes=["card-panel"]):
                        gr.HTML('<div class="card-title">Calendar & Trailing Meter Telemetry</div>')
                        with gr.Row():
                            day_name_in = gr.Dropdown(
                                choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                value="Wednesday",
                                label="Day of Week",
                                info="Calendar day."
                            )
                            is_weekend_in = gr.Checkbox(value=False, label="Weekend Schedule", info="Higher occupancy patterns.")
                            is_holiday_in = gr.Checkbox(value=False, label="Public Holiday", info="Official holiday.")
                        with gr.Row():
                            prior_day_in = gr.Number(value=24.5, label="Prior Day Consumption (kWh)", info="Yesterday's net meter reading.")
                            prior_week_in = gr.Number(value=25.0, label="7-Day Trailing Average (kWh)", info="Weekly trailing baseline.")
                            tariff_in = gr.Number(value=0.16, label="Electricity Tariff ($/kWh)", info="Local utility rate.")

                    # Action Buttons
                    with gr.Row():
                        calculate_btn = gr.Button("Calculate Projected Demand", elem_classes=["btn-primary"])

                # RIGHT COLUMN: Forecast Outputs & Analytics
                with gr.Column(scale=5):
                    # Output KPI Banner
                    kpi_banner_out = gr.HTML()

                    # Secondary Metrics Grid
                    secondary_metrics_out = gr.HTML()

                    # Telemetry Comparison Chart
                    chart_out = gr.Plot(label="Consumption Benchmark Comparison")

                    # Contributing Factors Table
                    with gr.Group(elem_classes=["card-panel"]):
                        gr.HTML('<div class="card-title">Key Physical Driver Breakdown</div>')
                        drivers_table_out = gr.Dataframe(
                            headers=["Attribute", "Category", "Effect", "Impact Shift"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False
                        )

            # Wire calculation events
            inputs_list = [
                num_residents_in, home_sqft_in, has_ev_in, has_solar_in, has_pool_in,
                heating_type_in, hvac_age_in, temp_avg_in, temp_min_in, temp_max_in,
                humidity_in, wind_in, precip_in, solar_idx_in, day_name_in,
                is_weekend_in, is_holiday_in, prior_day_in, prior_week_in, tariff_in
            ]
            outputs_list = [kpi_banner_out, secondary_metrics_out, drivers_table_out, chart_out]

            calculate_btn.click(
                fn=predict_single_household,
                inputs=inputs_list,
                outputs=outputs_list
            )

            # Auto-update when preset is loaded
            apply_preset_btn.click(
                fn=load_preset,
                inputs=[preset_dropdown],
                outputs=[
                    num_residents_in, home_sqft_in, has_ev_in, has_solar_in, has_pool_in,
                    heating_type_in, hvac_age_in, temp_avg_in, temp_min_in, temp_max_in,
                    humidity_in, wind_in, precip_in, solar_idx_in, day_name_in,
                    is_weekend_in, is_holiday_in, prior_day_in, prior_week_in
                ]
            ).then(
                fn=predict_single_household,
                inputs=inputs_list,
                outputs=outputs_list
            )

            # Auto-sync weekend toggle when day changes
            def sync_day(day_name):
                return day_name in ["Saturday", "Sunday"]
            day_name_in.change(fn=sync_day, inputs=[day_name_in], outputs=[is_weekend_in])

            # Initial load on demo open
            demo.load(
                fn=predict_single_household,
                inputs=inputs_list,
                outputs=outputs_list
            )

        # TAB 2: Batch Processing
        with gr.Tab("Batch Dataset Processing"):
            gr.HTML("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0;">Bulk CSV Scoring & Telemetry Inference</h3>
                <p style="font-size: 13px; color: #64748b; margin: 0;">
                    Upload a dataset formatted with building and meteorological variables to generate batch predictions. If <code>day_of_week</code> is absent, it will be automatically computed from <code>date</code>.
                </p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=3):
                    batch_file_in = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
                    batch_tariff_in = gr.Number(value=0.16, label="Electricity Tariff for Cost Estimates ($/kWh)")
                    with gr.Row():
                        run_batch_btn = gr.Button("Execute Batch Prediction", elem_classes=["btn-primary"])
                        sample_batch_btn = gr.Button("Generate Test Sample CSV", elem_classes=["btn-secondary"])
                    sample_file_out = gr.File(label="Download Generated Sample CSV", interactive=False)
                with gr.Column(scale=4):
                    batch_summary_out = gr.Markdown("Upload a CSV file and click Execute to view batch summary statistics.")
                    batch_download_out = gr.File(label="Download Processed Predictions CSV", interactive=False)

            with gr.Row():
                batch_preview_table = gr.Dataframe(label="Batch Results Preview (First 50 Rows)", interactive=False)

            run_batch_btn.click(
                fn=predict_batch_csv,
                inputs=[batch_file_in, batch_tariff_in],
                outputs=[batch_preview_table, batch_summary_out, batch_download_out]
            )

            sample_batch_btn.click(
                fn=generate_sample_batch_csv,
                inputs=[],
                outputs=[sample_file_out]
            )

        # TAB 3: Model Pipeline Architecture & Technical Registry
        with gr.Tab("Pipeline Architecture & Technical Registry"):
            gr.HTML("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                <h3 style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0;">Fitted Pipeline Architecture & Validation Registry</h3>
                <p style="font-size: 13px; color: #64748b; margin: 0;">
                    Complete technical inspection of the scikit-learn pipeline saved at <code>models/best_model_pipeline.pkl</code>, including standardization scalers, one-hot encoders, and regression weights.
                </p>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="card-title">Validation Performance Leaderboard (Holdout Last 14 Days)</div>')
                    gr.Dataframe(BENCHMARK_MODELS_DF, interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="card-title">Standardized Feature Weights & Physical Interpretations</div>')
                    gr.Dataframe(coef_registry_df, interactive=False)

print("Interface successfully constructed.")



# Step 5: Launch the Gradio Web Application
# Launches local server and opens browser automatically
demo.launch(
    share=False,
    inbrowser=True,
    show_error=True
)

