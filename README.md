# 🌾 AgriAI — Agricultural Decision Intelligence Platform

A production-ready AI-powered agricultural decision support SaaS platform.

---

## 📁 Project Structure

```
agri_platform/
├── app.py                         # Flask application & all API routes
├── requirements.txt
├── database.db                    # Auto-created on first run
│
├── models/
│   ├── Failure_classification_Mpdel.pkl
│   ├── Stress_Level_classification_Mpdel.pkl
│   └── Suitability_Score_Mpdel.pkl
│
├── utils/
│   ├── db.py          # SQLite init & connection
│   ├── predict.py     # 3-model inference pipeline
│   ├── explain.py     # SHAP feature importance
│   ├── gpt_explain.py # OpenAI GPT-4o reasoning layer
│   ├── chat.py        # Context-aware chat assistant
│   ├── recommend.py   # Rule-based recommendation engine
│   └── report.py      # ReportLab PDF generation
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── analytics.html
│   └── admin.html
│
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## 🚀 Setup Instructions

### 1. Extract Models

Extract `Models.rar` and place the `.pkl` files in the `models/` directory:
```
models/Failure_classification_Mpdel.pkl
models/Stress_Level_classification_Mpdel.pkl
models/Suitability_Score_Mpdel.pkl
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (optional)

```bash
# For GPT-powered explanations and chat
export OPENAI_API_KEY="sk-..."

# For production
export SECRET_KEY="your-secure-random-key"
```

> Without `OPENAI_API_KEY`, the platform uses a high-quality rule-based fallback for explanations and chat.

### 4. Run the Application

```bash
cd agri_platform
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🔐 Default Credentials

| Role  | Username | Password  |
|-------|----------|-----------|
| Admin | admin    | admin123  |

> Change the admin password immediately in production.

---

## 🎯 Features

| Feature | Description |
|---|---|
| Multi-model prediction | Failure classification, stress level, suitability regression |
| Real-time simulation | Sliders with instant AJAX results |
| SHAP explainability | Top-3 feature importance with bar chart |
| GPT reasoning layer | GPT-4o explanations with rule-based fallback |
| Context-aware chat | Agricultural assistant with prediction memory |
| Recommendation engine | 15+ agronomic rules, priority-ranked |
| PDF report download | ReportLab-generated professional reports |
| Analytics dashboard | Failure trend, stress distribution, KPIs |
| Prediction history | Full audit trail per user |
| Role-based auth | Admin panel with user management |

---

## 🔧 Feature Input Columns

The platform expects these features from the trained models:

| Column | Description | Range |
|---|---|---|
| `air_temp_c` | Air temperature | -5 to 50°C |
| `soil_moisture_pct` | Volumetric soil moisture | 0–100% |
| `nitrogen_ppm` | Nitrogen concentration | 0–300 ppm |
| `humidity_pct` | Relative humidity | 0–100% |
| `ph_level` | Soil pH | 3.0–10.0 |
| `rainfall_mm` | Rainfall | 0–300 mm |
| `solar_radiation` | Solar irradiance | 0–800 W/m² |
| `wind_speed_kmh` | Wind speed | 0–100 km/h |

> If your models were trained on a different feature set, edit `FEATURE_COLUMNS` in `utils/predict.py`.
