# AI Agricultural Decision Intelligence System

> Transforming crop prediction into **actionable decision-making** using AI, simulation, and explainable insights.

---

##  Overview

This project is an **end-to-end AI-powered decision support system** designed for agriculture.  
Unlike traditional ML projects that stop at prediction, this system focuses on:

-  **Climate Simulation (What-if Analysis)**
-  **Multi-Target Prediction**
-  **Explainable AI (SHAP)**
-  **Actionable Recommendations**
-  **Interactive Web Dashboard (Flask)**

---

##  Key Features

###  1. Multi-Output Machine Learning
- Crop Failure Prediction (Classification)
- Stress Level Detection (Multi-class)
- Suitability Score (Regression)

---

###  2. Climate Simulation Engine
Simulate environmental changes such as:
- Temperature increase
- Soil moisture variation

 Observe real-time impact on predictions

---

###  3. Explainable AI (SHAP)
- Feature importance visualization  
- Model transparency  
- Insight into **why predictions change**

---

### 4. Decision Recommendation System
- Suggests actions like:
  - Improve irrigation  
  - Adjust nutrient levels  
- Converts predictions → **practical decisions**

---

### 5. Interactive Dashboard
- Real-time sliders (What-if simulation)
- Prediction comparison (Before vs After)
- SHAP-based feature impact charts
- Clean and modern UI

---

###  6. Authentication System
- User login & registration  
- Role-based access (Admin/User)  
- Protected dashboard  

---

##  System Architecture
User Input → ML Models → Predictions
↓
Simulation Engine
↓
Explainability (SHAP)
↓
Recommendation Engine
↓
Flask Dashboard


---

## 🛠️ Tech Stack

**Machine Learning**
- Python
- Scikit-learn
- XGBoost

**Data Analysis**
- Pandas
- NumPy
- Matplotlib / Seaborn

**Explainability**
- SHAP

**Backend**
- Flask

**Frontend**
- HTML, CSS, JavaScript
- Chart.js

**Database**
- SQLite

---

## 📊 Dataset Highlights

- Environmental & soil features  
- Nutrient composition (N, P, K)  
- Climate variables (temperature, moisture)  
- Derived targets:
  - Failure risk  
  - Stress level  
  - Suitability score  

---

## 🎯 Key Insights

- Extreme environmental conditions significantly impact crop failure  
- Moisture + nutrient interaction plays a critical role in stress levels  
- Suitability score shows **multi-modal behavior**, indicating diverse agricultural conditions  
