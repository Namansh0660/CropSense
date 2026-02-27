# 🌾 CropSense — Adaptive Crop Recommendation System

> An AI-powered, feedback-driven crop recommendation framework using XGBoost and explainable AI (XAI) for sustainable precision agriculture in Tamil Nadu, India.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Model & Performance](#-model--performance)
- [Explainability (SHAP)](#-explainability-shap)
- [Dashboard](#-dashboard)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Authors](#-authors)

---

## 🔍 Overview

CropSense predicts the most suitable dominant crop for a Tamil Nadu district based on satellite and soil data. It uses **NDVI** (vegetation health), **Rainfall**, and **Soil pH** as inputs — all sourced from public satellite databases via Google Earth Engine.

What makes it different from a standard ML model is its **closed-loop feedback system**: after farmers harvest, they report their actual yield and satisfaction. The model automatically retrains on this data, improving its recommendations each cycle.

> **Final model accuracy: 82.35% (XGBoost, post-tuning) → 96.04% after adaptive retraining with farmer feedback**

---

## ✨ Key Features

- 🛰️ **Satellite-driven inputs** — NDVI (MODIS), Rainfall (NASA GPM), Soil pH (SoilGrids)
- 🤖 **XGBoost classifier** with SMOTE balancing and hyperparameter tuning
- ⚖️ **Class imbalance handled** via SMOTE (9 crops balanced to equal representation)
- 🔍 **SHAP explainability** — understand *why* each crop was recommended
- 🔁 **Adaptive retraining loop** — model improves with every farmer feedback submission
- 📊 **Interactive Streamlit dashboard** with predictions, analytics, and leaderboards
- 🏆 **District leaderboard** ranked by yield and farmer satisfaction

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│          Phase 1: Data Collection               │
│  MODIS NDVI · NASA GPM Rainfall · SoilGrids pH  │
│  + Tamil Nadu Govt Crop Area Statistics         │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│        Phase 2: Feature Engineering             │
│  NDVI×Rainfall · Rainfall×pH · NDVI×pH          │
│  Min-Max Scaling · SMOTE Balancing               │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│         Phase 3: Model Training                 │
│  XGBoost · Hyperparameter Tuning                 │
│  5-Fold Cross-Validation · SHAP Analysis         │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│      Phase 4: Streamlit Dashboard               │
│  Prediction UI · Farmer Feedback Collection      │
│  Analytics · Temporal Trends · Leaderboard       │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│      Phase 5: Closed-Loop Retraining            │
│  Feedback → Retrain → Deploy → Repeat            │
└─────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Source:** Google Earth Engine + Tamil Nadu Government Statistics

| Feature | Description | Source |
|---|---|---|
| `NDVI` | Vegetation greenness index (0–1) | MODIS MOD13Q1 |
| `Rainfall` | Seasonal precipitation (normalized) | NASA GPM IMERG |
| `Soil_pH` | Soil acidity / alkalinity | SoilGrids (ISRIC-FAO) |
| `Season` | Kharif / Rabi / Zaid | Derived |
| `District` | ADM2-level boundary | FAO GAUL |
| `dominant_crop` | Most cultivated crop **(target)** | Govt. Area Under Food Crops |

- **Records:** 87 district-season combinations across 29 Tamil Nadu districts
- **Target classes:** 9 crops — Paddy, Maize, Groundnut, Banana, Cotton, Coconut, Vegetables, Millet, Tea/Coffee
- **Class balancing:** SMOTE applied to equalize all 9 classes to 28 samples each

### Engineered Interaction Features
```python
NDVI_Rainfall = NDVI × Rainfall
Rainfall_pH   = Rainfall × (7 - |7 - Soil_pH|)
NDVI_Soil     = NDVI × Soil_pH
```

---

## 🤖 Model & Performance

After benchmarking 13 algorithms, **XGBoost** was selected as the production model based on highest accuracy and F1-score balance.

### Final Model: XGBoost

| Metric | Score |
|---|---|
| **Accuracy** | **82.35%** |
| **F1-Score (Weighted)** | **0.813** |
| **F1-Score (Macro)** | **0.820** |
| **Cohen's Kappa** | **0.823** |

**Best hyperparameters found via RandomizedSearchCV:**
```python
{
  "n_estimators": 300,
  "max_depth": 5,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 1.0,
  "gamma": 0
}
```

**Evaluation strategy:**
- Train/Test split: 80:20
- Validation: 5-fold Stratified Cross-Validation
- Tuning: RandomizedSearchCV (25 iterations, 3 folds)

> Full model comparison results across all 13 algorithms are available in `tuned_model_leaderboard.csv`.

### After Adaptive Retraining with Farmer Feedback

| Metric | Score |
|---|---|
| Accuracy | **96.04%** |
| F1-Score (Weighted) | **94.57%** |

---

## 🔍 Explainability (SHAP)

SHAP values reveal the three features driving every recommendation:

| Rank | Feature | What it captures |
|---|---|---|
| 🥇 | **NDVI** | Perennial crop suitability (Coconut, Tea/Coffee need dense canopy) |
| 🥈 | **Rainfall** | Monsoon crops (Paddy, Maize need high precipitation) |
| 🥉 | **Soil pH** | Separates acidic-soil crops (Tea in Nilgiris) from neutral-soil crops (Paddy) |

The SHAP panel in the app shows per-prediction feature contributions — farmers and agronomists can see exactly *why* a crop was recommended.

---

## 🖥️ Dashboard

The Streamlit app (`app1.py`) has 7 sections:

| Section | What it does |
|---|---|
| **Input Panel** | Select district, season; adjust NDVI, Rainfall, Soil pH sliders |
| **Prediction Result** | Shows recommended crop, confidence %, probability chart for all 9 crops |
| **SHAP Explanation** | Feature importance bar chart for the current prediction |
| **Farmer Feedback** | Log actual crop grown, yield (kg/ha), and satisfaction (1–5 ⭐) |
| **Analytics Dashboard** | KPIs, satisfaction by crop, yield vs. satisfaction scatter plot |
| **Temporal Trends** | Yield and satisfaction trends over submission timeline |
| **Adaptive Retraining** | One-click model retraining + displays updated accuracy/F1 |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Namansh0660/cropsense.git
cd cropsense
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app1.py
```

The app will open at `http://localhost:8501`

---

## 🧭 Usage

### Making a Prediction

1. Select your **district** and **season** from the dropdowns
2. Set **NDVI** (0–1), **Rainfall** (0–1.5), and **Soil pH** (4.0–8.0) using the sliders
3. Click **"Predict Dominant Crop"**
4. View the recommended crop, confidence score, and probability breakdown

### Submitting Farmer Feedback

1. After harvest, go to the **Farmer Feedback** section
2. Select the crop you actually grew, enter yield and satisfaction
3. Click **"Submit Feedback"** — saved to `feedback_data.csv`

### Retraining the Model

1. After collecting feedback, scroll to **Adaptive Retraining**
2. Click **"Retrain Model Using Latest Feedback"**
3. Updated model is saved as `best_tuned_model.pkl` and deployed instantly

---

## 📁 Project Structure

```
cropsense/
│
├── app1.py                      # 🖥️  Main Streamlit dashboard
├── notebook.ipynb               # 📓  Full ML training pipeline
│
├── final_dataset1.csv           # 📊  Tamil Nadu district-season dataset
├── feedback_data.csv            # 💬  Farmer feedback (auto-generated)
│
├── best_tuned_model.pkl         # 🤖  Deployed XGBoost model
├── label_encoder.pkl            # 🏷️  Crop label encoder
├── label_encoder.joblib         # 🏷️  Crop label encoder (joblib format)
├── scaler.pkl                   # ⚖️  MinMaxScaler
├── poly_features.pkl            # 🔢  Polynomial features transformer
├── tuned_model_leaderboard.csv  # 📈  All 13 models comparison results
│
├── requirements.txt             # 📦  Python dependencies
└── LICENSE                      # 📄  MIT License
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Data Collection** | Google Earth Engine, MODIS, NASA GPM, SoilGrids |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Class Balancing** | imbalanced-learn (SMOTE) |
| **ML Model** | XGBoost |
| **Explainability** | SHAP |
| **Frontend** | Streamlit |
| **Serialization** | Joblib, Cloudpickle |
| **Visualization** | Matplotlib, Seaborn |

---

## 👨‍💻 Authors
| Name - **Namansh Singh Maurya** |
