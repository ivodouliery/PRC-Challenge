# ✈️ PRC 2025 Challenge - Fuel Prediction

**Performance:** RMSE ~224 (Top 10) 🏆  
**Goal:** Predict aircraft fuel consumption based on trajectory data.

## 📋 Overview

This repository contains the complete pipeline to process flight data, generate features, and train an ensemble of Gradient Boosting models (XGBoost, LightGBM, CatBoost) to predict fuel consumption.

**Key Features:**
*   **Robust Data Cleaning:** Automatic handling of "ghost" segments (missing trajectory data) and physical outlier removal.
*   **Context-Aware Features:** Implementation of Lag/Lead features to capture flight dynamics (climb history, next phase anticipation).
*   **Ensemble Learning:** Weighted blending of XGBoost, LightGBM, and CatBoost optimized via `scipy.minimize`.
*   **Physics-Informed (V2):** Includes an experimental mass estimator (`mass_estimator.py`) based on inverse flight dynamics.

## 🛠️ Project Structure

```
├── clean_trajectories.py    # Step 1: Cleans raw ADS-B trajectories (outliers, interpolation)
├── feature_engineering.py   # Step 2: Generates physical features from trajectories
├── data_utils.py            # 🧠 Core logic: Centralized data loading & context feature generation
├── optimize_infinity.py     # Step 3: Hyperparameter optimization (Optuna) for XGB/LGB/CatBoost
├── train_blend.py           # Step 4: Final training, blending, and submission generation
├── run_robust.py            # Utility: Robust runner for long processes (auto-restart)
├── run_robust_rank.py       # Utility: Robust runner for the ranking dataset
└── mass_estimator.py        # Experimental: Inverse physics for mass estimation (V2)
```

## 🚀 Usage

### 1. Environment Setup
```bash
pip install pandas numpy xgboost lightgbm catboost optuna scipy pyarrow fastparquet
```

### 2. Pipeline Execution

**Step 1: Clean Trajectories**
```bash
python clean_trajectories.py --input data/raw --output data/clean
```

**Step 2: Feature Engineering**
```bash
python run_robust.py # Wrapper around feature_engineering.py to handle crashes
```

**Step 3: Model Optimization**
Runs Optuna optimization for 50 trials per model.
```bash
python optimize_infinity.py
```

**Step 4: Generate Submission**
Trains final models on full data, optimizes blending weights, and generates `submission.parquet`.
```bash
python train_blend.py
```

## 📊 Methodology

### 1. Data Preprocessing (`clean_trajectories.py`)
*   **Outlier Removal:** Filters points with unrealistic altitudes or speeds.
*   **Interpolation:** Fills small gaps (up to 60s) in trajectory data to maintain continuity.
*   **Flight Phase Detection:** Uses OpenAP or a fallback heuristic to label phases (CLIMB, CRUISE, DESCENT).
*   **Airspeed Calculation:** Derives True Airspeed (TAS) and Mach number from ground speed and altitude if not available.

### 2. Feature Engineering (`feature_engineering.py` & `data_utils.py`)
*   **Spatial:** Distance flown, track changes.
*   **Temporal:** Duration, time since takeoff.
*   **Dynamics:** Vertical rate, ground speed, altitude (Mean/Min/Max/Std).
*   **Context (Lag/Lead):**
    *   `prev_alt_mean`, `prev_vrate_mean`: Captures the state of the aircraft in the previous segment.
    *   `next_phase`: Anticipates the next flight phase (e.g., transition from Cruise to Descent).
*   **Physics-Based:** Estimates fuel flow using OpenAP and Acropole models as baseline features.

### 3. Modeling Strategy (`optimize_infinity.py` & `train_blend.py`)
We use an ensemble of three gradient boosting libraries to maximize performance and diversity:

*   **XGBoost:** Configured with `hist` tree method for speed. Excellent for structured data.
*   **LightGBM:** Uses leaf-wise growth. Very fast and handles categorical features natively.
*   **CatBoost:** Uses symmetric trees and ordered boosting. Best performance on categorical data (Aircraft Type, Phase).

**Optimization:**
Hyperparameters (learning rate, depth, regularization) are optimized using **Optuna** with 5-fold GroupKFold cross-validation (grouped by `flight_id` to prevent leakage).

**Blending:**
The final predictions are a weighted average of the three models. The weights are optimized using `scipy.optimize.minimize` to minimize RMSE on the Out-Of-Fold (OOF) predictions.

## 🛡️ Robustness (`run_robust.py`)
Processing large datasets can be unstable due to memory leaks or specific corrupted files. The `run_robust.py` script:
1.  Monitors the feature engineering process.
2.  Automatically restarts the script if it crashes.
3.  Identifies and blacklists flights that cause repeated crashes (`blacklist.txt`).

## 📈 Results

| Model | RMSE (CV) |
|-------|-----------|
| XGBoost | ~246.8 |
| LightGBM | ~245.4 |
| CatBoost | ~244.9 |
| **Ensemble** | **~224.0** |

## 🔮 Future Work (V2)
*   **Mass Estimation:** Integrate `mass_estimator.py` to estimate initial aircraft mass, which is a critical factor for fuel consumption.
*   **Weather Integration:** Incorporate wind and temperature data using `fastmeteo` to improve ground speed and fuel flow calculations.
