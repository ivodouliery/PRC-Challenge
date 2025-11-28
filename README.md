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

1.  **Data Preprocessing:**
    *   Missing duration imputation using theoretical time.
    *   Filtering of physically impossible fuel flow (>20 kg/s).
    *   Categorical encoding for Aircraft Type, Phase, etc.

2.  **Feature Engineering:**
    *   **Spatial:** Distance, Track change.
    *   **Temporal:** Duration, Time since takeoff.
    *   **Dynamics:** Vertical rate, Ground speed, Altitude (Mean/Min/Max/Std).
    *   **Context (New):** Previous segment stats (`prev_alt_mean`, `prev_vrate_mean`) and Next Phase (`next_phase`).

3.  **Modeling:**
    *   **XGBoost:** Histogram-based, optimized for speed/accuracy.
    *   **LightGBM:** Leaf-wise growth, handles categories natively.
    *   **CatBoost:** Symmetric trees, best for categorical data.
    *   **Blending:** Linear combination of predictions weights optimized on Out-Of-Fold (OOF) predictions.

## 📈 Results

| Model | RMSE (CV) |
|-------|-----------|
| XGBoost | ~246.8 |
| LightGBM | ~245.4 |
| CatBoost | ~244.9 |
| **Ensemble** | **~224.0** |

## 🔮 Future Work (V2)
*   Integrate `mass_estimator.py` to estimate initial aircraft mass.
*   Incorporate weather data (Wind/Temp) using `fastmeteo`.
