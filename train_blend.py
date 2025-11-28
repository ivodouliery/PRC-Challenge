import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import joblib
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# ==========================================
# CONFIG
# ==========================================
PATH_TRAIN = "data/features_train.parquet"
PATH_RANK = "data/features_rank.parquet"
MODELS_DIR = "models"
N_FOLDS = 5

from data_utils import load_and_preprocess_data

def load_data(path, is_train=True):
    # Use centralized loader
    # For rank (is_train=False), the loader will not filter outliers, which is correct
    return load_and_preprocess_data(path, is_train=is_train, subsample_ratio=1.0)

def train_and_predict_oof(model_type, params, X, y, groups, cat_cols):
    kf = GroupKFold(n_splits=N_FOLDS)
    oof_preds = np.zeros(len(X))
    models = []
    
    print(f"Training {model_type} (CV={N_FOLDS})...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        if model_type == 'xgb':
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        elif model_type == 'cat':
            train_pool = Pool(X_train, y_train, cat_features=cat_cols)
            val_pool = Pool(X_val, y_val, cat_features=cat_cols)
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool)
            
        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)
        print(f"  Fold {fold+1}/{N_FOLDS} done.")
        
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"  => RMSE OOF {model_type}: {rmse:.4f}")
    return oof_preds, models

def main():
    # 1. Load Data
    X_train, y_train, groups_train, cat_cols, _ = load_data(PATH_TRAIN, is_train=True)
    X_rank, _, _, _, df_rank = load_data(PATH_RANK, is_train=False)
    
    # 2. Load Params
    preds_dict = {}
    models_dict = {}
    
    # XGBoost
    if os.path.exists(f"{MODELS_DIR}/best_params_xgb_night.pkl"):
        params_xgb = joblib.load(f"{MODELS_DIR}/best_params_xgb_night.pkl")
        preds_dict['xgb'], models_dict['xgb'] = train_and_predict_oof('xgb', params_xgb, X_train, y_train, groups_train, cat_cols)
        
    # LightGBM
    if os.path.exists(f"{MODELS_DIR}/best_params_lgb_night.pkl"):
        params_lgb = joblib.load(f"{MODELS_DIR}/best_params_lgb_night.pkl")
        preds_dict['lgb'], models_dict['lgb'] = train_and_predict_oof('lgb', params_lgb, X_train, y_train, groups_train, cat_cols)

    # CatBoost
    if os.path.exists(f"{MODELS_DIR}/best_params_cat_night.pkl"):
        params_cat = joblib.load(f"{MODELS_DIR}/best_params_cat_night.pkl")
        preds_dict['cat'], models_dict['cat'] = train_and_predict_oof('cat', params_cat, X_train, y_train, groups_train, cat_cols)

    if not preds_dict:
        print("No models found! Run optimization first.")
        return

    # 3. Optimize Blending Weights
    print("\nOptimizing Blending...")
    model_names = list(preds_dict.keys())
    X_blend = np.column_stack([preds_dict[m] for m in model_names])
    
    def loss_func(weights):
        final_pred = np.average(X_blend, axis=1, weights=weights)
        return np.sqrt(mean_squared_error(y_train, final_pred))
    
    init_weights = [1/len(model_names)] * len(model_names)
    res = minimize(loss_func, init_weights, bounds=[(0,1)]*len(model_names), constraints={'type':'eq', 'fun': lambda w: np.sum(w)-1})
    
    best_weights = res.x
    print(f"Optimal weights: {dict(zip(model_names, best_weights))}")
    print(f"Blending RMSE: {res.fun:.4f}")
    
    # 4. Predict on Rank
    print("\nPredicting on Rank...")
    final_preds_rank = np.zeros(len(X_rank))
    
    for i, m_name in enumerate(model_names):
        weight = best_weights[i]
        if weight < 0.01: continue # Skip if weight is negligible
        
        # Average predictions of all folds for this model
        model_preds = np.zeros(len(X_rank))
        for model in models_dict[m_name]:
            if m_name == 'cat':
                model_preds += model.predict(Pool(X_rank, cat_features=cat_cols)) / N_FOLDS
            else:
                model_preds += model.predict(X_rank) / N_FOLDS
        
        final_preds_rank += model_preds * weight
        
    # 5. Submission
    print("\nGenerating submission...")
    
    # Create DF with predictions
    df_preds = pd.DataFrame({
        'flight_id': df_rank['flight_id'],
        'idx': df_rank['idx'],
        'fuel_kg': final_preds_rank
    })
    
    # Load submission template to ensure order
    submission_template = pd.read_parquet("prc_data/fuel_rank_submission.parquet")
    
    # Merge (in case order is different)
    # Keep start and end from template
    final_submission = pd.merge(
        submission_template[['flight_id', 'idx', 'start', 'end']], 
        df_preds, 
        on=['flight_id', 'idx'], 
        how='left'
    )
    
    # Fill potential missing values (should not happen)
    n_missing = final_submission['fuel_kg'].isna().sum()
    if n_missing > 0:
        print(f"⚠️ WARNING: {n_missing} missing segments in predictions!")
        final_submission['fuel_kg'] = final_submission['fuel_kg'].fillna(0)
    
    # Save as Parquet
    final_submission.to_parquet("submission.parquet", index=False)
    print(f"Submission generated: submission.parquet ({len(final_submission)} rows)")

if __name__ == "__main__":
    main()
