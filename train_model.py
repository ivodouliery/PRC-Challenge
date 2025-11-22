import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os

# CONFIGURATION
PATH_DATASET = "data/processed/X_train_final.parquet"
OUTPUT_FILE = "models/best_params_log_optuna.pkl"
N_TRIALS = 50
N_FOLDS = 3

def load_data_log():
    print(">>> CHARGEMENT (MODE LOG-TRANSFORM) <<<")
    df = pd.read_parquet(PATH_DATASET)
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg']
    features = [c for c in df.columns if c not in cols_exclude]
    
    X = df[features]
    # --- TRANSFORMATION CRITIQUE ---
    # On prédit log(1 + fuel) au lieu de fuel brut
    y = np.log1p(df['fuel_kg']) 
    
    print(f"Features ({len(features)}) : {features}")
    return X, y

def objective(trial, X, y):
    params = {
        'n_estimators': 5000,
        'objective': 'reg:squarederror', # Sur du log, cela minimise le RMSLE
        'n_jobs': -1,
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'device': 'cpu',
        'verbosity': 0
    }
    
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
    params['max_depth'] = trial.suggest_int('max_depth', 5, 12)
    params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 20) # Augmenté pour la robustesse
    params['subsample'] = trial.suggest_float('subsample', 0.6, 0.95)
    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 0.95)
    params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rmse_scores = []

    try:
        from optuna.integration import XGBoostPruningCallback
    except ImportError:
        from optuna.pruners import XGBoostPruningCallback

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        current_callbacks = [pruning_callback] if fold_idx == 0 else []

        model = xgb.XGBRegressor(**params, callbacks=current_callbacks)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Prédiction (Log Scale)
        preds_log = model.predict(X_val)
        
        # --- INVERSE TRANSFORM POUR LE SCORE RÉEL ---
        # On remet à l'échelle réelle pour calculer le vrai RMSE physique
        preds_real = np.expm1(preds_log)
        y_real = np.expm1(y_val)
        
        rmse = np.sqrt(mean_squared_error(y_real, preds_real))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    X, y = load_data_log()
    
    study = optuna.create_study(direction='minimize', study_name="fuel_log_opt")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS)

    print(f"Meilleur RMSE (Reconstruit) : {study.best_value:.4f}")
    
    best_params = study.best_params.copy()
    best_params.update({'n_estimators': 5000, 'objective': 'reg:squarederror', 
                        'n_jobs': -1, 'enable_categorical': True, 'tree_method': 'hist',
                        'device': 'cpu'}) # On retire early_stopping pour le save final
    
    joblib.dump(best_params, OUTPUT_FILE)
    print(f"Paramètres sauvés dans {OUTPUT_FILE}")