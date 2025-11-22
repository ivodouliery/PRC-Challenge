import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
PATH_DATASET = "data/processed/X_train_final.parquet"
OUTPUT_FILE  = "models/best_params_raw_optuna.pkl"
N_TRIALS     = 50  # Nombre d'essais pour trouver le réglage parfait
N_FOLDS      = 3   # K-Fold

def load_data_raw():
    print(">>> CHARGEMENT (MODE RAW - CIBLE BRUTE) <<<")
    if not os.path.exists(PATH_DATASET):
        raise FileNotFoundError(f"Fichier introuvable: {PATH_DATASET}")
        
    df = pd.read_parquet(PATH_DATASET)
    
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg']
    features = [c for c in df.columns if c not in cols_exclude]
    
    X = df[features]
    y = df['fuel_kg'] # Cible BRUTE (Pas de Log)
    
    print(f"Volume : {len(df)} segments.")
    print(f"Features ({len(features)}) : {features}")
    return X, y

def objective(trial, X, y):
    # 1. Paramètres Fixes
    params = {
        'n_estimators': 5000,
        'objective': 'reg:squarederror', # Optimise directement le MSE (donc le RMSE)
        'n_jobs': -1,
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'device': 'cpu',
        'verbosity': 0
    }
    
    # 2. Espace de Recherche (Adapté aux grandes valeurs)
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    params['max_depth'] = trial.suggest_int('max_depth', 6, 14) # On autorise plus de profondeur pour capturer la physique
    params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 20)
    params['subsample'] = trial.suggest_float('subsample', 0.6, 0.95)
    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 0.95)
    
    # Régularisation L2 (Lambda) importante pour les valeurs bruitées/rondes
    params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.1, 10.0, log=True)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)

    # 3. Cross-Validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rmse_scores = []

    # Callback Optuna (Gestion XGBoost 2.0+)
    try:
        from optuna.integration import XGBoostPruningCallback
    except ImportError:
        from optuna.pruners import XGBoostPruningCallback

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Pruning uniquement sur le premier fold pour éviter les warnings
        current_callbacks = [pruning_callback] if fold_idx == 0 else []

        # Instanciation avec callbacks
        model = xgb.XGBRegressor(**params, callbacks=current_callbacks)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        # Pas de transformation inverse ici, on est déjà en kg
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    
    # Chargement
    X, y = load_data_raw()
    
    # Lancement étude
    print(f"\n>>> DÉBUT OPTIMISATION OPTUNA (RAW) <<<")
    study = optuna.create_study(direction='minimize', study_name="fuel_raw_opt")
    
    try:
        study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nArrêt utilisateur. Sauvegarde du meilleur résultat...")

    # Résultats
    print("\n>>> RÉSULTATS <<<")
    print(f"Meilleur RMSE (Raw) : {study.best_value:.4f}")
    print("Meilleurs Paramètres :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Sauvegarde
    best_params = study.best_params.copy()
    best_params.update({
        'n_estimators': 5000,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'enable_categorical': True,
        'tree_method': 'hist',
        'device': 'cpu',
        # On retire early_stopping_rounds pour la sauvegarde finale (usage futur en fit complet)
    })
    
    joblib.dump(best_params, OUTPUT_FILE)
    print(f"\n[Sauvegarde] Paramètres sauvés dans : {OUTPUT_FILE}")
    print("Vous pouvez maintenant lancer 'generate_submission_ensemble.py' (pensez à mettre à jour le chemin du pkl !)")