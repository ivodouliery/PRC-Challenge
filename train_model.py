import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os

# ==========================================
# CONFIGURATION (Mise à jour selon votre arborescence)
# ==========================================
# Chemin vers le fichier de features généré précédemment
PATH_DATASET = "data/processed/X_train_final.parquet"

# Dossier de sortie pour les meilleurs paramètres
OUTPUT_FOLDER = "models"
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, "best_params_optuna.pkl")

# Paramètres de l'étude
N_TRIALS = 50  # Nombre d'essais
N_FOLDS = 3    # K-Fold

def load_data():
    print(">>> CHARGEMENT DES DONNÉES <<<")
    if not os.path.exists(PATH_DATASET):
        raise FileNotFoundError(f"ERREUR: Le fichier {PATH_DATASET} est introuvable. Avez-vous déplacé le fichier généré ?")
        
    df = pd.read_parquet(PATH_DATASET)
    
    # Séparation Features / Cible
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg']
    features = [c for c in df.columns if c not in cols_exclude]
    target = 'fuel_kg'
    
    X = df[features]
    y = df[target]
    
    print(f"Fichier : {PATH_DATASET}")
    print(f"Volume : {len(df)} segments.")
    print(f"Features ({len(features)}) : {features}")
    return X, y

# ==========================================
# FONCTION OBJECTIVE
# ==========================================
def objective(trial, X, y):
    # 1. Paramètres Fixes
    params = {
        'n_estimators': 5000,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'device': 'cpu',
        'verbosity': 0
    }
    
    # 2. Paramètres à Optimiser
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
    params['max_depth'] = trial.suggest_int('max_depth', 5, 12)
    params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 15)
    params['subsample'] = trial.suggest_float('subsample', 0.6, 0.95)
    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 0.95)
    params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)

    # 3. Cross-Validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rmse_scores = []

    # Importation sécurisée du callback
    try:
        from optuna.integration import XGBoostPruningCallback
    except ImportError:
        from optuna.pruners import XGBoostPruningCallback

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # --- ASTUCE EXPERT ---
        # On n'active le pruning QUE sur le premier fold (fold_idx == 0)
        # Cela évite les conflits de steps et les warnings, tout en gardant l'efficacité.
        current_callbacks = [pruning_callback] if fold_idx == 0 else []

        model = xgb.XGBRegressor(**params, callbacks=current_callbacks)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# ==========================================
# LANCEUR
# ==========================================
if __name__ == "__main__":
    # Vérification du dossier de sortie
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Dossier '{OUTPUT_FOLDER}' créé.")

    # Chargement
    X, y = load_data()

    # Optimisation
    print(f"\n>>> DÉBUT DE L'OPTIMISATION OPTUNA ({N_TRIALS} Essais) <<<")
    study = optuna.create_study(direction='minimize', study_name="fuel_xgb_opt")
    
    # Utilisation de lambda pour passer X et y
    func = lambda trial: objective(trial, X, y)
    
    try:
        study.optimize(func, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nArrêt utilisateur ! Sauvegarde du meilleur résultat actuel...")

    # Résultats
    print("\n>>> RÉSULTATS OPTUNA <<<")
    print(f"Meilleur RMSE : {study.best_value:.4f}")
    print("Meilleurs Paramètres :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Sauvegarde
    best_params = study.best_params.copy()
    # On rajoute les paramètres fixes indispensables
    best_params.update({
        'n_estimators': 5000,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'device': 'cpu'
    })
    
    joblib.dump(best_params, OUTPUT_FILE)
    print(f"\n[Sauvegarde] Les paramètres optimisés sont dans : {OUTPUT_FILE}")
    print("Prêt pour l'entraînement final.")