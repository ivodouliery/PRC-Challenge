import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os
import sys
import multiprocessing
import time

# Fix multiprocessing on macOS
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# ==========================================
# CONFIGURATION
# ==========================================
# ==========================================
# CONFIGURATION
# ==========================================
PATH_DATASET = "data/features_train.parquet"
OUTPUT_FOLDER = "models"

# Bases de données isolées (Crucial pour le multiprocessing)
DB_XGB = "sqlite:///optuna_xgb.db"
DB_LGB = "sqlite:///optuna_lgb.db"
DB_CAT = "sqlite:///optuna_cat.db"

N_FOLDS = 5

# ==========================================
# CHARGEMENT DONNÉES
# ==========================================
from data_utils import load_and_preprocess_data

# ==========================================
# CHARGEMENT DONNÉES
# ==========================================
def load_data():
    # Wrapper pour garder la compatibilité avec le code existant
    # On passe subsample_ratio=1.0 car l'optimisation "Infinity" se veut précise
    # Mais on peut le changer ici si on veut aller vite
    return load_and_preprocess_data(PATH_DATASET, is_train=True, subsample_ratio=1.0)[:4]

# ==========================================
# FONCTIONS OBJECTIVES (IDENTIQUES)
# ==========================================
def objective_xgb(trial, X, y, groups):
    params = {
        'n_estimators': 5000,
        'objective': 'reg:squarederror',
        'n_jobs': 1, # Important: 1 coeur par modèle pour ne pas surcharger le CPU global
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'device': 'cpu',
        'verbosity': 0,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 14),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100.0, log=True),
    }
    
    from sklearn.model_selection import GroupKFold
    kf = GroupKFold(n_splits=N_FOLDS)
    rmse_scores = []
    
    # Callbacks silencieux
    try:
        from optuna.integration import XGBoostPruningCallback
        pruning = XGBoostPruningCallback(trial, "validation_0-rmse")
    except:
        pruning = None

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        cbs = [pruning] if pruning and fold_idx == 0 else []
        model = xgb.XGBRegressor(**params, callbacks=cbs)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    return np.mean(rmse_scores)

def objective_lgb(trial, X, y, groups):
    params = {
        'n_estimators': 5000,
        'metric': 'rmse',
        'n_jobs': 1,
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100.0, log=True),
    }
    
    from sklearn.model_selection import GroupKFold
    kf = GroupKFold(n_splits=N_FOLDS)
    rmse_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    return np.mean(rmse_scores)

def objective_cat(trial, X, y, groups, cat_cols):
    params = {
        'iterations': 3000,
        'loss_function': 'RMSE',
        'task_type': 'CPU',
        'thread_count': 2, # CatBoost gère bien un peu de multi-thread
        'verbose': False,
        'allow_writing_files': False,
        'early_stopping_rounds': 50,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
    }

    from sklearn.model_selection import GroupKFold
    kf = GroupKFold(n_splits=3)
    rmse_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool)
        preds = model.predict(val_pool)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        
        trial.report(np.sqrt(mean_squared_error(y_val, preds)), fold_idx)
        if trial.should_prune(): raise optuna.TrialPruned()

    return np.mean(rmse_scores)

# ==========================================
# WORKERS (PROCESSUS INDÉPENDANTS)
# ==========================================

def run_worker_xgb():
    print("[XGBoost Worker] Démarré.")
    X, y, groups, _ = load_data()
    study = optuna.create_study(direction='minimize', study_name="xgb_night", storage=DB_XGB, load_if_exists=True)
    
    while len(study.trials) < 50:
        try:
            n_batch = 5
            study.optimize(lambda t: objective_xgb(t, X, y, groups), n_trials=n_batch)
            
            # Sauvegarde régulière
            best = study.best_params.copy()
            best.update({'n_estimators': 5000, 'objective': 'reg:squarederror', 'n_jobs': -1, 'tree_method': 'hist', 'enable_categorical': True})
            if 'early_stopping_rounds' in best: del best['early_stopping_rounds']
            joblib.dump(best, "models/best_params_xgb_night.pkl")
            print(f"[XGBoost] Trials: {len(study.trials)}/50 - Best RMSE: {study.best_value:.4f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[XGBoost Error] {e}")
            time.sleep(5)
    print("[XGBoost] Terminé (50 itérations).")

def run_worker_lgb():
    print("[LightGBM Worker] Démarré.")
    X, y, groups, _ = load_data()
    study = optuna.create_study(direction='minimize', study_name="lgb_night", storage=DB_LGB, load_if_exists=True)
    
    while len(study.trials) < 50:
        try:
            n_batch = 5
            study.optimize(lambda t: objective_lgb(t, X, y, groups), n_trials=n_batch)
            
            best = study.best_params.copy()
            best.update({'n_estimators': 5000, 'metric': 'rmse', 'n_jobs': -1, 'verbosity': -1})
            joblib.dump(best, "models/best_params_lgb_night.pkl")
            print(f"[LightGBM] Trials: {len(study.trials)}/50 - Best RMSE: {study.best_value:.4f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[LightGBM Error] {e}")
            time.sleep(5)
    print("[LightGBM] Terminé (50 itérations).")

def run_worker_cat():
    print("[CatBoost Worker] Démarré.")
    X, y, groups, cat_cols = load_data()
    study = optuna.create_study(direction='minimize', study_name="cat_night", storage=DB_CAT, load_if_exists=True)
    
    while len(study.trials) < 50:
        try:
            n_batch = 5
            study.optimize(lambda t: objective_cat(t, X, y, groups, cat_cols), n_trials=n_batch)
            
            best = study.best_params.copy()
            best.update({'iterations': 3000, 'loss_function': 'RMSE', 'verbose': 0, 'allow_writing_files': False})
            joblib.dump(best, "models/best_params_cat_night.pkl")
            print(f"[CatBoost] Trials: {len(study.trials)}/50 - Best RMSE: {study.best_value:.4f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[CatBoost Error] {e}")
            time.sleep(5)
    print("[CatBoost] Terminé (50 itérations).")

# ==========================================
# ORCHESTRATEUR
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    print(f">>> LANCEMENT DES 3 WORKERS EN PARALLÈLE <<<")
    print("Pour arrêter : CTRL + C (Il faudra peut-être insister ou fermer le terminal)")

    # Création des processus
    p_xgb = multiprocessing.Process(target=run_worker_xgb)
    p_lgb = multiprocessing.Process(target=run_worker_lgb)
    p_cat = multiprocessing.Process(target=run_worker_cat)

    # Démarrage
    p_xgb.start()
    p_lgb.start()
    p_cat.start()

    try:
        p_xgb.join()
        p_lgb.join()
        p_cat.join()
    except KeyboardInterrupt:
        print("\n>>> ARRÊT DEMANDÉ <<<")
        p_xgb.terminate()
        p_lgb.terminate()
        p_cat.terminate()
        print("Processus arrêtés. Vos fichiers .pkl sont dans 'models/'.")