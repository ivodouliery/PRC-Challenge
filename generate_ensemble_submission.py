import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
PATH_TRAIN = "data/processed/X_train_final.parquet"
PATH_TEST  = "data/processed/X_test_final.parquet"
PATH_SUBMISSION_TEMPLATE = "data/raw/fuel_rank_submission.parquet"
PATH_XGB_PARAMS = "models/best_params_optuna.pkl"
OUTPUT_FILE = "submission_ensemble.parquet"

# POIDS AJUSTÉS SELON VOTRE BENCHMARK
# XGBoost domine (244 RMSE), on lui donne le lead.
WEIGHTS = {
    'xgboost': 0.70, 
    'lightgbm': 0.20, 
    'catboost': 0.10 
}

def generate_ensemble():
    print(f">>> GÉNÉRATION ENSEMBLE OPTIMISÉ ({WEIGHTS}) <<<")

    # 1. Chargement
    print("[1/6] Chargement des données...")
    df_train = pd.read_parquet(PATH_TRAIN)
    df_test_feat = pd.read_parquet(PATH_TEST)
    df_submission = pd.read_parquet(PATH_SUBMISSION_TEMPLATE)

    # Alignement Temporel pour la jointure
    df_submission['start'] = pd.to_datetime(df_submission['start'])
    df_submission['end'] = pd.to_datetime(df_submission['end'])

    # Préparation Train
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg']
    features = [c for c in df_train.columns if c not in cols_exclude]
    X_train = df_train[features]
    y_train = df_train['fuel_kg']

    # Préparation Test (Merge)
    print("[2/6] Alignement des features Test...")
    df_predict = df_submission.merge(
        df_test_feat, on=['flight_id', 'start', 'end'], how='left'
    )
    
    # Imputation des manquants (Médiane/Mode)
    for col in features:
        if df_train[col].dtype.name == 'category':
            # Mode pour les catégories
            fill_val = df_train[col].mode()[0]
            df_predict[col] = df_predict[col].fillna(fill_val)
        else:
            # Médiane pour le numérique
            fill_val = df_train[col].median()
            df_predict[col] = df_predict[col].fillna(fill_val)
            
    X_test = df_predict[features]
    X_test = X_test[X_train.columns] # Ordre strict des colonnes

    # Identification catégories
    cat_features = [c for c in features if X_train[c].dtype.name == 'category']

    # ==========================================
    # 1. XGBOOST (Le Champion)
    # ==========================================
    print(f"\n[3/6] Entraînement XGBoost (Poids: {WEIGHTS['xgboost']})...")
    if os.path.exists(PATH_XGB_PARAMS):
        xgb_params = joblib.load(PATH_XGB_PARAMS)
        if 'early_stopping_rounds' in xgb_params: del xgb_params['early_stopping_rounds']
    else:
        print("Attention: Params Optuna introuvables, utilisation défaut.")
        xgb_params = {'n_estimators': 2000, 'n_jobs': -1, 'tree_method': 'hist', 'enable_categorical': True}

    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)

    # ==========================================
    # 2. LIGHTGBM (Le Challenger)
    # ==========================================
    print(f"\n[4/6] Entraînement LightGBM (Poids: {WEIGHTS['lightgbm']})...")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(X_train, y_train)
    pred_lgb = model_lgb.predict(X_test)

    # ==========================================
    # 3. CATBOOST (La Sécurité)
    # ==========================================
    print(f"\n[5/6] Entraînement CatBoost (Poids: {WEIGHTS['catboost']})...")
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    
    model_cb = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function='RMSE',
        verbose=0,
        allow_writing_files=False,
        thread_count=-1
    )
    model_cb.fit(train_pool)
    pred_cb = model_cb.predict(X_test)

    # ==========================================
    # FUSION ET EXPORT
    # ==========================================
    print("\n[6/6] Mélange pondéré et Sauvegarde...")
    final_pred = (
        pred_xgb * WEIGHTS['xgboost'] +
        pred_lgb * WEIGHTS['lightgbm'] +
        pred_cb  * WEIGHTS['catboost']
    )
    
    # Pas de fuel négatif
    final_pred = np.maximum(final_pred, 0)

    # Injection dans le template original
    df_submission['fuel_kg'] = final_pred
    
    df_submission.to_parquet(OUTPUT_FILE)
    print(f"\n>>> TERMINE : {OUTPUT_FILE} <<<")
    print(f"Moyenne Fuel : {final_pred.mean():.2f} kg")

if __name__ == "__main__":
    generate_ensemble()