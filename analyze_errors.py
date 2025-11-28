import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

# ==========================================
# CONFIG
# ==========================================
PATH_DATASET = "data/features_train.parquet"
MODELS_DIR = "models"
N_FOLDS = 5

from data_utils import load_and_preprocess_data

def load_data():
    # On utilise le loader centralisé qui corrige les durées nulles
    # load_and_preprocess_data retourne: X, y, groups, cat_cols, df
    # On veut: X, y, groups, df
    X, y, groups, _, df = load_and_preprocess_data(PATH_DATASET, is_train=True, subsample_ratio=1.0)
    return X, y, groups, df

def analyze_errors():
    # 1. Charger les données
    X, y, groups, df_full = load_data()
    
    # 2. Charger les meilleurs params XGBoost (ou défaut)
    param_path = f"{MODELS_DIR}/best_params_xgb_night.pkl"
    if os.path.exists(param_path):
        print(f"Chargement des paramètres depuis {param_path}")
        params = joblib.load(param_path)
    else:
        print("⚠️ Pas de paramètres trouvés, utilisation des défauts.")
        params = {
            'n_estimators': 1000, 'objective': 'reg:squarederror', 'n_jobs': -1, 
            'tree_method': 'hist', 'enable_categorical': True, 'max_depth': 8, 'learning_rate': 0.05
        }
    
    # 3. Générer les prédictions OOF (Out-Of-Fold)
    print(f"\nCalcul des prédictions OOF (Cross-Validation {N_FOLDS} folds)...")
    kf = GroupKFold(n_splits=N_FOLDS)
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        oof_preds[val_idx] = model.predict(X_val)
        print(f"  Fold {fold+1}/{N_FOLDS} terminé.")
        
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\n✅ RMSE Global: {rmse:.4f}")
    
    # 4. Analyse des erreurs
    df_full['pred'] = oof_preds
    df_full['error'] = df_full['pred'] - df_full['fuel_kg']
    df_full['abs_error'] = df_full['error'].abs()
    df_full['rel_error_pct'] = (df_full['abs_error'] / df_full['fuel_kg']) * 100
    
    print("\n" + "="*50)
    print("🏆 TOP 20 DES PIRES ERREURS (Absolues)")
    print("="*50)
    # On affiche les colonnes pertinentes
    cols_show = ['flight_id', 'typecode', 'phase', 'duration_sec', 'fuel_kg', 'pred', 'error', 'rel_error_pct']
    print(df_full.sort_values('abs_error', ascending=False)[cols_show].head(20).to_string(index=False))
    
    print("\n" + "="*50)
    print("📊 ERREUR MOYENNE PAR TYPE D'AVION")
    print("="*50)
    print(df_full.groupby('typecode')['abs_error'].agg(['count', 'mean', 'median', 'max']).sort_values('mean', ascending=False))

    print("\n" + "="*50)
    print("📊 ERREUR MOYENNE PAR PHASE DE VOL")
    print("="*50)
    if 'phase' in df_full.columns:
        print(df_full.groupby('phase')['abs_error'].agg(['count', 'mean', 'median']).sort_values('mean', ascending=False))

    # Sauvegarde pour analyse approfondie
    df_full.sort_values('abs_error', ascending=False).to_csv("analysis_errors_detailed.csv", index=False)
    print("\n💾 Détails sauvegardés dans 'analysis_errors_detailed.csv'")

if __name__ == "__main__":
    analyze_errors()
