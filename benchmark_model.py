import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
PATH_DATASET = "data/processed/X_train_final.parquet"
N_FOLDS = 5
RANDOM_STATE = 42

def load_data():
    print(">>> CHARGEMENT DES DONNÉES <<<")
    df = pd.read_parquet(PATH_DATASET)
    
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg']
    features = [c for c in df.columns if c not in cols_exclude]
    target = 'fuel_kg'
    
    X = df[features]
    y = df[target]
    
    cat_cols = [c for c in X.columns if X[c].dtype.name == 'category']
    print(f"Features ({len(features)}) : {features}")
    print(f"Features Catégorielles : {cat_cols}")
    
    return X, y, cat_cols

def score_model(model_name, model_obj, X, y, cat_features=None):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    times = []
    
    print(f"\n--- Test de {model_name} ---")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
        
        start_time = time.time()
        
        # --- GESTION SPÉCIFIQUE PAR LIBRAIRIE ---
        
        if model_name == "RandomForest":
            # Scikit-Learn ne gère pas le type 'category' natif -> on convertit en codes entiers
            for col in cat_features:
                X_train[col] = X_train[col].cat.codes
                X_val[col] = X_val[col].cat.codes
            # Pas d'early stopping pour RF standard, on fit directement
            model_obj.fit(X_train, y_train)

        elif model_name == "CatBoost":
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            model_obj.fit(train_pool, eval_set=val_pool, verbose=False, early_stopping_rounds=50)
            
        elif model_name == "LightGBM":
            model_obj.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
        else: # XGBoost
            model_obj.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Prédiction
        if model_name == "RandomForest":
            # Pour RF, les features doivent avoir le même encodage
            preds = model_obj.predict(X_val)
        else:
            preds = model_obj.predict(X_val)
            
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
        print(f"   Fold {fold+1} RMSE: {rmse:.2f} (Temps: {elapsed:.2f}s)")
        
    mean_rmse = np.mean(rmse_scores)
    mean_time = np.mean(times)
    
    return mean_rmse, mean_time

# ==========================================
# LANCEUR
# ==========================================
if __name__ == "__main__":
    X, y, cat_cols = load_data()
    
    results = []

    # 1. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=8, 
        subsample=0.8, colsample_bytree=0.8, 
        enable_categorical=True, tree_method='hist', 
        early_stopping_rounds=50, n_jobs=-1
    )
    results.append(("XGBoost", *score_model("XGBoost", xgb_model, X, y)))

    # 2. LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1
    )
    results.append(("LightGBM", *score_model("LightGBM", lgb_model, X, y)))

    # 3. CatBoost
    cb_model = CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=8,
        loss_function='RMSE', verbose=False, thread_count=-1,
        allow_writing_files=False
    )
    results.append(("CatBoost", *score_model("CatBoost", cb_model, X, y, cat_features=cat_cols)))

    # 4. Random Forest (Attention, c'est souvent plus lent !)
    # On limite n_estimators et max_depth pour que le benchmark ne dure pas 1 heure
    rf_model = RandomForestRegressor(
        n_estimators=100,  # Moins d'arbres que le boosting sinon c'est trop lent
        max_depth=15,      # Profondeur limitée pour éviter l'explosion RAM
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    results.append(("RandomForest", *score_model("RandomForest", rf_model, X, y, cat_features=cat_cols)))

    # --- CLASSEMENT FINAL ---
    print("\n>>> PODIUM FINAL <<<")
    results.sort(key=lambda x: x[1]) # Tri par RMSE croissant
    
    print(f"{'MODÈLE':<15} | {'RMSE MOYEN':<15} | {'TEMPS/FOLD (s)':<15}")
    print("-" * 50)
    for name, rmse, t in results:
        print(f"{name:<15} | {rmse:<15.4f} | {t:<15.2f}")