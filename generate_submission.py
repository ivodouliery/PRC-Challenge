import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
PATH_TRAIN_FEATURES = "data/processed/X_train_final.parquet"
PATH_TEST_FEATURES  = "data/processed/X_test_final.parquet"
PATH_PARAMS         = "models/best_params_optuna.pkl"
PATH_SUBMISSION_TEMPLATE = "data/raw/fuel_rank_submission.parquet"
OUTPUT_FILE         = "submission.parquet"

def generate_submission():
    print(">>> GÉNÉRATION DE LA SOUMISSION (STRICT TEMPLATE) <<<")

    # 1. Chargement
    print("[1/5] Chargement des données...")
    try:
        df_train = pd.read_parquet(PATH_TRAIN_FEATURES)
        df_test_feat = pd.read_parquet(PATH_TEST_FEATURES)
        # On charge le template qui servira de base au fichier final
        df_final = pd.read_parquet(PATH_SUBMISSION_TEMPLATE)
    except FileNotFoundError as e:
        print(f"ERREUR : {e}")
        return

    # 2. Préparation Features
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg']
    features = [c for c in df_train.columns if c not in cols_exclude]
    
    X_train = df_train[features]
    y_train = df_train['fuel_kg']
    
    # 3. Alignement Test
    # On s'assure que les dates sont au bon format pour la jointure
    df_final['start'] = pd.to_datetime(df_final['start'])
    df_final['end'] = pd.to_datetime(df_final['end'])
    
    print("[2/5] Alignement des features...")
    # On merge le template avec nos features calculées
    # On utilise un DataFrame temporaire pour la prédiction
    df_predict = df_final.merge(
        df_test_feat, 
        on=['flight_id', 'start', 'end'], 
        how='left'
    )
    
    # Gestion des manquants (Imputation robuste)
    missing_mask = df_predict['duration'].isna()
    if missing_mask.sum() > 0:
        print(f"   /!\\ Remplissage de {missing_mask.sum()} segments manquants.")
        for col in features:
            if df_train[col].dtype.name == 'category':
                df_predict[col] = df_predict[col].fillna(df_train[col].mode()[0])
            else:
                df_predict[col] = df_predict[col].fillna(df_train[col].median())

    X_test = df_predict[features]
    # Sécurité : Ordre des colonnes identique au train
    X_test = X_test[X_train.columns]

    # 4. Modèle
    print("[3/5] Chargement et Entraînement...")
    if not os.path.exists(PATH_PARAMS):
        raise FileNotFoundError("Paramètres introuvables.")
    
    best_params = joblib.load(PATH_PARAMS)
    if 'early_stopping_rounds' in best_params:
        del best_params['early_stopping_rounds']

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    
    # 5. Prédiction
    print("[4/5] Calcul des prédictions...")
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0) # Pas de fuel négatif
    
    # 6. Remplissage et Sauvegarde
    print("[5/5] Insertion dans le template...")
    
    # On injecte les prédictions directement dans le DataFrame original
    # Cela préserve toutes les autres colonnes (idx, flight_id, dates...)
    df_final['fuel_kg'] = preds
    
    # Vérification visuelle
    print("\nAperçu du fichier final :")
    print(df_final.head())
    
    df_final.to_parquet(OUTPUT_FILE)
    print(f"\n>>> SUCCÈS : Fichier '{OUTPUT_FILE}' généré. <<<")

if __name__ == "__main__":
    generate_submission()