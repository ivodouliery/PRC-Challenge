import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Chemins des fichiers (Basés sur votre arborescence validée)
PATH_TRAIN_FEATURES = "data/processed/X_train_final.parquet"
PATH_TEST_FEATURES  = "data/processed/X_test_final.parquet"
PATH_PARAMS         = "models/best_params_log_optuna.pkl" # Attention : on prend les params optimisés pour le LOG
PATH_SUBMISSION_TEMPLATE = "data/raw/fuel_rank_submission.parquet"
OUTPUT_FILE         = "submission.parquet"

def generate_submission_log():
    print(">>> GÉNÉRATION DE LA SOUMISSION (STRATÉGIE LOG-TRANSFORM) <<<")

    # -------------------------------------------------------
    # 1. CHARGEMENT DES DONNÉES
    # -------------------------------------------------------
    print("[1/6] Chargement des datasets...")
    try:
        df_train = pd.read_parquet(PATH_TRAIN_FEATURES)
        df_test_feat = pd.read_parquet(PATH_TEST_FEATURES)
        # On charge le template officiel pour garantir le format de sortie
        df_final = pd.read_parquet(PATH_SUBMISSION_TEMPLATE)
    except FileNotFoundError as e:
        print(f"ERREUR CRITIQUE : {e}")
        print("Vérifiez que vous avez bien lancé le feature engineering sur le TRAIN et le TEST.")
        return

    # -------------------------------------------------------
    # 2. PRÉPARATION DU TRAIN
    # -------------------------------------------------------
    print("[2/6] Préparation des données d'entraînement...")
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg']
    features = [c for c in df_train.columns if c not in cols_exclude]
    
    X_train = df_train[features]
    y_train = df_train['fuel_kg']
    
    # -------------------------------------------------------
    # 3. ALIGNEMENT DU TEST (MERGE)
    # -------------------------------------------------------
    print("[3/6] Alignement et Imputation du Test Set...")
    
    # Conversion explicite des dates pour éviter les échecs de jointure
    df_final['start'] = pd.to_datetime(df_final['start'])
    df_final['end'] = pd.to_datetime(df_final['end'])
    
    # Jointure Gauche : On garde 100% des lignes du template de soumission
    # On y colle nos features calculées
    df_predict = df_final.merge(
        df_test_feat, 
        on=['flight_id', 'start', 'end'], 
        how='left'
    )
    
    # Gestion des segments sans features (Cas des vols fantômes ou filtrés)
    missing_mask = df_predict['duration'].isna()
    if missing_mask.sum() > 0:
        print(f"   /!\\ ATTENTION : {missing_mask.sum()} segments n'ont pas de features.")
        print("   -> Remplissage par Médiane/Mode pour éviter le crash.")
        
        for col in features:
            if df_train[col].dtype.name == 'category':
                # Pour le type avion, on prend le plus fréquent
                most_frequent = df_train[col].mode()[0]
                df_predict[col] = df_predict[col].fillna(most_frequent)
            else:
                # Pour les chiffres, on prend la médiane globale
                median_val = df_train[col].median()
                df_predict[col] = df_predict[col].fillna(median_val)

    X_test = df_predict[features]
    
    # Sécurité : On force l'ordre des colonnes pour qu'il soit identique au Train
    X_test = X_test[X_train.columns]

    # -------------------------------------------------------
    # 4. CHARGEMENT DU MODÈLE ET TRANSFORMATION CIBLE
    # -------------------------------------------------------
    print("[4/6] Configuration du modèle...")
    if not os.path.exists(PATH_PARAMS):
        raise FileNotFoundError(f"Fichier de paramètres introuvable : {PATH_PARAMS}")
    
    best_params = joblib.load(PATH_PARAMS)
    
    # Nettoyage : On retire 'early_stopping_rounds' car on entraîne sur tout le dataset sans eval_set
    if 'early_stopping_rounds' in best_params:
        del best_params['early_stopping_rounds']
        
    print(f"   Learning Rate: {best_params.get('learning_rate', 'N/A')}")
    print(f"   Max Depth: {best_params.get('max_depth', 'N/A')}")

    # --- TRANSFORMATION LOGARITHMIQUE ---
    print("   -> Application de log1p sur la cible (y_train)...")
    y_train_log = np.log1p(y_train)

    # -------------------------------------------------------
    # 5. ENTRAÎNEMENT FINAL
    # -------------------------------------------------------
    print("[5/6] Entraînement sur 100% du Dataset Train...")
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train_log)
    
    # -------------------------------------------------------
    # 6. PRÉDICTION ET RETOUR À L'ÉCHELLE
    # -------------------------------------------------------
    print("[6/6] Prédiction et Sauvegarde...")
    
    # Prédiction (L'échelle est logarithmique ici)
    preds_log = model.predict(X_test)
    
    # --- TRANSFORMATION INVERSE (EXPONENTIELLE) ---
    print("   -> Application de expm1 pour revenir en kg...")
    predictions = np.expm1(preds_log)
    
    # Post-traitement : Pas de consommation négative
    predictions = np.maximum(predictions, 0)
    
    # Injection dans le template original (respect strict du format)
    df_final['fuel_kg'] = predictions
    
    # Sauvegarde Parquet
    df_final.to_parquet(OUTPUT_FILE)
    
    print(f"\n>>> SUCCÈS <<<")
    print(f"Fichier généré : {OUTPUT_FILE}")
    print(f"Moyenne Fuel prédite : {predictions.mean():.2f} kg")
    print(f"Aperçu :\n{df_final[['flight_id', 'fuel_kg']].head()}")

if __name__ == "__main__":
    generate_submission_log()