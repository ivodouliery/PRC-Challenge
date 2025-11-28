import pandas as pd
import numpy as np
import os
import sys

def load_and_preprocess_data(path, is_train=True, subsample_ratio=1.0):
    """
    Charge et nettoie les données pour l'entraînement et l'inférence.
    """
    if not os.path.exists(path):
        print(f"ERREUR: {path} introuvable.")
        sys.exit(1)
        
    print(f"Chargement de {path}...")
    df = pd.read_parquet(path)
    
    # --- 1. GESTION DES SEGMENTS VIDES (GHOSTS) ---
    # Calculer la durée théorique d'après les timestamps
    # Assurons-nous que start/end sont des datetimes
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    
    theoretical_duration = (df['end'] - df['start']).dt.total_seconds()
    
    # Si duration_sec est NaN ou 0, on prend la théorique
    # On ajoute une feature pour dire "Attention, c'était vide"
    df['is_missing_data'] = (df['n_points'] == 0) | (df['duration_sec'].isna()) | (df['duration_sec'] < 1)
    df['is_missing_data'] = df['is_missing_data'].astype(int)
    
    # Remplacer les durées nulles/manquantes
    mask_bad_duration = (df['duration_sec'].isna()) | (df['duration_sec'] < 0.1)
    df.loc[mask_bad_duration, 'duration_sec'] = theoretical_duration[mask_bad_duration]

    # --- 1.5. AJOUT DE FEATURES TEMPORELLES (CONTEXTE) ---
    print("⏳ Ajout du contexte temporel (Lag/Lead/Cumul)...")
    
    # On s'assure que c'est trié
    df = df.sort_values(['flight_id', 'idx'])
    
    # Groupby pour les opérations vectorisées
    g = df.groupby('flight_id')
    
    # 1. Cumulatifs (Depuis le début du vol)
    df['time_since_takeoff'] = g['duration_sec'].cumsum() - df['duration_sec'] # Temps écoulé AVANT ce segment
    if 'distance_km' in df.columns:
        df['dist_flown'] = g['distance_km'].cumsum() - df['distance_km']
    
    # 2. Lag Features (Ce qui s'est passé juste avant)
    # On décale de 1 vers le bas
    cols_to_lag = ['alt_mean', 'gs_mean', 'vrate_mean', 'phase']
    for col in cols_to_lag:
        if col in df.columns:
            df[f'prev_{col}'] = g[col].shift(1)
            
    # 3. Lead Features (Ce qui va se passer juste après - Anticipation)
    # On décale de -1 (vers le haut)
    if 'phase' in df.columns:
        df['next_phase'] = g['phase'].shift(-1)
        
    # Remplissage des NaNs créés par le shift (début/fin de vol)
    # Pour les numériques, on met -1 ou 0
    num_lag_cols = [c for c in df.columns if c.startswith('prev_') and df[c].dtype.kind in 'ifc']
    df[num_lag_cols] = df[num_lag_cols].fillna(0)
    
    # Pour les cat, on met 'NA'
    cat_lag_cols = ['prev_phase', 'next_phase']
    for c in cat_lag_cols:
        if c in df.columns:
            df[c] = df[c].fillna('NA').astype(str)

    # --- 2. FILTRAGE DES OUTLIERS (TRAIN ONLY) ---
    if is_train:
        # Calcul de la consommation par seconde (kg/s)
        # Un A380 consomme max ~4 kg/s (14t/h). Prenons une marge large (20 kg/s)
        # Pour éviter de diviser par 0, on clip la durée
        safe_duration = df['duration_sec'].clip(lower=1.0)
        consumption_rate = df['fuel_kg'] / safe_duration
        
        # Seuil physique : 20 kg/s (72 tonnes/heure) -> Impossible même pour un A380 en montée
        # On ne filtre que si fuel_kg > 10 (pour ne pas virer les petits segments bruyants)
        mask_outlier = (consumption_rate > 20) & (df['fuel_kg'] > 10)
        n_outliers = mask_outlier.sum()
        
        if n_outliers > 0:
            print(f"🧹 NETTOYAGE: Suppression de {n_outliers} outliers physiques (>20 kg/s)")
            df = df[~mask_outlier].reset_index(drop=True)

    # --- 3. SUBSAMPLING (OPTIONNEL) ---
    if is_train and subsample_ratio < 1.0:
        flight_ids = df['flight_id'].unique()
        n_keep = int(len(flight_ids) * subsample_ratio)
        selected_flights = np.random.choice(flight_ids, size=n_keep, replace=False)
        df = df[df['flight_id'].isin(selected_flights)].copy()
        print(f"⚠️ SUBSAMPLING: Dataset réduit à {subsample_ratio*100}% ({len(df)} lignes)")

    # --- 4. GESTION DES CATÉGORIES ---
    cat_cols_names = ['aircraft_type', 'fuel_source', 'mass_source', 'origin_icao', 'destination_icao', 'phase', 'typecode', 'prev_phase', 'next_phase']
    for c in cat_cols_names:
        if c in df.columns:
            df[c] = df[c].fillna('UNKNOWN').astype(str).astype('category')
            
    # --- 5. SÉLECTION DES FEATURES ---
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg', 'flight_date', 'takeoff', 'landed', 'origin_name', 'destination_name', 'timestamp_start', 'timestamp_end', 'idx']
    features = [c for c in df.columns if c not in cols_exclude]
    
    X = df[features]
    y = df['fuel_kg'] if 'fuel_kg' in df.columns else None
    groups = df['flight_id']
    
    cat_cols = [c for c in X.columns if X[c].dtype.name == 'category']
    
    return X, y, groups, cat_cols, df
