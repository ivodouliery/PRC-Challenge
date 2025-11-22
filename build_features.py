import pandas as pd
import numpy as np
import glob
import os
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm  # <--- NOUVEL IMPORT

# ==========================================
# CONFIGURATION (MODE SOUMISSION)
# ==========================================
PATH_PRC_FOLDER = "data/raw/flights_rank"      # ATTENTION : Vérifiez le nom du dossier contenant les prc de test
PATH_FLIGHTLIST = "data/raw/flightlist_rank.parquet"
PATH_FUEL_TARGET = "data/raw/fuel_rank_submission.parquet" # La cible à prédire
PATH_OUTPUT     = "data/processed/X_test_final.parquet"    # Le fichier de sortie
N_CORES         = 8

# ==========================================
# 1. FONCTIONS UTILITAIRES
# ==========================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calcul vectorisé de la distance orthodromique (km)."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    a = np.clip(a, 0, 1)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# ==========================================
# 2. WORKER (Sans print pour garder la barre propre)
# ==========================================

def process_single_file(f_path, df_target, flight_ac_map):
    local_features = []
    fname = os.path.basename(f_path)
    
    try:
        # A. Lecture & Nettoyage
        df = pd.read_parquet(f_path)
        
        mask_physic = (
            (df['groundspeed'] >= 0) & (df['groundspeed'] <= 1200) &
            (df['altitude'] >= -1000) & (df['altitude'] <= 60000) &
            (df['latitude'].notna()) & (df['longitude'].notna())
        )
        df = df[mask_physic].copy()

        if df.empty: return []

        # B. Déduplication (MÉDIANE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cols_to_agg = ['latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate']
        for col in ['mach', 'TAS', 'CAS']:
            if col in df.columns: cols_to_agg.append(col)
            
        df = df.groupby(['flight_id', 'timestamp'])[cols_to_agg].median().reset_index()
        df = df.sort_values(['flight_id', 'timestamp'])

        # C. Calculs & Optimisation
        for c in cols_to_agg:
            if c in df.columns: df[c] = df[c].astype(np.float32)

        df['prev_lat'] = df.groupby('flight_id')['latitude'].shift(1)
        df['prev_lon'] = df.groupby('flight_id')['longitude'].shift(1)
        
        df['step_dist'] = haversine_vectorized(
            df['latitude'], df['longitude'], 
            df['prev_lat'], df['prev_lon']
        ).fillna(0).astype(np.float32)
        
        track_rad = np.radians(df['track'].fillna(0))
        df['sin_track'] = np.sin(track_rad).astype(np.float32)
        df['cos_track'] = np.cos(track_rad).astype(np.float32)

        # D. Windowing
        relevant_ids = df['flight_id'].unique()
        sub_target = df_target[df_target['flight_id'].isin(relevant_ids)]
        
        if not sub_target.empty:
            prc_grouped = df.groupby('flight_id')
            
            for idx, row in sub_target.iterrows():
                fid = row['flight_id']
                try:
                    flight_data = prc_grouped.get_group(fid)
                    mask_time = (flight_data['timestamp'] >= row['start']) & (flight_data['timestamp'] <= row['end'])
                    seg = flight_data[mask_time]
                    
                    if len(seg) < 2: continue
                    
                    duration = (seg['timestamp'].iloc[-1] - seg['timestamp'].iloc[0]).total_seconds()
                    if duration <= 0: continue

                    if len(seg) > 5:
                        start_alt = seg['altitude'].iloc[:3].mean()
                        end_alt = seg['altitude'].iloc[-3:].mean()
                    else:
                        start_alt = seg['altitude'].iloc[0]
                        end_alt = seg['altitude'].iloc[-1]

                    feat = {
                        'flight_id': fid,
                        'aircraft_type': flight_ac_map.get(fid, 'Unknown'),
                        'start': row['start'],
                        'end': row['end'],
                        'duration': duration,
                        'distance_km': seg['step_dist'].sum(),
                        'groundspeed_mean': seg['groundspeed'].mean(),
                        'groundspeed_max': seg['groundspeed'].max(),
                        'altitude_mean': seg['altitude'].mean(),
                        'altitude_delta': end_alt - start_alt,
                        'vertical_rate_mean': seg['vertical_rate'].mean(),
                        'sin_track_mean': seg['sin_track'].mean(),
                        'cos_track_mean': seg['cos_track'].mean(),
                        'latitude_mean': seg['latitude'].mean(),
                        'n_points': len(seg),
                        'fuel_kg': row.get('fuel_kg', np.nan)
                    }
                    local_features.append(feat)
                except KeyError: continue
                    
    except Exception as e:
        return [] # En cas d'erreur, on renvoie vide pour ne pas bloquer
    
    del df
    gc.collect()
    return local_features

# ==========================================
# 3. ORCHESTRATEUR (MODIFIÉ AVEC TQDM)
# ==========================================

def run_parallel_pipeline():
    print(f">>> DÉMARRAGE MULTIPROCESSING ({N_CORES} CŒURS) AVEC TQDM <<<")
    
    # 1. Chargement Métadonnées
    print("[1/3] Chargement Flightlist & Target...")
    df_fl = pd.read_parquet(PATH_FLIGHTLIST)
    flight_ac_map = df_fl.set_index('flight_id')['aircraft_type'].to_dict()
    valid_flights = set(flight_ac_map.keys())
    
    df_target = pd.read_parquet(PATH_FUEL_TARGET)
    df_target['start'] = pd.to_datetime(df_target['start'])
    df_target['end'] = pd.to_datetime(df_target['end'])
    df_target = df_target[df_target['flight_id'].isin(valid_flights)]
    
    # 2. Préparation
    prc_files = glob.glob(os.path.join(PATH_PRC_FOLDER, "prc*.parquet"))
    prc_files.sort()
    
    print(f"[2/3] Traitement des {len(prc_files)} fichiers...")
    worker_func = partial(process_single_file, df_target=df_target, flight_ac_map=flight_ac_map)
    
    # 3. Exécution avec TQDM
    all_results = []
    with Pool(processes=N_CORES) as pool:
        # imap_unordered est crucial ici : il rend les résultats dès qu'un fichier est fini
        # ce qui permet à la barre de progression d'avancer en temps réel.
        results_iterator = pool.imap_unordered(worker_func, prc_files)
        
        # La barre de progression englobe l'itérateur
        for res in tqdm(results_iterator, total=len(prc_files), unit="file", desc="Processing"):
            all_results.extend(res)
            
    # 4. Consolidation
    print(f"\n[3/3] Consolidation de {len(all_results)} segments...")
    df_final = pd.DataFrame(all_results)
    
    if not df_final.empty:
        df_final['aircraft_type'] = df_final['aircraft_type'].astype('category')
        print(f"Sauvegarde dans {PATH_OUTPUT}")
        df_final.to_parquet(PATH_OUTPUT)
        print(">>> SUCCÈS <<<")
    else:
        print(">>> ECHEC : Aucun segment généré <<<")

if __name__ == "__main__":
    run_parallel_pipeline()