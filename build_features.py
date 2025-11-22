import pandas as pd
import numpy as np
import glob
import os
import gc
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================
N_CORES = 8

# --- CONFIGURATION TRAIN ---
DIR_PRC_TRAIN       = "data/raw/flights_train"        # Dossier des PRC d'entraînement
FILE_FLIGHTLIST_TR  = "data/raw/flightlist_train.parquet"
FILE_TARGET_TR      = "data/raw/fuel_train.parquet"
FILE_OUTPUT_TR      = "data/processed/X_train_final.parquet"

# --- CONFIGURATION RANK (SOUMISSION) ---
# /!\ ATTENTION : Vérifiez le nom exact du dossier contenant les prc de test sur votre disque
DIR_PRC_RANK        = "data/raw/flights_rank"         # Dossier des PRC de test (ou flights_rank)
FILE_FLIGHTLIST_RK  = "data/raw/flightlist_rank.parquet"
FILE_TARGET_RK      = "data/raw/fuel_rank_submission.parquet"
FILE_OUTPUT_RK      = "data/processed/X_test_final.parquet"

# ==========================================
# 1. MOTEUR PHYSIQUE (WORKER)
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

def process_single_file(f_path, df_target, flight_ac_map):
    """Traite un fichier PRC complet (Exécuté par un coeur CPU)."""
    local_features = []
    fname = os.path.basename(f_path)
    
    try:
        # A. Lecture & Nettoyage Physique
        df = pd.read_parquet(f_path)
        
        mask_physic = (
            (df['groundspeed'] >= 0) & (df['groundspeed'] <= 1200) &
            (df['altitude'] >= -1000) & (df['altitude'] <= 60000) &
            (df['latitude'].notna()) & (df['longitude'].notna())
        )
        df = df[mask_physic].copy()

        if df.empty: return []

        # B. Déduplication Temporelle (MÉDIANE) - Crucial pour ADS-B
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cols_to_agg = ['latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate']
        for col in ['mach', 'TAS', 'CAS']:
            if col in df.columns: cols_to_agg.append(col)
            
        # Groupby par seconde + Median pour éliminer le bruit et les doublons
        df = df.groupby(['flight_id', 'timestamp'])[cols_to_agg].median().reset_index()
        df = df.sort_values(['flight_id', 'timestamp'])

        # C. Calculs Cinématiques
        for c in cols_to_agg:
            if c in df.columns: df[c] = df[c].astype(np.float32)

        df['prev_lat'] = df.groupby('flight_id')['latitude'].shift(1)
        df['prev_lon'] = df.groupby('flight_id')['longitude'].shift(1)
        
        df['step_dist'] = haversine_vectorized(
            df['latitude'], df['longitude'], 
            df['prev_lat'], df['prev_lon']
        ).fillna(0).astype(np.float32)
        
        # Proxy Météo (Vecteurs de vent)
        track_rad = np.radians(df['track'].fillna(0))
        df['sin_track'] = np.sin(track_rad).astype(np.float32)
        df['cos_track'] = np.cos(track_rad).astype(np.float32)

        # D. Windowing (Agrégation par segment cible)
        relevant_ids = df['flight_id'].unique()
        sub_target = df_target[df_target['flight_id'].isin(relevant_ids)]
        
        if not sub_target.empty:
            prc_grouped = df.groupby('flight_id')
            
            for idx, row in sub_target.iterrows():
                fid = row['flight_id']
                try:
                    flight_data = prc_grouped.get_group(fid)
                    # Extraction fenêtre temporelle
                    mask_time = (flight_data['timestamp'] >= row['start']) & (flight_data['timestamp'] <= row['end'])
                    seg = flight_data[mask_time]
                    
                    if len(seg) < 2: continue
                    
                    duration = (seg['timestamp'].iloc[-1] - seg['timestamp'].iloc[0]).total_seconds()
                    if duration <= 0: continue

                    # Delta Altitude Robuste
                    if len(seg) > 5:
                        start_alt = seg['altitude'].iloc[:3].mean()
                        end_alt = seg['altitude'].iloc[-3:].mean()
                    else:
                        start_alt = seg['altitude'].iloc[0]
                        end_alt = seg['altitude'].iloc[-1]
                    
                    delta_alt = end_alt - start_alt

                    # --- FEATURES AVANCÉES (PHYSIQUE) ---
                    # 1. Proxy Énergie (Distance * Vitesse)
                    energy_proxy = seg['step_dist'].sum() * seg['groundspeed'].mean()
                    
                    # 2. Coût de Montée (Temps * Dénivelé)
                    climb_factor = delta_alt * duration
                    
                    # 3. Efficacité Verticale (Vz * Altitude)
                    vert_efficiency = seg['vertical_rate'].mean() * seg['altitude'].mean()

                    feat = {
                        'flight_id': fid,
                        'aircraft_type': flight_ac_map.get(fid, 'Unknown'),
                        'start': row['start'],
                        'end': row['end'],
                        # Cinématique de base
                        'duration': duration,
                        'distance_km': seg['step_dist'].sum(),
                        'groundspeed_mean': seg['groundspeed'].mean(),
                        'groundspeed_max': seg['groundspeed'].max(),
                        'altitude_mean': seg['altitude'].mean(),
                        'altitude_delta': delta_alt,
                        'vertical_rate_mean': seg['vertical_rate'].mean(),
                        # Météo implicite
                        'sin_track_mean': seg['sin_track'].mean(),
                        'cos_track_mean': seg['cos_track'].mean(),
                        'latitude_mean': seg['latitude'].mean(),
                        'n_points': len(seg),
                        # Features Physiques Avancées
                        'energy_proxy': energy_proxy,
                        'climb_factor': climb_factor,
                        'vert_efficiency': vert_efficiency,
                        # Cible (NaN pour le Rank)
                        'fuel_kg': row.get('fuel_kg', np.nan)
                    }
                    local_features.append(feat)
                except KeyError: continue
                    
    except Exception as e:
        return [] 
    
    del df
    gc.collect()
    return local_features

# ==========================================
# 2. ORCHESTRATEUR DE PIPELINE
# ==========================================

def run_pipeline(prc_folder, fl_path, target_path, output_path, label):
    print(f"\n>>> DÉMARRAGE PIPELINE : {label.upper()} <<<")
    print(f"Source PRC : {prc_folder}")
    print(f"Cible      : {target_path}")
    
    if not os.path.exists(prc_folder):
        print(f"ERREUR: Le dossier {prc_folder} n'existe pas !")
        return

    # 1. Chargement Métadonnées
    print("[1/3] Chargement Flightlist & Target...")
    df_fl = pd.read_parquet(fl_path)
    flight_ac_map = df_fl.set_index('flight_id')['aircraft_type'].to_dict()
    valid_flights = set(flight_ac_map.keys())
    
    df_target = pd.read_parquet(target_path)
    df_target['start'] = pd.to_datetime(df_target['start'])
    df_target['end'] = pd.to_datetime(df_target['end'])
    # Filtre pour ne chercher que les vols connus
    df_target = df_target[df_target['flight_id'].isin(valid_flights)]
    
    # 2. Préparation Fichiers
    prc_files = glob.glob(os.path.join(prc_folder, "prc*.parquet"))
    prc_files.sort()
    
    if not prc_files:
        print("ERREUR: Aucun fichier prc*.parquet trouvé dans le dossier source.")
        return

    print(f"[2/3] Traitement parallèle ({len(prc_files)} fichiers, {N_CORES} cœurs)...")
    
    # Partial pour figer les arguments constants
    worker_func = partial(process_single_file, df_target=df_target, flight_ac_map=flight_ac_map)
    
    all_results = []
    with Pool(processes=N_CORES) as pool:
        results_iterator = pool.imap_unordered(worker_func, prc_files)
        for res in tqdm(results_iterator, total=len(prc_files), unit="file", desc=f"{label}"):
            all_results.extend(res)
            
    # 3. Sauvegarde
    print(f"[3/3] Consolidation ({len(all_results)} segments)...")
    if not all_results:
        print("ERREUR: Résultat vide. Vérifiez vos chemins !")
        return

    df_final = pd.DataFrame(all_results)
    df_final['aircraft_type'] = df_final['aircraft_type'].astype('category')
    
    # Création dossier de sortie si inexistant
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_final.to_parquet(output_path)
    print(f"SUCCÈS : Fichier sauvegardé -> {output_path}")

# ==========================================
# 3. POINT D'ENTRÉE (CLI)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de Features Aéronautiques")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'rank', 'all'], 
                        help="Mode d'exécution : 'train' (apprentissage), 'rank' (soumission), ou 'all' (les deux)")
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'all']:
        run_pipeline(DIR_PRC_TRAIN, FILE_FLIGHTLIST_TR, FILE_TARGET_TR, FILE_OUTPUT_TR, "Train Set")
        
    if args.mode in ['rank', 'all']:
        run_pipeline(DIR_PRC_RANK, FILE_FLIGHTLIST_RK, FILE_TARGET_RK, FILE_OUTPUT_RK, "Rank Set")