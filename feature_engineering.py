#!/usr/bin/env python3
"""
Feature Engineering pour le PRC Data Challenge 2025.

Intègre:
- Estimation de masse PRC2024 (team_likable_jelly) via energy rate
- ACROPOLE (types supportés) + OpenAP (fallback) pour estimer fuel flow
- Calcul itératif avec correction de masse

Multiprocessing avec 8 workers.

Usage:
    python feature_engineering.py \
        --trajectories /path/to/prc_files/ \
        --flightlist flightlist_train.parquet \
        --fuel fuel_train.parquet \
        --output features_train.parquet \
        --workers 8
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES
# =============================================================================

# Constantes physiques
G = 9.80665  # m/s²
FT2M = 0.3048
KT2MS = 0.514444
FPM2MS = 0.00508

# Types supportés par ACROPOLE (à vérifier/ajuster selon la version installée)
ACROPOLE_TYPES = {
    'A319', 'A320', 'A321', 'A20N', 'A21N',  # A320 family
    'B737', 'B738', 'B739',                   # B737 family
    'A332', 'A333',                           # A330 family
}

# Types supportés par OpenAP
OPENAP_TYPES = {
    'A319', 'A320', 'A321', 'A20N', 'A21N',
    'A332', 'A333', 'A339', 'A359', 'A35K',
    'B737', 'B738', 'B739', 'B38M', 'B39M',
    'B744', 'B748', 'B752', 'B763', 'B772', 'B773', 'B77W', 'B788', 'B789', 'B78X',
    'E190', 'E195', 'E290', 'E295',
    'C560', 'CRJ9', 'DH8D', 'ATR72',
}

# Masse typique en % du MTOW pour estimation (fallback)
TYPICAL_MASS_RATIO = 0.85


# =============================================================================
# INITIALISATION DES ESTIMATEURS (lazy loading)
# =============================================================================

_acropole_estimator = None
_openap_fuelflows = {}
_openap_props = {}


def get_acropole():
    """Lazy loading de l'estimateur ACROPOLE."""
    global _acropole_estimator
    if _acropole_estimator is None:
        try:
            from acropole import FuelEstimator
            _acropole_estimator = FuelEstimator()
        except ImportError:
            print("WARNING: ACROPOLE non installé, fallback vers OpenAP")
            _acropole_estimator = False
    return _acropole_estimator


def get_openap_fuelflow(typecode):
    """Lazy loading des FuelFlow OpenAP par type."""
    global _openap_fuelflows, _openap_props
    
    if typecode not in _openap_fuelflows:
        try:
            from openap import FuelFlow, prop
            
            # Récupérer les props de l'avion
            ac = prop.aircraft(typecode)
            _openap_props[typecode] = ac
            
            # Créer l'estimateur
            _openap_fuelflows[typecode] = FuelFlow(typecode)
        except Exception as e:
            _openap_fuelflows[typecode] = None
            _openap_props[typecode] = None
    
    return _openap_fuelflows.get(typecode), _openap_props.get(typecode)


# =============================================================================
# ESTIMATION DE MASSE (simple fallback 85% MTOW)
# =============================================================================


def estimate_initial_mass(df, typecode):
    """
    Retourne une estimation simple de la masse initiale : 85% MTOW.
    
    Note: On pourrait implémenter une méthode plus sophistiquée (PRC2024),
    mais pour l'instant on utilise le fallback simple car :
    - ACROPOLE gère la masse internement
    - OpenAP fonctionne bien avec 85% MTOW comme approximation
    
    Parameters:
    -----------
    df : DataFrame de la trajectoire (non utilisé, pour compatibilité)
    typecode : str, type d'avion
    
    Returns:
    --------
    float : masse estimée en kg, ou None si type non supporté
    """
    try:
        from openap import prop
        ac = prop.aircraft(typecode)
        mtow = ac.get('mtow', 75000)
        return mtow * TYPICAL_MASS_RATIO  # 85%
    except:
        return None


# =============================================================================
# ESTIMATION FUEL FLOW
# =============================================================================

def estimate_fuel_acropole(df):
    """
    Estime le fuel flow avec ACROPOLE.
    
    Utilise la colonne 'airspeed' (TAS) si disponible, sinon 'groundspeed'.
    ACROPOLE est plus précis avec TAS, surtout en croisière avec du vent.
    
    Parameters:
    -----------
    df : DataFrame avec colonnes: typecode, groundspeed, altitude, vertical_rate
         et optionnellement 'airspeed' (TAS extrait des données ACARS)
    
    Returns:
    --------
    Array de fuel_flow en kg/s
    """
    fe = get_acropole()
    
    if fe is False or fe is None:
        return np.full(len(df), np.nan)
    
    try:
        # Préparer le DataFrame pour ACROPOLE
        input_df = df[['typecode', 'groundspeed', 'altitude', 'vertical_rate']].copy()
        
        # IMPORTANT: Interpoler les NaN résiduels dans groundspeed et vertical_rate
        # Ces NaN peuvent exister si le gap temporel était > 5s lors du nettoyage
        input_df['groundspeed'] = input_df['groundspeed'].interpolate(method='linear', limit_direction='both')
        input_df['vertical_rate'] = input_df['vertical_rate'].interpolate(method='linear', limit_direction='both')
        input_df['altitude'] = input_df['altitude'].interpolate(method='linear', limit_direction='both')
        
        # Remplir les NaN restants (début/fin) avec ffill/bfill
        input_df['groundspeed'] = input_df['groundspeed'].ffill().bfill()
        input_df['vertical_rate'] = input_df['vertical_rate'].fillna(0)  # 0 = vol en palier
        input_df['altitude'] = input_df['altitude'].ffill().bfill()
        
        # Créer la colonne airspeed: TAS si disponible, sinon GS
        if 'airspeed' in df.columns:
            # airspeed = TAS (ACARS) où disponible, sinon groundspeed interpolé
            input_df['airspeed'] = df['airspeed'].fillna(input_df['groundspeed'])
        else:
            # Pas de TAS disponible, ACROPOLE utilisera groundspeed par défaut
            input_df['airspeed'] = input_df['groundspeed']
        
        # Estimer avec airspeed
        result = fe.estimate(
            input_df,
            typecode='typecode',
            groundspeed='groundspeed',
            altitude='altitude',
            vertical_rate='vertical_rate',
            airspeed='airspeed'
        )
        
        return result['fuel_flow'].values
    
    except Exception as e:
        # Fallback sans airspeed
        try:
            input_df = df[['typecode', 'groundspeed', 'altitude', 'vertical_rate']].copy()
            input_df['groundspeed'] = input_df['groundspeed'].interpolate(method='linear').ffill().bfill()
            input_df['vertical_rate'] = input_df['vertical_rate'].interpolate(method='linear').fillna(0)
            input_df['altitude'] = input_df['altitude'].interpolate(method='linear').ffill().bfill()
            result = fe.estimate(input_df)
            return result['fuel_flow'].values
        except:
            return np.full(len(df), np.nan)


def estimate_fuel_openap_vectorized(df, typecode, estimated_mass=None):
    """
    Estimation du fuel flow avec OpenAP - VERSION VECTORISÉE.
    
    Approche en 2 passes (exactement comme la doc OpenAP):
    1. Premier pass avec masse initiale (estimée PRC2024 ou 85% MTOW en fallback)
    2. Correction de la masse à chaque pas (masse -= fuel consommé cumulé)
    3. Second pass avec masse corrigée
    
    Parameters:
    -----------
    df : DataFrame de la trajectoire
    typecode : str, type d'avion
    estimated_mass : float, masse initiale estimée (si None, sera estimée ou fallback)
    
    Returns:
    --------
    tuple: (fuel_flow array en kg/s, total_fuel en kg, mass0 utilisée, mass_source)
    """
    ff_model, ac_props = get_openap_fuelflow(typecode)
    
    if ff_model is None:
        return np.full(len(df), np.nan), np.nan, np.nan, 'none'
    
    try:
        # Paramètres avion
        mtow = ac_props.get('mtow', 75000)
        oew = ac_props.get('oew', mtow * 0.5)
        
        # --- Déterminer la masse initiale ---
        if estimated_mass is not None and oew * 0.8 < estimated_mass < mtow * 1.1:
            # Utiliser la masse fournie (déjà estimée)
            mass0 = estimated_mass
            mass_source = 'prc2024'
        else:
            # Essayer d'estimer la masse avec la méthode PRC2024
            mass0 = estimate_initial_mass(df, typecode)
            
            if mass0 is not None and oew * 0.8 < mass0 < mtow * 1.1:
                mass_source = 'prc2024'
            else:
                # Fallback: 85% du MTOW
                mass0 = mtow * TYPICAL_MASS_RATIO
                mass_source = 'fallback_85pct'
        
        # Données de vol - interpoler les NaN plutôt que mettre 0
        tas = df['groundspeed'].interpolate(method='linear').ffill().bfill().values
        alt = df['altitude'].interpolate(method='linear').ffill().bfill().values
        vs = df['vertical_rate'].interpolate(method='linear').fillna(0).values if 'vertical_rate' in df.columns else np.zeros(len(df))
        
        # Si airspeed (TAS ACARS) est disponible, l'utiliser en croisière
        if 'airspeed' in df.columns:
            airspeed = df['airspeed'].values
            # Remplacer tas par airspeed là où disponible
            mask = ~np.isnan(airspeed)
            tas[mask] = airspeed[mask]
        
        # dt = time step en secondes (bfill pour le premier point)
        dt = df['timestamp'].diff().bfill().dt.total_seconds().values
        
        # --- Premier pass avec masse initiale ---
        fuel_flow_initial = ff_model.enroute(
            mass=mass0,
            tas=tas,
            alt=alt,
            vs=vs
        ).flatten()
        
        # Remplacer les NaN par 0 pour éviter la propagation dans cumsum
        fuel_flow_initial_clean = np.nan_to_num(fuel_flow_initial, nan=0.0)
        
        # --- Correction de la masse à chaque pas ---
        mass = mass0 - (fuel_flow_initial_clean * dt).cumsum()
        
        # S'assurer que la masse ne descend pas sous OEW
        mass = np.maximum(mass, oew)
        
        # --- Second pass avec masse corrigée ---
        fuel_flow = ff_model.enroute(
            mass=mass,
            tas=tas,
            alt=alt,
            vs=vs
        ).flatten()
        
        # Fuel total consommé
        total_fuel = np.sum(fuel_flow * dt)
        
        return fuel_flow, total_fuel, mass0, mass_source
    
    except Exception as e:
        return np.full(len(df), np.nan), np.nan, np.nan, 'error'


# =============================================================================
# CALCUL DES FEATURES PAR SEGMENT
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points."""
    R = 6371.0  # Rayon de la Terre en km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def compute_segment_features(segment_df, segment_info):
    """
    Calcule les features pour un segment.
    
    Parameters:
    -----------
    segment_df : DataFrame des points de trajectoire dans ce segment
    segment_info : dict avec flight_id, idx, etc.
    
    Returns:
    --------
    dict de features
    """
    features = {
        'flight_id': segment_info['flight_id'],
        'idx': segment_info['idx'],
    }
    
    n_points = len(segment_df)
    features['n_points'] = n_points
    
    if n_points == 0:
        return features
    
    # --- Temporel ---
    if 'timestamp' in segment_df.columns:
        t_min = segment_df['timestamp'].min()
        t_max = segment_df['timestamp'].max()
        duration_sec = (t_max - t_min).total_seconds()
        features['duration_sec'] = duration_sec
        features['timestamp_start'] = t_min
        features['timestamp_end'] = t_max
    
    # --- Spatial ---
    if 'latitude' in segment_df.columns and 'longitude' in segment_df.columns:
        lat = segment_df['latitude'].dropna()
        lon = segment_df['longitude'].dropna()
        
        if len(lat) >= 2:
            # Distance totale parcourue
            distances = haversine_distance(
                lat.iloc[:-1].values, lon.iloc[:-1].values,
                lat.iloc[1:].values, lon.iloc[1:].values
            )
            features['distance_km'] = np.sum(distances)
            
            # Distance directe (great circle)
            features['distance_direct_km'] = haversine_distance(
                lat.iloc[0], lon.iloc[0],
                lat.iloc[-1], lon.iloc[-1]
            )
    
    # --- Altitude ---
    if 'altitude' in segment_df.columns:
        alt = segment_df['altitude'].dropna()
        if len(alt) > 0:
            features['alt_mean'] = alt.mean()
            features['alt_min'] = alt.min()
            features['alt_max'] = alt.max()
            features['alt_std'] = alt.std() if len(alt) > 1 else 0
            features['alt_delta'] = alt.iloc[-1] - alt.iloc[0] if len(alt) >= 2 else 0
    
    # --- Groundspeed ---
    if 'groundspeed' in segment_df.columns:
        gs = segment_df['groundspeed'].dropna()
        if len(gs) > 0:
            features['gs_mean'] = gs.mean()
            features['gs_min'] = gs.min()
            features['gs_max'] = gs.max()
            features['gs_std'] = gs.std() if len(gs) > 1 else 0
    
    # --- Vertical Rate ---
    if 'vertical_rate' in segment_df.columns:
        vr = segment_df['vertical_rate'].dropna()
        if len(vr) > 0:
            features['vrate_mean'] = vr.mean()
            features['vrate_abs_mean'] = vr.abs().mean()
            features['vrate_min'] = vr.min()
            features['vrate_max'] = vr.max()
            features['vrate_std'] = vr.std() if len(vr) > 1 else 0
    
    # --- Track (cap) ---
    if 'track' in segment_df.columns:
        track = segment_df['track'].dropna()
        if len(track) > 1:
            # Changement de cap total (en tenant compte du wrap-around)
            track_diff = np.diff(track.values)
            track_diff = np.abs((track_diff + 180) % 360 - 180)
            features['track_change_total'] = np.sum(track_diff)
            features['track_std'] = track.std()
    
    # --- Fuel Flow estimé ---
    if 'fuel_flow' in segment_df.columns:
        ff = segment_df['fuel_flow'].dropna()
        if len(ff) > 0:
            features['fuel_flow_mean'] = ff.mean()
            features['fuel_flow_min'] = ff.min()
            features['fuel_flow_max'] = ff.max()
            features['fuel_flow_std'] = ff.std() if len(ff) > 1 else 0
            
            # Fuel total estimé (intégration trapézoïdale)
            if 'timestamp' in segment_df.columns and len(ff) >= 2:
                ff_valid = segment_df.dropna(subset=['fuel_flow', 'timestamp'])
                if len(ff_valid) >= 2:
                    dt = ff_valid['timestamp'].diff().dt.total_seconds().iloc[1:].values
                    ff_vals = ff_valid['fuel_flow'].values
                    # Trapèze
                    fuel_estimated = np.sum((ff_vals[:-1] + ff_vals[1:]) / 2 * dt)
                    features['fuel_estimated_kg'] = fuel_estimated
    
    # --- Phase de vol (utiliser celle calculée par clean_trajectories si disponible) ---
    if 'phase' in segment_df.columns:
        # Utiliser la phase majoritaire dans le segment
        phase_counts = segment_df['phase'].value_counts()
        if len(phase_counts) > 0:
            features['phase'] = phase_counts.index[0]
            
            # Pourcentage de chaque phase dans le segment
            for phase in ['GND', 'CL', 'CR', 'DE', 'LVL']:
                pct = (segment_df['phase'] == phase).mean()
                features[f'phase_{phase}_pct'] = pct
        else:
            features['phase'] = 'NA'
    else:
        # Fallback: détection simple basée sur alt_delta et vrate_mean
        alt_delta = features.get('alt_delta', 0)
        vrate_mean = features.get('vrate_mean', 0)
        
        if alt_delta > 2000 or vrate_mean > 500:
            features['phase'] = 'CL'
        elif alt_delta < -2000 or vrate_mean < -500:
            features['phase'] = 'DE'
        else:
            features['phase'] = 'CR'
    
    return features


# =============================================================================
# TRAITEMENT D'UN VOL
# =============================================================================

def process_single_flight(args):
    """
    Traite un vol complet : charge trajectoire, estime fuel, calcule features par segment.
    
    Parameters:
    -----------
    args : tuple (flight_id, typecode, traj_path, segments_df, flight_info)
    
    Returns:
    --------
    list of dict (features par segment)
    """
    flight_id, typecode, traj_path, segments_df, flight_info = args
    
    results = []
    
    try:
        # Charger la trajectoire
        if not os.path.exists(traj_path):
            return results
        
        traj_df = pd.read_parquet(traj_path)
        traj_df = traj_df.sort_values('timestamp').reset_index(drop=True)
        
        # --- INTERPOLER LES DONNÉES MANQUANTES ---
        # Certaines lignes (ACARS) n'ont pas groundspeed, vertical_rate, track
        # On les interpole depuis les points ADS-B voisins
        for col in ['groundspeed', 'altitude', 'vertical_rate', 'track']:
            if col in traj_df.columns:
                traj_df[col] = traj_df[col].interpolate(method='linear', limit_direction='both')
                traj_df[col] = traj_df[col].ffill().bfill()
        
        # vertical_rate : NaN restants → 0 (vol en palier)
        if 'vertical_rate' in traj_df.columns:
            traj_df['vertical_rate'] = traj_df['vertical_rate'].fillna(0)
        
        # --- Estimer le fuel flow ---
        mass_estimated = np.nan
        mass_source = 'none'
        
        if typecode in ACROPOLE_TYPES:
            traj_df['fuel_flow'] = estimate_fuel_acropole(traj_df)
            traj_df['fuel_source'] = 'acropole'
            # ACROPOLE gère la masse internement, pas besoin d'estimer
            mass_estimated = np.nan
            mass_source = 'acropole_internal'
        elif typecode in OPENAP_TYPES:
            fuel_flow, total_fuel, mass_estimated, mass_source = estimate_fuel_openap_vectorized(traj_df, typecode)
            traj_df['fuel_flow'] = fuel_flow
            traj_df['fuel_source'] = 'openap'
        else:
            traj_df['fuel_flow'] = np.nan
            traj_df['fuel_source'] = 'none'
        
        # --- Infos du vol depuis flightlist ---
        origin_icao = flight_info.get('origin_icao', '')
        destination_icao = flight_info.get('destination_icao', '')
        takeoff_time = flight_info.get('takeoff')
        landed_time = flight_info.get('landed')
        
        # Durée totale du vol
        if takeoff_time and landed_time:
            try:
                flight_duration_min = (pd.to_datetime(landed_time) - pd.to_datetime(takeoff_time)).total_seconds() / 60
            except:
                flight_duration_min = np.nan
        else:
            flight_duration_min = np.nan
        
        # Nombre total de segments pour ce vol
        n_segments_total = len(segments_df)
        
        # --- Traiter chaque segment ---
        for seg_idx, (_, seg_row) in enumerate(segments_df.iterrows()):
            # Colonnes réelles : idx, flight_id, start, end, fuel_kg
            segment_idx = seg_row['idx']
            seg_start = pd.to_datetime(seg_row['start'])
            seg_end = pd.to_datetime(seg_row['end'])
            
            # Filtrer les points du segment
            mask = (traj_df['timestamp'] >= seg_start) & (traj_df['timestamp'] <= seg_end)
            segment_traj = traj_df[mask].copy()
            
            # Calculer les features
            segment_info = {
                'flight_id': flight_id,
                'idx': segment_idx,
            }
            
            features = compute_segment_features(segment_traj, segment_info)
            
            # Ajouter des infos du segment
            features['typecode'] = typecode
            features['start'] = seg_start
            features['end'] = seg_end
            
            # Infos du vol
            features['origin_icao'] = origin_icao
            features['destination_icao'] = destination_icao
            features['flight_duration_min'] = flight_duration_min
            
            # Masse estimée (PRC2024 style)
            features['mass_estimated_kg'] = mass_estimated
            features['mass_source'] = mass_source
            
            # Position relative du segment dans le vol
            features['segment_position'] = seg_idx / max(n_segments_total - 1, 1)
            features['n_segments_total'] = n_segments_total
            features['is_first_segment'] = 1 if seg_idx == 0 else 0
            features['is_last_segment'] = 1 if seg_idx == n_segments_total - 1 else 0
            
            # Heure de la journée (UTC)
            features['hour_utc'] = seg_start.hour
            features['day_of_week'] = seg_start.dayofweek
            
            # Garder fuel_kg si disponible (train set)
            if 'fuel_kg' in seg_row.index:
                fuel_val = seg_row['fuel_kg']
                if pd.notna(fuel_val) and fuel_val != 'None' and fuel_val is not None:
                    try:
                        features['fuel_kg'] = float(fuel_val)
                    except:
                        pass
            
            results.append(features)
    
    except Exception as e:
        pass  # Silently skip errors
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def _init_worker():
    """Initialise les caches dans chaque worker."""
    import warnings
    warnings.filterwarnings('ignore')


def save_flight_checkpoint(flight_id, features_list, checkpoint_dir):
    """Sauvegarde les features d'un vol dans un fichier checkpoint."""
    if not features_list:
        return
    checkpoint_path = Path(checkpoint_dir) / f"features_{flight_id}.parquet"
    df = pd.DataFrame(features_list)
    df.to_parquet(checkpoint_path, index=False)


def process_and_save(args_with_checkpoint):
    """Wrapper qui traite un vol et sauvegarde immédiatement."""
    task, checkpoint_dir = args_with_checkpoint
    flight_id = task[0]
    
    # Traiter le vol
    features_list = process_single_flight(task)
    
    # Sauvegarder immédiatement
    if features_list:
        save_flight_checkpoint(flight_id, features_list, checkpoint_dir)
    
    return flight_id, len(features_list)


def get_completed_flights(checkpoint_dir):
    """Retourne l'ensemble des flight_id déjà traités."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return set()
    
    completed = set()
    for f in checkpoint_path.glob("features_*.parquet"):
        # Extraire flight_id du nom: features_prc770870642.parquet -> prc770870642
        flight_id = f.stem.replace("features_", "")
        completed.add(flight_id)
    
    return completed


def concat_checkpoints(checkpoint_dir, output_path):
    """Concatène tous les fichiers checkpoint en un seul fichier."""
    checkpoint_path = Path(checkpoint_dir)
    
    all_files = list(checkpoint_path.glob("features_*.parquet"))
    if not all_files:
        print("Aucun fichier checkpoint trouvé!")
        return None
    
    print(f"Concaténation de {len(all_files)} fichiers...")
    
    dfs = []
    for f in tqdm(all_files, desc="Lecture"):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Erreur lecture {f.name}: {e}")
    
    if not dfs:
        return None
    
    features_df = pd.concat(dfs, ignore_index=True)
    
    # S'assurer que idx est bien typé
    if 'idx' in features_df.columns:
        features_df['idx'] = features_df['idx'].astype(int)
    
    # Trier par flight_id et idx
    if 'flight_id' in features_df.columns and 'idx' in features_df.columns:
        features_df = features_df.sort_values(['flight_id', 'idx']).reset_index(drop=True)
    
    # Sauvegarder
    features_df.to_parquet(output_path, index=False)
    
    return features_df


def main():
    parser = argparse.ArgumentParser(description='Feature Engineering PRC Challenge')
    parser.add_argument('--trajectories', required=True, help='Dossier des fichiers trajectoire (prc*.parquet)')
    parser.add_argument('--flightlist', required=True, help='Fichier flightlist.parquet')
    parser.add_argument('--fuel', required=True, help='Fichier fuel_train.parquet ou fuel_rank_submission.parquet')
    parser.add_argument('--output', required=True, help='Fichier de sortie features.parquet')
    parser.add_argument('--workers', type=int, default=8, help='Nombre de workers (défaut: 8)')
    parser.add_argument('--no-parallel', action='store_true', help='Désactiver le multiprocessing')
    parser.add_argument('--max-flights', type=int, default=None, help='Limite de vols (pour test)')
    
    # Nouvelles options pour checkpoint/reprise
    parser.add_argument('--checkpoint-dir', default=None, 
                        help='Dossier pour sauvegarder les checkpoints (1 fichier par vol)')
    parser.add_argument('--resume', action='store_true',
                        help='Reprendre: skip les vols déjà dans checkpoint-dir')
    parser.add_argument('--concat-only', action='store_true',
                        help='Ne pas traiter, juste concaténer les checkpoints existants')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FEATURE ENGINEERING - PRC Data Challenge 2025")
    print("=" * 60)
    print(f"Workers: {args.workers}")
    
    # --- Configurer checkpoint ---
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint dir: {checkpoint_dir}")
    else:
        # Créer un dossier temporaire à côté du fichier output
        checkpoint_dir = Path(args.output).parent / "checkpoints_temp"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint dir (auto): {checkpoint_dir}")
    
    # --- Mode concat-only ---
    if args.concat_only:
        print(f"\nMode CONCAT-ONLY: concaténation des checkpoints...")
        features_df = concat_checkpoints(checkpoint_dir, args.output)
        if features_df is not None:
            print(f"\nSauvegardé: {args.output}")
            print(f"Segments: {len(features_df)}")
            print(f"Taille: {os.path.getsize(args.output) / 1024**2:.1f} Mo")
        return
    
    # --- Charger les métadonnées ---
    print(f"\nChargement flightlist: {args.flightlist}")
    flightlist = pd.read_parquet(args.flightlist)
    print(f"  Vols: {len(flightlist)}")
    print(f"  Colonnes: {flightlist.columns.tolist()}")
    
    print(f"\nChargement fuel: {args.fuel}")
    fuel_df = pd.read_parquet(args.fuel)
    print(f"  Segments: {len(fuel_df)}")
    
    # Afficher les colonnes pour diagnostic
    print(f"  Colonnes fuel: {fuel_df.columns.tolist()}")
    
    # --- Vérifier les vols déjà traités (mode resume) ---
    completed_flights = set()
    if args.resume:
        completed_flights = get_completed_flights(checkpoint_dir)
        print(f"\nMode RESUME: {len(completed_flights)} vols déjà traités")
    
    # --- Préparer les tâches ---
    trajectories_dir = Path(args.trajectories)
    
    # Grouper segments par vol
    segments_by_flight = fuel_df.groupby('flight_id')
    
    # Créer un dict avec toutes les infos du flightlist
    flightlist_dict = flightlist.set_index('flight_id').to_dict('index')
    
    tasks = []
    missing_traj = 0
    skipped_completed = 0
    
    for flight_id, segments in segments_by_flight:
        # Skip si déjà traité (mode resume)
        if flight_id in completed_flights:
            skipped_completed += 1
            continue
        
        # Récupérer les infos du vol
        flight_info = flightlist_dict.get(flight_id, {})
        typecode = flight_info.get('aircraft_type', 'UNKNOWN')
        
        # Chercher le fichier trajectoire
        traj_file = trajectories_dir / f"{flight_id}.parquet"
        
        if not traj_file.exists():
            missing_traj += 1
            continue
        
        tasks.append((flight_id, typecode, str(traj_file), segments, flight_info))
    
    print(f"\nTâches préparées: {len(tasks)}")
    print(f"Trajectoires manquantes: {missing_traj}")
    if args.resume:
        print(f"Vols skippés (déjà traités): {skipped_completed}")
    
    if args.max_flights:
        tasks = tasks[:args.max_flights]
        print(f"Limité à {args.max_flights} vols")
    
    if len(tasks) == 0:
        print("\nAucun vol à traiter!")
        if args.resume and completed_flights:
            print("Tous les vols sont déjà traités. Utilisez --concat-only pour concaténer.")
        return
    
    # --- Statistiques des types ---
    type_counts = {}
    for _, typecode, _, _, _ in tasks:
        type_counts[typecode] = type_counts.get(typecode, 0) + 1
    
    acropole_count = sum(v for k, v in type_counts.items() if k in ACROPOLE_TYPES)
    openap_count = sum(v for k, v in type_counts.items() if k in OPENAP_TYPES and k not in ACROPOLE_TYPES)
    unknown_count = sum(v for k, v in type_counts.items() if k not in OPENAP_TYPES)
    
    print(f"\nTypes d'avions:")
    print(f"  ACROPOLE: {acropole_count} vols ({acropole_count/len(tasks)*100:.1f}%)")
    print(f"  OpenAP: {openap_count} vols ({openap_count/len(tasks)*100:.1f}%)")
    print(f"  Non supporté: {unknown_count} vols ({unknown_count/len(tasks)*100:.1f}%)")
    
    # --- Traitement avec sauvegarde checkpoint ---
    print(f"\nTraitement en cours avec {args.workers} workers...")
    print(f"Chaque vol est sauvegardé dans: {checkpoint_dir}/")
    print("(Vous pouvez interrompre avec Ctrl+C et reprendre avec --resume)")
    
    # Préparer les tâches avec checkpoint_dir
    tasks_with_checkpoint = [(task, str(checkpoint_dir)) for task in tasks]
    
    n_processed = 0
    n_segments = 0
    
    try:
        if args.no_parallel:
            # Mode séquentiel (debug)
            print("Mode séquentiel (--no-parallel)")
            for task_cp in tqdm(tasks_with_checkpoint, desc="Vols"):
                flight_id, n_seg = process_and_save(task_cp)
                n_processed += 1
                n_segments += n_seg
        else:
            # Mode parallèle avec initialisation des workers
            with Pool(processes=args.workers, initializer=_init_worker) as pool:
                for flight_id, n_seg in tqdm(
                    pool.imap_unordered(process_and_save, tasks_with_checkpoint),
                    total=len(tasks_with_checkpoint),
                    desc="Vols"
                ):
                    n_processed += 1
                    n_segments += n_seg
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  INTERRUPTION - {n_processed} vols traités")
        print(f"Pour reprendre: ajoutez --resume")
        print(f"Pour concaténer: ajoutez --concat-only")
        return
    
    print(f"\nVols traités: {n_processed}")
    print(f"Segments calculés: {n_segments}")
    
    # --- Concaténer les checkpoints ---
    print(f"\nConcaténation des checkpoints...")
    features_df = concat_checkpoints(checkpoint_dir, args.output)
    
    if features_df is None:
        print("Erreur: aucune feature à concaténer")
        return
    
    # --- Résumé ---
    print(f"\n{'=' * 60}")
    print("RÉSUMÉ")
    print(f"{'=' * 60}")
    print(f"Fichier: {args.output}")
    print(f"Segments: {len(features_df)}")
    print(f"Colonnes: {features_df.columns.tolist()}")
    print(f"Taille: {os.path.getsize(args.output) / 1024**2:.1f} Mo")
    
    if 'fuel_flow_mean' in features_df.columns:
        nan_ff = features_df['fuel_flow_mean'].isna().sum()
        print(f"Fuel flow NaN: {nan_ff} ({nan_ff/len(features_df)*100:.1f}%)")
    
    if 'fuel_estimated_kg' in features_df.columns:
        nan_est = features_df['fuel_estimated_kg'].isna().sum()
        print(f"Fuel estimé NaN: {nan_est} ({nan_est/len(features_df)*100:.1f}%)")
    
    if 'fuel_kg' in features_df.columns and 'fuel_estimated_kg' in features_df.columns:
        valid = features_df.dropna(subset=['fuel_kg', 'fuel_estimated_kg'])
        if len(valid) > 0:
            rmse = np.sqrt(((valid['fuel_kg'] - valid['fuel_estimated_kg'])**2).mean())
            mae = (valid['fuel_kg'] - valid['fuel_estimated_kg']).abs().mean()
            bias = (valid['fuel_estimated_kg'] - valid['fuel_kg']).mean()
            print(f"\nPerformance estimateur physique:")
            print(f"  RMSE: {rmse:.1f} kg")
            print(f"  MAE: {mae:.1f} kg")
            print(f"  Biais: {bias:+.1f} kg")
    
    print("\nAperçu:")
    print(features_df.head(10))
    
    # Info sur les checkpoints
    n_checkpoints = len(list(checkpoint_dir.glob("features_*.parquet")))
    print(f"\n💾 Checkpoints conservés: {n_checkpoints} fichiers dans {checkpoint_dir}/")
    print("   (Supprimez ce dossier manuellement si vous n'en avez plus besoin)")


if __name__ == "__main__":
    main()