#!/usr/bin/env python3
"""
Nettoyage des trajectoires pour le PRC Data Challenge 2025.

Opérations:
1. Tri par timestamp
2. Déduplication (1 point max par seconde)
3. Filtrage des valeurs aberrantes (groundspeed > 700 kt, |vertical_rate| > 6000 ft/min)
4. Interpolation des petits gaps (< 5s)
5. Détection de phase de vol (OpenAP FlightPhase)

Usage:
    python clean_trajectories.py \
        --input prc_data/flights_train/ \
        --output data/clean_flights_train/ \
        --workers 8
        
    python clean_trajectories.py \
        --input prc_data/flights_rank/ \
        --output data/clean_flights_rank/ \
        --workers 8
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONSTANTES
# =============================================================================

# Seuils pour valeurs aberrantes
MAX_GROUNDSPEED_KT = 700
MAX_VERTICAL_RATE_FPM = 6000

# Seuil pour interpolation (secondes)
MAX_INTERPOLATION_GAP_SEC = 5

# Colonnes à interpoler
INTERPOLATE_COLS = ['latitude', 'longitude', 'altitude', 'groundspeed', 'vertical_rate', 'track']

# Constantes physiques pour conversion TAS
GAMMA = 1.4  # Ratio chaleur spécifique air
R_AIR = 287.05  # Constante gaz parfait air (J/kg/K)
RHO_0 = 1.225  # Densité air au niveau de la mer (kg/m³)
T_0 = 288.15  # Température ISA au niveau de la mer (K)
TROPOPAUSE_ALT_M = 11000  # Altitude tropopause (m)
T_TROPOPAUSE = 216.65  # Température à la tropopause (K)


# =============================================================================
# FONCTIONS DE CONVERSION VITESSE (ISA)
# =============================================================================

def get_isa_temperature(altitude_ft):
    """
    Retourne la température ISA à une altitude donnée.
    
    Parameters:
    -----------
    altitude_ft : float ou array, altitude en pieds
    
    Returns:
    --------
    float ou array : température en Kelvin
    """
    altitude_m = np.asarray(altitude_ft) * 0.3048
    
    # Troposphère (< 11000m) : T = T0 - 6.5 × h/1000
    # Stratosphère (≥ 11000m) : T = 216.65 K (constant)
    T = np.where(
        altitude_m < TROPOPAUSE_ALT_M,
        T_0 - 0.0065 * altitude_m,
        T_TROPOPAUSE
    )
    return T


def get_isa_density(altitude_ft):
    """
    Retourne la densité de l'air ISA à une altitude donnée.
    
    Parameters:
    -----------
    altitude_ft : float ou array, altitude en pieds
    
    Returns:
    --------
    float ou array : densité en kg/m³
    """
    altitude_m = np.asarray(altitude_ft) * 0.3048
    
    # Approximation exponentielle
    rho = RHO_0 * np.exp(-altitude_m / 8500)
    return rho


def mach_to_tas(mach, altitude_ft):
    """
    Convertit le nombre de Mach en TAS (True Airspeed).
    
    TAS = Mach × vitesse_du_son
    vitesse_du_son = sqrt(gamma × R × T)
    
    Parameters:
    -----------
    mach : float ou array, nombre de Mach
    altitude_ft : float ou array, altitude en pieds
    
    Returns:
    --------
    float ou array : TAS en nœuds
    """
    mach = np.asarray(mach)
    T = get_isa_temperature(altitude_ft)
    
    # Vitesse du son (m/s)
    a = np.sqrt(GAMMA * R_AIR * T)
    
    # TAS (m/s → kt)
    tas_ms = mach * a
    tas_kt = tas_ms / 0.514444
    
    return tas_kt


def cas_to_tas(cas_kt, altitude_ft):
    """
    Convertit la CAS (Calibrated Airspeed) en TAS (True Airspeed).
    
    Formule simplifiée (basse vitesse, incompressible) :
    TAS = CAS × sqrt(rho_0 / rho)
    
    Parameters:
    -----------
    cas_kt : float ou array, CAS en nœuds
    altitude_ft : float ou array, altitude en pieds
    
    Returns:
    --------
    float ou array : TAS en nœuds
    """
    cas_kt = np.asarray(cas_kt)
    rho = get_isa_density(altitude_ft)
    
    # Facteur de correction
    sigma = rho / RHO_0
    
    # TAS = CAS / sqrt(sigma)
    tas_kt = cas_kt / np.sqrt(sigma)
    
    return tas_kt


def extract_acars_tas(df):
    """
    Extrait le TAS des lignes ACARS.
    
    Logique :
    1. Si TAS disponible → utiliser directement
    2. Si Mach disponible → convertir avec altitude (interpolée si nécessaire)
    3. Si CAS disponible → convertir avec altitude
    
    Parameters:
    -----------
    df : DataFrame avec les colonnes source, TAS, mach, CAS, altitude, timestamp
    
    Returns:
    --------
    dict : {
        'tas_values': list de (timestamp, tas_kt),
        'cruise_tas_median': float ou None,
        'n_acars_points': int
    }
    """
    result = {
        'tas_values': [],
        'cruise_tas_median': None,
        'n_acars_points': 0
    }
    
    # Vérifier si colonne source existe
    if 'source' not in df.columns:
        return result
    
    # Extraire les lignes ACARS
    acars_mask = df['source'] == 'acars'
    if not acars_mask.any():
        return result
    
    acars_df = df[acars_mask].copy()
    result['n_acars_points'] = len(acars_df)
    
    tas_values = []
    
    # Préparer l'interpolation d'altitude sur tout le dataframe une seule fois
    # On crée une série d'altitude interpolée pour tout le vol
    df_alt_interp = df['altitude'].interpolate(method='linear', limit_direction='both')
    
    for idx, row in acars_df.iterrows():
        timestamp = row['timestamp']
        tas = None
        
        # 1. TAS directement disponible
        if 'TAS' in df.columns and pd.notna(row.get('TAS')):
            tas = row['TAS']
        
        # Si pas de TAS, on a besoin de l'altitude pour convertir Mach/CAS
        if tas is None:
            # Récupérer l'altitude (de la ligne ACARS ou interpolée)
            if pd.notna(row.get('altitude')):
                alt = row['altitude']
            else:
                # Utiliser l'altitude interpolée à cet index
                alt = df_alt_interp.loc[idx]
                
                # Si toujours NaN (cas rare), essayer d'interpoler par timestamp
                if pd.isna(alt):
                    alt = _interpolate_altitude_at_timestamp(df, timestamp)
            
            if alt is not None and alt > 0:
                # 2. Mach disponible → convertir
                if 'mach' in df.columns and pd.notna(row.get('mach')):
                    mach = row['mach']
                    tas = mach_to_tas(mach, alt)
                
                # 3. CAS disponible → convertir
                elif 'CAS' in df.columns and pd.notna(row.get('CAS')):
                    cas = row['CAS']
                    tas = cas_to_tas(cas, alt)
        
        if tas is not None and np.isfinite(tas) and 100 < tas < 600:
            tas_values.append((timestamp, float(tas)))
    
    result['tas_values'] = tas_values
    
    # Calculer le TAS médian en croisière (points à haute altitude)
    if tas_values:
        # Filtrer les points probablement en croisière (TAS > 350 kt typique)
        cruise_tas = [t for _, t in tas_values if t > 350]
        if cruise_tas:
            result['cruise_tas_median'] = float(np.median(cruise_tas))
        elif tas_values:
            result['cruise_tas_median'] = float(np.median([t for _, t in tas_values]))
    
    return result


def _interpolate_altitude_at_timestamp(df, target_ts):
    """
    Interpole l'altitude à un timestamp donné depuis les lignes ADS-B.
    
    Parameters:
    -----------
    df : DataFrame avec timestamp et altitude
    target_ts : timestamp cible
    
    Returns:
    --------
    float : altitude interpolée, ou None si impossible
    """
    # Filtrer les lignes avec altitude valide (typiquement ADS-B)
    valid = df[df['altitude'].notna()].copy()
    
    if len(valid) == 0:
        return None
    
    # Trouver les points avant et après
    before = valid[valid['timestamp'] <= target_ts]
    after = valid[valid['timestamp'] >= target_ts]
    
    if len(before) == 0 and len(after) == 0:
        return None
    
    if len(before) == 0:
        return float(after.iloc[0]['altitude'])
    
    if len(after) == 0:
        return float(before.iloc[-1]['altitude'])
    
    # Interpolation linéaire
    p_before = before.iloc[-1]
    p_after = after.iloc[0]
    
    t_before = p_before['timestamp']
    t_after = p_after['timestamp']
    
    if t_before == t_after:
        return float(p_before['altitude'])
    
    # Ratio temporel
    total_dt = (t_after - t_before).total_seconds()
    dt = (target_ts - t_before).total_seconds()
    ratio = dt / total_dt
    
    alt = p_before['altitude'] + ratio * (p_after['altitude'] - p_before['altitude'])
    return float(alt)


def create_airspeed_column(df, acars_info):
    """
    Crée la colonne 'airspeed' en propageant le TAS ACARS sur la phase de croisière.
    
    Logique :
    - En croisière (phase CR ou altitude stable haute) : airspeed = TAS ACARS médian
    - Ailleurs : airspeed = NaN (ACROPOLE utilisera groundspeed)
    
    Parameters:
    -----------
    df : DataFrame de la trajectoire (avec colonne 'phase' si disponible)
    acars_info : dict retourné par extract_acars_tas()
    
    Returns:
    --------
    DataFrame avec colonne 'airspeed' ajoutée
    """
    df = df.copy()
    
    # Initialiser à NaN
    df['airspeed'] = np.nan
    
    cruise_tas = acars_info.get('cruise_tas_median')
    
    if cruise_tas is None:
        return df
    
    # Identifier la phase de croisière
    if 'phase' in df.columns:
        # Utiliser la détection de phase
        cruise_mask = df['phase'] == 'CR'
    else:
        # Fallback : altitude > 25000 ft et vertical_rate faible
        cruise_mask = (
            (df['altitude'] > 25000) & 
            (df['vertical_rate'].abs() < 500)
        )
    
    # Propager le TAS ACARS sur la croisière
    df.loc[cruise_mask, 'airspeed'] = cruise_tas
    
    return df


# =============================================================================
# FONCTIONS DE NETTOYAGE
# =============================================================================

def deduplicate_by_second(df):
    """
    Garde un seul point par seconde (le premier).
    """
    df = df.copy()
    df['second'] = df['timestamp'].dt.floor('s')
    df = df.drop_duplicates(subset=['second'], keep='first')
    df = df.drop(columns=['second'])
    return df.reset_index(drop=True)


def filter_outliers(df):
    """
    Met à NaN les valeurs aberrantes.
    """
    df = df.copy()
    
    # Groundspeed > 700 kt
    if 'groundspeed' in df.columns:
        mask_gs = df['groundspeed'] > MAX_GROUNDSPEED_KT
        n_gs = mask_gs.sum()
        if n_gs > 0:
            df.loc[mask_gs, 'groundspeed'] = np.nan
    
    # |Vertical rate| > 6000 ft/min
    if 'vertical_rate' in df.columns:
        mask_vr = df['vertical_rate'].abs() > MAX_VERTICAL_RATE_FPM
        n_vr = mask_vr.sum()
        if n_vr > 0:
            df.loc[mask_vr, 'vertical_rate'] = np.nan
    
    return df


def interpolate_small_gaps(df):
    """
    Interpole les colonnes numériques pour les gaps < 5 secondes.
    Ne crée PAS de nouvelles lignes, remplit juste les NaN existants
    si le gap temporel est petit.
    """
    df = df.copy()
    
    # Calculer le gap temporel avec le point précédent
    df['dt'] = df['timestamp'].diff().dt.total_seconds()
    
    for col in INTERPOLATE_COLS:
        if col not in df.columns:
            continue
        
        # Identifier les NaN
        is_nan = df[col].isna()
        
        if not is_nan.any():
            continue
        
        # Pour chaque NaN, vérifier si le gap est petit
        nan_indices = df[is_nan].index
        
        for idx in nan_indices:
            # Vérifier le gap avant et après
            pos = df.index.get_loc(idx)
            
            # Gap avec le point précédent
            if pos > 0:
                dt_before = df.iloc[pos]['dt']
                if pd.notna(dt_before) and dt_before <= MAX_INTERPOLATION_GAP_SEC:
                    # Interpoler linéairement
                    # Trouver la valeur précédente non-NaN
                    prev_val = None
                    next_val = None
                    
                    for i in range(pos - 1, -1, -1):
                        if pd.notna(df.iloc[i][col]):
                            prev_val = df.iloc[i][col]
                            prev_pos = i
                            break
                    
                    for i in range(pos + 1, len(df)):
                        if pd.notna(df.iloc[i][col]):
                            next_val = df.iloc[i][col]
                            next_pos = i
                            break
                    
                    if prev_val is not None and next_val is not None:
                        # Interpolation linéaire
                        ratio = (pos - prev_pos) / (next_pos - prev_pos)
                        df.loc[idx, col] = prev_val + ratio * (next_val - prev_val)
    
    df = df.drop(columns=['dt'])
    return df


def interpolate_small_gaps_vectorized(df):
    """
    Version vectorisée plus rapide de l'interpolation.
    Interpole les NaN seulement si le gap temporel est < 5s.
    """
    df = df.copy()
    
    # Calculer les gaps temporels
    dt = df['timestamp'].diff().dt.total_seconds()
    
    # Identifier les zones où on peut interpoler (gap < 5s)
    can_interpolate = dt <= MAX_INTERPOLATION_GAP_SEC
    
    for col in INTERPOLATE_COLS:
        if col not in df.columns:
            continue
        
        # Masque des NaN dans cette colonne
        is_nan = df[col].isna()
        
        if not is_nan.any():
            continue
        
        # On interpole seulement si le gap est petit
        # Créer une série temporaire pour interpolation
        temp = df[col].copy()
        
        # Interpolation linéaire sur toute la série
        temp_interp = temp.interpolate(method='linear', limit_direction='both')
        
        # Appliquer l'interpolation seulement là où:
        # 1. La valeur originale était NaN
        # 2. Le gap temporel est acceptable
        mask = is_nan & can_interpolate
        df.loc[mask, col] = temp_interp.loc[mask]
    
    return df


def detect_flight_phases_openap(df):
    """
    Détecte les phases de vol avec OpenAP FlightPhase.
    Returns: DataFrame avec colonne 'phase' ajoutée
    """
    df = df.copy()
    
    try:
        from openap import FlightPhase
        
        # Préparer les données
        ts = df['timestamp'].values
        alt = df['altitude'].fillna(0).values
        spd = df['groundspeed'].fillna(0).values
        roc = df['vertical_rate'].fillna(0).values
        
        # Convertir timestamps en secondes depuis le début
        t0 = pd.Timestamp(ts[0])
        ts_seconds = np.array([(pd.Timestamp(t) - t0).total_seconds() for t in ts])
        
        # Créer l'objet FlightPhase
        fp = FlightPhase()
        fp.set_trajectory(ts_seconds, alt, spd, roc)
        
        # Obtenir les labels
        phases = fp.phaselabel()
        
        df['phase'] = phases
        return df
        
    except ImportError:
        # OpenAP non installé, utiliser fallback
        return detect_flight_phases_simple(df)
    except Exception as e:
        # Autre erreur, utiliser fallback
        return detect_flight_phases_simple(df)


def detect_flight_phases_simple(df):
    """
    Fallback simple pour la détection de phase si OpenAP non disponible.
    Basé sur l'altitude et le vertical_rate.
    
    GND: altitude < 1500 ft et groundspeed < 100 kt
    CL:  vertical_rate > 500 ft/min (montée significative)
    DE:  vertical_rate < -500 ft/min (descente significative)
    CR:  reste (palier)
    LVL: non utilisé dans ce fallback
    """
    df = df.copy()
    
    alt = df['altitude'].fillna(0).values
    vr = df['vertical_rate'].fillna(0).values
    gs = df['groundspeed'].fillna(0).values
    
    # Smoother le vertical_rate avec une moyenne mobile
    window = min(31, len(vr))
    if window >= 3:
        vr_smooth = pd.Series(vr).rolling(window=window, center=True, min_periods=1).mean().values
    else:
        vr_smooth = vr
    
    phases = []
    for i in range(len(df)):
        alt_val = alt[i]
        vr_val = vr_smooth[i]
        gs_val = gs[i]
        
        if alt_val < 1500 and gs_val < 100:
            phases.append('GND')
        elif vr_val > 500:
            phases.append('CL')
        elif vr_val < -500:
            phases.append('DE')
        else:
            phases.append('CR')
    
    df['phase'] = phases
    return df


# =============================================================================
# TRAITEMENT D'UN VOL
# =============================================================================

def clean_single_trajectory(args):
    """
    Nettoie une trajectoire.
    
    Parameters:
    -----------
    args : tuple (input_path, output_path)
    
    Returns:
    --------
    dict avec stats du nettoyage
    """
    input_path, output_path = args
    
    flight_id = Path(input_path).stem
    stats = {
        'flight_id': flight_id,
        'status': 'success',
        'n_points_raw': 0,
        'n_points_clean': 0,
        'n_duplicates_removed': 0,
        'n_outliers_gs': 0,
        'n_outliers_vr': 0,
        'n_interpolated': 0,
        'n_acars_points': 0,
        'cruise_tas': None,
    }
    
    try:
        # Charger
        df = pd.read_parquet(input_path)
        stats['n_points_raw'] = len(df)
        
        # 1. Trier par timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 2. EXTRAIRE LES DONNÉES ACARS AVANT TOUT NETTOYAGE
        #    (pour récupérer TAS/Mach/CAS avant de potentiellement les perdre)
        acars_info = extract_acars_tas(df)
        stats['n_acars_points'] = acars_info['n_acars_points']
        stats['cruise_tas'] = acars_info['cruise_tas_median']
        
        # 3. Dédupliquer (1 point/seconde)
        n_before = len(df)
        df = deduplicate_by_second(df)
        stats['n_duplicates_removed'] = n_before - len(df)
        
        # 4. Filtrer les outliers
        if 'groundspeed' in df.columns:
            n_gs_before = df['groundspeed'].isna().sum()
        if 'vertical_rate' in df.columns:
            n_vr_before = df['vertical_rate'].isna().sum()
        
        df = filter_outliers(df)
        
        if 'groundspeed' in df.columns:
            stats['n_outliers_gs'] = df['groundspeed'].isna().sum() - n_gs_before
        if 'vertical_rate' in df.columns:
            stats['n_outliers_vr'] = df['vertical_rate'].isna().sum() - n_vr_before
        
        # 5. Interpoler les petits gaps
        n_nan_before = df[INTERPOLATE_COLS].isna().sum().sum() if all(c in df.columns for c in INTERPOLATE_COLS[:3]) else 0
        df = interpolate_small_gaps_vectorized(df)
        n_nan_after = df[INTERPOLATE_COLS].isna().sum().sum() if all(c in df.columns for c in INTERPOLATE_COLS[:3]) else 0
        stats['n_interpolated'] = n_nan_before - n_nan_after
        
        # 6. Détecter les phases de vol
        try:
            df = detect_flight_phases_openap(df)
        except:
            df = detect_flight_phases_simple(df)
        
        # 7. CRÉER LA COLONNE AIRSPEED (TAS ACARS propagé en croisière)
        df = create_airspeed_column(df, acars_info)
        
        stats['n_points_clean'] = len(df)
        
        # Sauvegarder
        df.to_parquet(output_path, index=False)
        
    except Exception as e:
        stats['status'] = f'error: {str(e)[:50]}'
    
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Clean flight trajectories')
    parser.add_argument('--input', required=True, help='Input directory with raw trajectories')
    parser.add_argument('--output', required=True, help='Output directory for clean trajectories')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--max-flights', type=int, default=None, help='Max flights to process (for testing)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NETTOYAGE DES TRAJECTOIRES - PRC Data Challenge 2025")
    print("=" * 60)
    
    # Créer le dossier de sortie
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lister les fichiers d'entrée (seulement les trajectoires prc*)
    input_dir = Path(args.input)
    input_files = list(input_dir.glob('prc*.parquet'))
    
    # Si pas de fichiers prc*, essayer tous les parquet
    if len(input_files) == 0:
        input_files = [f for f in input_dir.glob('*.parquet') 
                       if not any(x in f.stem for x in ['flightlist', 'fuel', 'apt', 'stats'])]
    
    print(f"\nFichiers trajectoire trouvés: {len(input_files)}")
    
    if args.max_flights:
        input_files = input_files[:args.max_flights]
        print(f"Limité à {args.max_flights} fichiers")
    
    # Préparer les tâches
    tasks = []
    for input_path in input_files:
        output_path = output_dir / input_path.name
        tasks.append((str(input_path), str(output_path)))
    
    # Traitement parallèle
    print(f"\nTraitement avec {args.workers} workers...")
    
    all_stats = []
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(clean_single_trajectory, tasks),
            total=len(tasks),
            desc="Nettoyage"
        ))
    
    all_stats = results
    
    # Résumé
    print(f"\n{'=' * 60}")
    print("RÉSUMÉ")
    print(f"{'=' * 60}")
    
    df_stats = pd.DataFrame(all_stats)
    
    n_success = (df_stats['status'] == 'success').sum()
    n_error = len(df_stats) - n_success
    
    print(f"\nVols traités: {n_success} / {len(df_stats)}")
    print(f"Erreurs: {n_error}")
    
    if n_success > 0:
        success_stats = df_stats[df_stats['status'] == 'success']
        
        print(f"\nPoints:")
        print(f"  Total raw:   {success_stats['n_points_raw'].sum():,}")
        print(f"  Total clean: {success_stats['n_points_clean'].sum():,}")
        print(f"  Réduction:   {(1 - success_stats['n_points_clean'].sum() / success_stats['n_points_raw'].sum()) * 100:.1f}%")
        
        print(f"\nNettoyage:")
        print(f"  Doublons supprimés: {success_stats['n_duplicates_removed'].sum():,}")
        print(f"  Outliers GS:        {success_stats['n_outliers_gs'].sum():,}")
        print(f"  Outliers VR:        {success_stats['n_outliers_vr'].sum():,}")
        print(f"  Points interpolés:  {success_stats['n_interpolated'].sum():,}")
    
    if n_error > 0:
        print(f"\nErreurs:")
        for _, row in df_stats[df_stats['status'] != 'success'].head(10).iterrows():
            print(f"  {row['flight_id']}: {row['status']}")
    
    # Sauvegarder les stats
    stats_path = output_dir / 'cleaning_stats.parquet'
    df_stats.to_parquet(stats_path, index=False)
    print(f"\nStats sauvegardées: {stats_path}")


if __name__ == "__main__":
    main()