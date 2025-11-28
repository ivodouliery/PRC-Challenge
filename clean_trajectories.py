#!/usr/bin/env python3
"""
Trajectory Cleaning Pipeline for PRC Data Challenge 2025.

Operations:
1. Sort by timestamp
2. Deduplication (max 1 point per second)
3. Outlier filtering (groundspeed > 700 kt, |vertical_rate| > 6000 ft/min)
4. Small gap interpolation (< 5s)
5. Flight phase detection (OpenAP FlightPhase)

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
# CONSTANTS
# =============================================================================

# Outlier thresholds
MAX_GROUNDSPEED_KT = 700
MAX_VERTICAL_RATE_FPM = 6000

# Interpolation threshold (seconds)
MAX_INTERPOLATION_GAP_SEC = 5

# Columns to interpolate
INTERPOLATE_COLS = ['latitude', 'longitude', 'altitude', 'groundspeed', 'vertical_rate', 'track']

# Physical constants for TAS conversion
GAMMA = 1.4  # Specific heat ratio
R_AIR = 287.05  # Ideal gas constant for air (J/kg/K)
RHO_0 = 1.225  # Air density at sea level (kg/m³)
T_0 = 288.15  # ISA temperature at sea level (K)
TROPOPAUSE_ALT_M = 11000  # Tropopause altitude (m)
T_TROPOPAUSE = 216.65  # Temperature at tropopause (K)


# =============================================================================
# SPEED CONVERSION FUNCTIONS (ISA)
# =============================================================================

def get_isa_temperature(altitude_ft):
    """
    Returns ISA temperature at a given altitude.
    
    Parameters:
    -----------
    altitude_ft : float or array, altitude in feet
    
    Returns:
    --------
    float or array : temperature in Kelvin
    """
    altitude_m = np.asarray(altitude_ft) * 0.3048
    
    # Troposphere (< 11000m) : T = T0 - 6.5 × h/1000
    # Stratosphere (≥ 11000m) : T = 216.65 K (constant)
    T = np.where(
        altitude_m < TROPOPAUSE_ALT_M,
        T_0 - 0.0065 * altitude_m,
        T_TROPOPAUSE
    )
    return T


def get_isa_density(altitude_ft):
    """
    Returns ISA air density at a given altitude.
    
    Parameters:
    -----------
    altitude_ft : float or array, altitude in feet
    
    Returns:
    --------
    float or array : density in kg/m³
    """
    altitude_m = np.asarray(altitude_ft) * 0.3048
    
    # Exponential approximation
    rho = RHO_0 * np.exp(-altitude_m / 8500)
    return rho


def mach_to_tas(mach, altitude_ft):
    """
    Converts Mach number to TAS (True Airspeed).
    
    TAS = Mach × speed_of_sound
    speed_of_sound = sqrt(gamma × R × T)
    
    Parameters:
    -----------
    mach : float or array, Mach number
    altitude_ft : float or array, altitude in feet
    
    Returns:
    --------
    float or array : TAS in knots
    """
    mach = np.asarray(mach)
    T = get_isa_temperature(altitude_ft)
    
    # Speed of sound (m/s)
    a = np.sqrt(GAMMA * R_AIR * T)
    
    # TAS (m/s → kt)
    tas_ms = mach * a
    tas_kt = tas_ms / 0.514444
    
    return tas_kt


def cas_to_tas(cas_kt, altitude_ft):
    """
    Converts CAS (Calibrated Airspeed) to TAS (True Airspeed).
    
    Simplified formula (low speed, incompressible):
    TAS = CAS × sqrt(rho_0 / rho)
    
    Parameters:
    -----------
    cas_kt : float or array, CAS in knots
    altitude_ft : float or array, altitude in feet
    
    Returns:
    --------
    float or array : TAS in knots
    """
    cas_kt = np.asarray(cas_kt)
    rho = get_isa_density(altitude_ft)
    
    # Correction factor
    sigma = rho / RHO_0
    
    # TAS = CAS / sqrt(sigma)
    tas_kt = cas_kt / np.sqrt(sigma)
    
    return tas_kt


def extract_acars_tas(df):
    """
    Extracts TAS from ACARS messages.
    
    Logic:
    1. If TAS available → use directly
    2. If Mach available → convert using altitude (interpolated if needed)
    3. If CAS available → convert using altitude
    
    Parameters:
    -----------
    df : DataFrame with source, TAS, mach, CAS, altitude, timestamp columns
    
    Returns:
    --------
    dict : {
        'tas_values': list of (timestamp, tas_kt),
        'cruise_tas_median': float or None,
        'n_acars_points': int
    }
    """
    result = {
        'tas_values': [],
        'cruise_tas_median': None,
        'n_acars_points': 0
    }
    
    # Check if source column exists
    if 'source' not in df.columns:
        return result
    
    # Extract ACARS lines
    acars_mask = df['source'] == 'acars'
    if not acars_mask.any():
        return result
    
    acars_df = df[acars_mask].copy()
    result['n_acars_points'] = len(acars_df)
    
    tas_values = []
    
    # Prepare altitude interpolation for the whole dataframe once
    df_alt_interp = df['altitude'].interpolate(method='linear', limit_direction='both')
    
    for idx, row in acars_df.iterrows():
        timestamp = row['timestamp']
        tas = None
        
        # 1. TAS directly available
        if 'TAS' in df.columns and pd.notna(row.get('TAS')):
            tas = row['TAS']
        
        # If no TAS, we need altitude to convert Mach/CAS
        if tas is None:
            # Get altitude (from ACARS line or interpolated)
            if pd.notna(row.get('altitude')):
                alt = row['altitude']
            else:
                # Use interpolated altitude at this index
                alt = df_alt_interp.loc[idx]
                
                # If still NaN (rare), try timestamp interpolation
                if pd.isna(alt):
                    alt = _interpolate_altitude_at_timestamp(df, timestamp)
            
            if alt is not None and alt > 0:
                # 2. Mach available → convert
                if 'mach' in df.columns and pd.notna(row.get('mach')):
                    mach = row['mach']
                    tas = mach_to_tas(mach, alt)
                
                # 3. CAS available → convert
                elif 'CAS' in df.columns and pd.notna(row.get('CAS')):
                    cas = row['CAS']
                    tas = cas_to_tas(cas, alt)
        
        if tas is not None and np.isfinite(tas) and 100 < tas < 600:
            tas_values.append((timestamp, float(tas)))
    
    result['tas_values'] = tas_values
    
    # Calculate median TAS in cruise (high altitude points)
    if tas_values:
        # Filter points likely in cruise (TAS > 350 kt typical)
        cruise_tas = [t for _, t in tas_values if t > 350]
        if cruise_tas:
            result['cruise_tas_median'] = float(np.median(cruise_tas))
        elif tas_values:
            result['cruise_tas_median'] = float(np.median([t for _, t in tas_values]))
    
    return result


def _interpolate_altitude_at_timestamp(df, target_ts):
    """
    Interpolates altitude at a given timestamp from ADS-B lines.
    
    Parameters:
    -----------
    df : DataFrame with timestamp and altitude
    target_ts : target timestamp
    
    Returns:
    --------
    float : interpolated altitude, or None if impossible
    """
    # Filter lines with valid altitude (typically ADS-B)
    valid = df[df['altitude'].notna()].copy()
    
    if len(valid) == 0:
        return None
    
    # Find points before and after
    before = valid[valid['timestamp'] <= target_ts]
    after = valid[valid['timestamp'] >= target_ts]
    
    if len(before) == 0 and len(after) == 0:
        return None
    
    if len(before) == 0:
        return float(after.iloc[0]['altitude'])
    
    if len(after) == 0:
        return float(before.iloc[-1]['altitude'])
    
    # Linear interpolation
    p_before = before.iloc[-1]
    p_after = after.iloc[0]
    
    t_before = p_before['timestamp']
    t_after = p_after['timestamp']
    
    if t_before == t_after:
        return float(p_before['altitude'])
    
    # Time ratio
    total_dt = (t_after - t_before).total_seconds()
    dt = (target_ts - t_before).total_seconds()
    ratio = dt / total_dt
    
    alt = p_before['altitude'] + ratio * (p_after['altitude'] - p_before['altitude'])
    return float(alt)


def create_airspeed_column(df, acars_info):
    """
    Creates 'airspeed' column by propagating ACARS TAS over the cruise phase.
    
    Logic:
    - In cruise (CR phase or stable high altitude): airspeed = median ACARS TAS
    - Elsewhere: airspeed = NaN (ACROPOLE will use groundspeed)
    
    Parameters:
    -----------
    df : Trajectory DataFrame (with 'phase' column if available)
    acars_info : dict returned by extract_acars_tas()
    
    Returns:
    --------
    DataFrame with added 'airspeed' column
    """
    df = df.copy()
    
    # Initialize to NaN
    df['airspeed'] = np.nan
    
    cruise_tas = acars_info.get('cruise_tas_median')
    
    if cruise_tas is None:
        return df
    
    # Identify cruise phase
    if 'phase' in df.columns:
        # Use phase detection
        cruise_mask = df['phase'] == 'CR'
    else:
        # Fallback: altitude > 25000 ft and low vertical_rate
        cruise_mask = (
            (df['altitude'] > 25000) & 
            (df['vertical_rate'].abs() < 500)
        )
    
    # Propagate ACARS TAS over cruise
    df.loc[cruise_mask, 'airspeed'] = cruise_tas
    
    return df


# =============================================================================
# FONCTIONS DE NETTOYAGE
# =============================================================================

def deduplicate_by_second(df):
    """
    Keeps only one point per second (the first one).
    """
    df = df.copy()
    df['second'] = df['timestamp'].dt.floor('s')
    df = df.drop_duplicates(subset=['second'], keep='first')
    df = df.drop(columns=['second'])
    return df.reset_index(drop=True)


def filter_outliers(df):
    """
    Sets outliers to NaN.
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
    Interpolates numeric columns for gaps < 5 seconds.
    Does NOT create new rows, only fills existing NaNs
    if the time gap is small.
    """
    df = df.copy()
    
    # Calculate time gap with previous point
    df['dt'] = df['timestamp'].diff().dt.total_seconds()
    
    for col in INTERPOLATE_COLS:
        if col not in df.columns:
            continue
        
        # Identify NaNs
        is_nan = df[col].isna()
        
        if not is_nan.any():
            continue
        
        # For each NaN, check if gap is small
        nan_indices = df[is_nan].index
        
        for idx in nan_indices:
            # Check gap before and after
            pos = df.index.get_loc(idx)
            
            # Gap with previous point
            if pos > 0:
                dt_before = df.iloc[pos]['dt']
                if pd.notna(dt_before) and dt_before <= MAX_INTERPOLATION_GAP_SEC:
                    # Linear interpolation
                    # Find previous non-NaN value
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
                        # Linear interpolation
                        ratio = (pos - prev_pos) / (next_pos - prev_pos)
                        df.loc[idx, col] = prev_val + ratio * (next_val - prev_val)
    
    df = df.drop(columns=['dt'])
    return df


def interpolate_small_gaps_vectorized(df):
    """
    Faster vectorized version of interpolation.
    Interpolates NaNs only if time gap is < 5s.
    """
    df = df.copy()
    
    # Calculate time gaps
    dt = df['timestamp'].diff().dt.total_seconds()
    
    # Identify areas where interpolation is allowed (gap < 5s)
    can_interpolate = dt <= MAX_INTERPOLATION_GAP_SEC
    
    for col in INTERPOLATE_COLS:
        if col not in df.columns:
            continue
        
        # Mask of NaNs in this column
        is_nan = df[col].isna()
        
        if not is_nan.any():
            continue
        
        # Interpolate only if gap is small
        # Create temporary series for interpolation
        temp = df[col].copy()
        
        # Linear interpolation on the whole series
        temp_interp = temp.interpolate(method='linear', limit_direction='both')
        
        # Apply interpolation only where:
        # 1. Original value was NaN
        # 2. Time gap is acceptable
        mask = is_nan & can_interpolate
        df.loc[mask, col] = temp_interp.loc[mask]
    
    return df


def detect_flight_phases_openap(df):
    """
    Detects flight phases using OpenAP FlightPhase.
    Returns: DataFrame with added 'phase' column
    """
    df = df.copy()
    
    try:
        from openap import FlightPhase
        
        # Prepare data
        ts = df['timestamp'].values
        alt = df['altitude'].fillna(0).values
        spd = df['groundspeed'].fillna(0).values
        roc = df['vertical_rate'].fillna(0).values
        
        # Convert timestamps to seconds from start
        t0 = pd.Timestamp(ts[0])
        ts_seconds = np.array([(pd.Timestamp(t) - t0).total_seconds() for t in ts])
        
        # Create FlightPhase object
        fp = FlightPhase()
        fp.set_trajectory(ts_seconds, alt, spd, roc)
        
        # Get labels
        phases = fp.phaselabel()
        
        df['phase'] = phases
        return df
        
    except ImportError:
        # OpenAP not installed, use fallback
        return detect_flight_phases_simple(df)
    except Exception as e:
        # Other error, use fallback
        return detect_flight_phases_simple(df)


def detect_flight_phases_simple(df):
    """
    Simple fallback for phase detection if OpenAP is not available.
    Based on altitude and vertical_rate.
    
    GND: altitude < 1500 ft and groundspeed < 100 kt
    CL:  vertical_rate > 500 ft/min (significant climb)
    DE:  vertical_rate < -500 ft/min (significant descent)
    CR:  rest (cruise)
    LVL: not used in this fallback
    """
    df = df.copy()
    
    alt = df['altitude'].fillna(0).values
    vr = df['vertical_rate'].fillna(0).values
    gs = df['groundspeed'].fillna(0).values
    
    # Smooth vertical_rate with rolling mean
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
# SINGLE FLIGHT PROCESSING
# =============================================================================

def clean_single_trajectory(args):
    """
    Cleans a single trajectory.
    
    Parameters:
    -----------
    args : tuple (input_path, output_path)
    
    Returns:
    --------
    dict with cleaning stats
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
        # Load
        df = pd.read_parquet(input_path)
        stats['n_points_raw'] = len(df)
        
        # 1. Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 2. EXTRACT ACARS DATA BEFORE CLEANING
        #    (to retrieve TAS/Mach/CAS before potentially losing them)
        acars_info = extract_acars_tas(df)
        stats['n_acars_points'] = acars_info['n_acars_points']
        stats['cruise_tas'] = acars_info['cruise_tas_median']
        
        # 3. Deduplicate (1 point/second)
        n_before = len(df)
        df = deduplicate_by_second(df)
        stats['n_duplicates_removed'] = n_before - len(df)
        
        # 4. Filter outliers
        if 'groundspeed' in df.columns:
            n_gs_before = df['groundspeed'].isna().sum()
        if 'vertical_rate' in df.columns:
            n_vr_before = df['vertical_rate'].isna().sum()
        
        df = filter_outliers(df)
        
        if 'groundspeed' in df.columns:
            stats['n_outliers_gs'] = df['groundspeed'].isna().sum() - n_gs_before
        if 'vertical_rate' in df.columns:
            stats['n_outliers_vr'] = df['vertical_rate'].isna().sum() - n_vr_before
        
        # 5. Interpolate small gaps
        n_nan_before = df[INTERPOLATE_COLS].isna().sum().sum() if all(c in df.columns for c in INTERPOLATE_COLS[:3]) else 0
        df = interpolate_small_gaps_vectorized(df)
        n_nan_after = df[INTERPOLATE_COLS].isna().sum().sum() if all(c in df.columns for c in INTERPOLATE_COLS[:3]) else 0
        stats['n_interpolated'] = n_nan_before - n_nan_after
        
        # 6. Detect flight phases
        try:
            df = detect_flight_phases_openap(df)
        except:
            df = detect_flight_phases_simple(df)
        
        # 7. CREATE AIRSPEED COLUMN (Propagated ACARS TAS in cruise)
        df = create_airspeed_column(df, acars_info)
        
        stats['n_points_clean'] = len(df)
        
        # Save
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
    print("TRAJECTORY CLEANING - PRC Data Challenge 2025")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List input files (only prc* trajectories)
    input_dir = Path(args.input)
    input_files = list(input_dir.glob('prc*.parquet'))
    
    # If no prc* files, try all parquet files
    if len(input_files) == 0:
        input_files = [f for f in input_dir.glob('*.parquet') 
                       if not any(x in f.stem for x in ['flightlist', 'fuel', 'apt', 'stats'])]
    
    print(f"\nTrajectory files found: {len(input_files)}")
    
    if args.max_flights:
        input_files = input_files[:args.max_flights]
        print(f"Limited to {args.max_flights} files")
    
    # Prepare tasks
    tasks = []
    for input_path in input_files:
        output_path = output_dir / input_path.name
        tasks.append((str(input_path), str(output_path)))
    
    # Parallel processing
    print(f"\nProcessing with {args.workers} workers...")
    
    all_stats = []
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(clean_single_trajectory, tasks),
            total=len(tasks),
            desc="Cleaning"
        ))
    
    all_stats = results
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    
    df_stats = pd.DataFrame(all_stats)
    
    n_success = (df_stats['status'] == 'success').sum()
    n_error = len(df_stats) - n_success
    
    print(f"\nProcessed flights: {n_success} / {len(df_stats)}")
    print(f"Errors: {n_error}")
    
    if n_success > 0:
        success_stats = df_stats[df_stats['status'] == 'success']
        
        print(f"\nPoints:")
        print(f"  Total raw:   {success_stats['n_points_raw'].sum():,}")
        print(f"  Total clean: {success_stats['n_points_clean'].sum():,}")
        print(f"  Reduction:   {(1 - success_stats['n_points_clean'].sum() / success_stats['n_points_raw'].sum()) * 100:.1f}%")
        
        print(f"\nCleaning:")
        print(f"  Duplicates removed: {success_stats['n_duplicates_removed'].sum():,}")
        print(f"  Outliers GS:        {success_stats['n_outliers_gs'].sum():,}")
        print(f"  Outliers VR:        {success_stats['n_outliers_vr'].sum():,}")
        print(f"  Interpolated pts:   {success_stats['n_interpolated'].sum():,}")
    
    if n_error > 0:
        print(f"\nErrors:")
        for _, row in df_stats[df_stats['status'] != 'success'].head(10).iterrows():
            print(f"  {row['flight_id']}: {row['status']}")
    
    # Save stats
    stats_path = output_dir / 'cleaning_stats.parquet'
    df_stats.to_parquet(stats_path, index=False)
    print(f"\nStats saved: {stats_path}")


if __name__ == "__main__":
    main()