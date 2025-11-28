"""
Feature Engineering Pipeline for PRC Data Challenge 2025.

Integrates:
- PRC2024 Mass Estimation (team_likable_jelly) via energy rate
- ACROPOLE (supported types) + OpenAP (fallback) for fuel flow estimation
- Iterative calculation with mass correction

Multiprocessing with 8 workers.

Usage:
    python feature_engineering.py \
        --trajectories prc_data/flight_train \
        --flightlist prc_data/flightlist_train.parquet \
        --fuel prc_data/fuel_train.parquet \
        --output data/features_train.parquet \
        --workers 8
"""

# --- FIX CRASH MAC (CRITICAL) ---
# Disable GPU (Metal) BEFORE any import to avoid multiprocessing conflicts
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # For standard TF
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Fix Protobuf/PyArrow/TensorFlow conflict (suggested by screenshot)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf
    # Hide GPUs from TF
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass
except Exception:
    pass
# --------------------------------

import argparse
import sys
import warnings
from pathlib import Path
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from datetime import datetime

# IMPORTANT: Use 'spawn' instead of 'fork' to avoid segfaults
# with ACROPOLE after processing many flights
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already configured

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
# Specifically ignore pkg_resources warnings that pollute output
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
# Ignore A20N/A21N not supported by Acropole warning (normal, fallback to OpenAP)
warnings.filterwarnings("ignore", message=".*Aircraft type .* flight_typecode not supported.*")
os.environ["PYTHONWARNINGS"] = "ignore"

# =============================================================================
# CONSTANTS
# =============================================================================

# Physical constants
G = 9.80665  # m/s²
FT2M = 0.3048
KT2MS = 0.514444
FPM2MS = 0.00508

# Types supported by ACROPOLE (check/adjust depending on installed version)
ACROPOLE_TYPES = {
    'A319', 'A320', 'A321', 'A20N', 'A21N',  # A320 family
    'B737', 'B738', 'B739',                   # B737 family
    'A332', 'A333',                           # A330 family
}

# Types supported by OpenAP
OPENAP_TYPES = {
    'A319', 'A320', 'A321', 'A20N', 'A21N',
    'A332', 'A333', 'A339', 'A359', 'A35K',
    'B737', 'B738', 'B739', 'B38M', 'B39M',
    'B744', 'B748', 'B752', 'B763', 'B772', 'B773', 'B77W', 'B788', 'B789', 'B78X',
    'E190', 'E195', 'E290', 'E295',
    'C560', 'CRJ9', 'DH8D', 'ATR72',
}

# Typical mass in % of MTOW for estimation (fallback)
TYPICAL_MASS_RATIO = 0.85


# =============================================================================
# ESTIMATOR INITIALIZATION (lazy loading)
# =============================================================================

_acropole_estimator = None
_openap_fuelflows = {}
_openap_props = {}


def get_acropole():
    """Lazy loading of ACROPOLE estimator."""
    global _acropole_estimator
    if _acropole_estimator is None:
        try:
            # Force usage of local Acropole (with warning fix)
            import sys
            local_acropole_path = os.path.join(os.getcwd(), 'Acropole')
            if local_acropole_path not in sys.path:
                sys.path.insert(0, local_acropole_path)

            # --- FIX CRASH MAC ---
            # Disable GPU (Metal) to avoid multiprocessing conflicts
            import tensorflow as tf
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            # ---------------------

            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
            
            from acropole import FuelEstimator
            _acropole_estimator = FuelEstimator()
        except ImportError as e:
            print(f"WARNING: ACROPOLE not installed, fallback to OpenAP. Error: {e}")
            _acropole_estimator = False
        except Exception as e:
            print(f"WARNING: ACROPOLE init error: {e}")
            _acropole_estimator = False
    return _acropole_estimator


def get_openap_fuelflow(typecode):
    """Lazy loading of OpenAP FuelFlow by type."""
    global _openap_fuelflows, _openap_props
    
    if typecode not in _openap_fuelflows:
        try:
            from openap import FuelFlow, prop
            
            # Get aircraft props
            ac = prop.aircraft(typecode)
            _openap_props[typecode] = ac
            
            # Create estimator
            _openap_fuelflows[typecode] = FuelFlow(typecode, use_synonym=True)
        except Exception as e:
            _openap_fuelflows[typecode] = None
            _openap_props[typecode] = None
    
    return _openap_fuelflows.get(typecode), _openap_props.get(typecode)


# =============================================================================
# MASS ESTIMATION (simple fallback 85% MTOW)
# =============================================================================


def estimate_initial_mass(df, typecode):
    """
    Returns a simple initial mass estimation: 85% MTOW.
    
    Note: A more sophisticated method (PRC2024) could be implemented,
    but for now we use the simple fallback because:
    - ACROPOLE handles mass internally
    - OpenAP works well with 85% MTOW as an approximation
    
    Parameters:
    -----------
    df : Trajectory DataFrame (unused, for compatibility)
    typecode : str, aircraft type
    
    Returns:
    --------
    float : estimated mass in kg, or None if type not supported
    """
    try:
        from openap import prop
        ac = prop.aircraft(typecode)
        mtow = ac.get('mtow', 75000)
        return mtow * TYPICAL_MASS_RATIO  # 85%
    except:
        return None


# =============================================================================
# FUEL FLOW ESTIMATION
# =============================================================================

def estimate_fuel_acropole(df):
    """
    Estimates fuel flow using ACROPOLE.
    
    Uses 'airspeed' (TAS) column if available, otherwise 'groundspeed'.
    ACROPOLE is more accurate with TAS, especially in cruise with wind.
    
    Parameters:
    -----------
    df : DataFrame with columns: typecode, groundspeed, altitude, vertical_rate
         and optionally 'airspeed' (TAS extracted from ACARS data)
    
    Returns:
    --------
    Array of fuel_flow in kg/s
    """
    fe = get_acropole()
    
    if fe is False or fe is None:
        return np.full(len(df), np.nan)
    
    try:
        # Prepare DataFrame for ACROPOLE
        input_df = df[['typecode', 'groundspeed', 'altitude', 'vertical_rate']].copy()
        
        # IMPORTANT: Interpolate residual NaNs in groundspeed and vertical_rate
        # These NaNs can exist if the time gap was > 5s during cleaning
        input_df['groundspeed'] = input_df['groundspeed'].interpolate(method='linear', limit_direction='both')
        input_df['vertical_rate'] = input_df['vertical_rate'].interpolate(method='linear', limit_direction='both')
        input_df['altitude'] = input_df['altitude'].interpolate(method='linear', limit_direction='both')
        
        # Fill remaining NaNs (start/end) with ffill/bfill
        input_df['groundspeed'] = input_df['groundspeed'].ffill().bfill()
        input_df['vertical_rate'] = input_df['vertical_rate'].fillna(0)  # 0 = level flight
        input_df['altitude'] = input_df['altitude'].ffill().bfill()
        
        # CORRECTION: Filter invalid values causing segfaults
        # Negative altitude → 0 (on ground)
        input_df.loc[input_df['altitude'] < 0, 'altitude'] = 0
        # Groundspeed too low → minimum 50 kt
        input_df.loc[input_df['groundspeed'] < 50, 'groundspeed'] = 50
        # Groundspeed too high → maximum 600 kt
        input_df.loc[input_df['groundspeed'] > 600, 'groundspeed'] = 600
        # Altitude too high → maximum 45000 ft
        input_df.loc[input_df['altitude'] > 45000, 'altitude'] = 45000
        
        # Create airspeed column: TAS if available, else GS
        if 'airspeed' in df.columns:
            # airspeed = TAS (ACARS) where available, else interpolated groundspeed
            input_df['airspeed'] = df['airspeed'].fillna(input_df['groundspeed'])
        else:
            # No TAS available, ACROPOLE will use groundspeed by default
            input_df['airspeed'] = input_df['groundspeed']
            
        # Create second column (time in seconds from start)
        # Required for Acropole to calculate derivatives (accelerations)
        if 'timestamp' in df.columns:
            t0 = df['timestamp'].min()
            input_df['second'] = (df['timestamp'] - t0).dt.total_seconds()
        else:
            # Fallback if no timestamp (should not happen)
            input_df['second'] = np.arange(len(df))
        
        # Estimate with airspeed and second
        result = fe.estimate(
            input_df,
            typecode='typecode',
            groundspeed='groundspeed',
            altitude='altitude',
            vertical_rate='vertical_rate',
            airspeed='airspeed',
            second='second'
        )
        
        return result['fuel_flow'].values
    
    except Exception as e:
        # Fallback without airspeed
        try:
            input_df = df[['typecode', 'groundspeed', 'altitude', 'vertical_rate']].copy()
            input_df['groundspeed'] = input_df['groundspeed'].interpolate(method='linear').ffill().bfill()
            input_df['vertical_rate'] = input_df['vertical_rate'].interpolate(method='linear').fillna(0)
            input_df['altitude'] = input_df['altitude'].interpolate(method='linear').ffill().bfill()
            
            # Same corrections
            input_df.loc[input_df['altitude'] < 0, 'altitude'] = 0
            input_df.loc[input_df['groundspeed'] < 50, 'groundspeed'] = 50
            input_df.loc[input_df['groundspeed'] > 600, 'groundspeed'] = 600
            input_df.loc[input_df['altitude'] > 45000, 'altitude'] = 45000
            
            result = fe.estimate(input_df)
            return result['fuel_flow'].values
        except:
            return np.full(len(df), np.nan)


def estimate_fuel_openap_vectorized(df, typecode, estimated_mass=None):
    """
    Fuel flow estimation with OpenAP - VECTORIZED VERSION.
    
    2-pass approach (exactly as per OpenAP docs):
    1. First pass with initial mass (PRC2024 estimate or 85% MTOW fallback)
    2. Mass correction at each step (mass -= cumulative fuel consumed)
    3. Second pass with corrected mass
    
    Parameters:
    -----------
    df : Trajectory DataFrame
    typecode : str, aircraft type
    estimated_mass : float, estimated initial mass (if None, will be estimated or fallback)
    
    Returns:
    --------
    tuple: (fuel_flow array in kg/s, total_fuel in kg, mass0 used, mass_source)
    """
    ff_model, ac_props = get_openap_fuelflow(typecode)
    
    if ff_model is None:
        return np.full(len(df), np.nan), np.nan, np.nan, 'none'
    
    try:
        # Aircraft parameters
        mtow = ac_props.get('mtow', 75000)
        oew = ac_props.get('oew', mtow * 0.5)
        
        # --- Determine initial mass ---
        if estimated_mass is not None and oew * 0.8 < estimated_mass < mtow * 1.1:
            # Use provided mass (already estimated)
            mass0 = estimated_mass
            mass_source = 'prc2024'
        else:
            # Try to use trained mass model
            mass_model_path = Path('data/mass_model.pkl')
            model_mass = None
            
            if mass_model_path.exists():
                try:
                    import pickle
                    with open(mass_model_path, 'rb') as f:
                        mass_model = pickle.load(f)
                    
                    if typecode in mass_model:
                        model_info = mass_model[typecode]
                        duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
                        
                        if model_info['method'] == 'linear':
                            model_mass = model_info['slope'] * duration + model_info['intercept']
                        else:
                            model_mass = model_info['mass_mean']
                except:
                    pass
            
            if model_mass is not None and oew * 0.8 < model_mass < mtow * 1.1:
                mass0 = model_mass
                mass_source = 'mass_model_v1'
            else:
                # Fallback: 85% of MTOW
                mass0 = mtow * TYPICAL_MASS_RATIO
                mass_source = 'fallback_85pct'
        
        # Flight data - interpolate NaNs instead of setting to 0
        tas = df['groundspeed'].interpolate(method='linear').ffill().bfill().values
        alt = df['altitude'].interpolate(method='linear').ffill().bfill().values
        vs = df['vertical_rate'].interpolate(method='linear').fillna(0).values if 'vertical_rate' in df.columns else np.zeros(len(df))
        
        # If airspeed (ACARS TAS) is available, use it in cruise
        if 'airspeed' in df.columns:
            airspeed = df['airspeed'].values
            # Replace tas with airspeed where available
            mask = ~np.isnan(airspeed)
            tas[mask] = airspeed[mask]
        
        # dt = time step in seconds (bfill for first point)
        dt = df['timestamp'].diff().bfill().dt.total_seconds().values
        
        # --- First pass with initial mass ---
        fuel_flow_initial = ff_model.enroute(
            mass=mass0,
            tas=tas,
            alt=alt,
            vs=vs
        ).flatten()
        
        # Replace NaNs with 0 to avoid propagation in cumsum
        fuel_flow_initial_clean = np.nan_to_num(fuel_flow_initial, nan=0.0)
        
        # --- Mass correction at each step ---
        mass = mass0 - (fuel_flow_initial_clean * dt).cumsum()
        
        # Ensure mass doesn't drop below OEW
        mass = np.maximum(mass, oew)
        
        # --- Second pass with corrected mass ---
        fuel_flow = ff_model.enroute(
            mass=mass,
            tas=tas,
            alt=alt,
            vs=vs
        ).flatten()
        
        # Total fuel consumed
        total_fuel = np.sum(fuel_flow * dt)
        
        return fuel_flow, total_fuel, mass0, mass_source
    
    except Exception as e:
        return np.full(len(df), np.nan), np.nan, np.nan, 'error'


# =============================================================================
# SEGMENT FEATURE CALCULATION
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in km between two points."""
    R = 6371.0  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def compute_segment_features(segment_df, segment_info):
    """
    Calculates features for a segment.
    
    Parameters:
    -----------
    segment_df : DataFrame of trajectory points in this segment
    segment_info : dict with flight_id, idx, etc.
    
    Returns:
    --------
    dict of features
    """
    features = {
        'flight_id': segment_info['flight_id'],
        'idx': segment_info['idx'],
    }
    
    n_points = len(segment_df)
    features['n_points'] = n_points
    
    if n_points == 0:
        return features
    
    # --- Temporal ---
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
            # Total distance flown
            distances = haversine_distance(
                lat.iloc[:-1].values, lon.iloc[:-1].values,
                lat.iloc[1:].values, lon.iloc[1:].values
            )
            features['distance_km'] = np.sum(distances)
            
            # Direct distance (great circle)
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
    
    # --- Track (heading) ---
    if 'track' in segment_df.columns:
        track = segment_df['track'].dropna()
        if len(track) > 1:
            # Total heading change (accounting for wrap-around)
            track_diff = np.diff(track.values)
            track_diff = np.abs((track_diff + 180) % 360 - 180)
            features['track_change_total'] = np.sum(track_diff)
            features['track_std'] = track.std()
    
    # --- Estimated Fuel Flow ---
    if 'fuel_flow' in segment_df.columns:
        ff = segment_df['fuel_flow'].dropna()
        if len(ff) > 0:
            features['fuel_flow_mean'] = ff.mean()
            features['fuel_flow_min'] = ff.min()
            features['fuel_flow_max'] = ff.max()
            features['fuel_flow_std'] = ff.std() if len(ff) > 1 else 0
            
            # Total estimated fuel (trapezoidal integration)
            if 'timestamp' in segment_df.columns and len(ff) >= 2:
                ff_valid = segment_df.dropna(subset=['fuel_flow', 'timestamp'])
                if len(ff_valid) >= 2:
                    dt = ff_valid['timestamp'].diff().dt.total_seconds().iloc[1:].values
                    ff_vals = ff_valid['fuel_flow'].values
                    # Trapezoid
                    fuel_estimated = np.sum((ff_vals[:-1] + ff_vals[1:]) / 2 * dt)
                    features['fuel_estimated_kg'] = fuel_estimated
    
    # --- Flight Phase (use the one computed by clean_trajectories if available) ---
    if 'phase' in segment_df.columns:
        # Use majority phase in segment
        phase_counts = segment_df['phase'].value_counts()
        if len(phase_counts) > 0:
            features['phase'] = phase_counts.index[0]
            
            # Percentage of each phase in segment
            for phase in ['GND', 'CL', 'CR', 'DE', 'LVL']:
                pct = (segment_df['phase'] == phase).mean()
                features[f'phase_{phase}_pct'] = pct
        else:
            features['phase'] = 'NA'
    else:
        # Fallback: simple detection based on alt_delta and vrate_mean
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
# SINGLE FLIGHT PROCESSING
# =============================================================================

def process_single_flight(args):
    """
    Processes a full flight: loads trajectory, estimates fuel, calculates segment features.
    
    Parameters:
    -----------
    args : tuple (flight_id, typecode, traj_path, segments_df, flight_info)
    
    Returns:
    --------
    list of dict (features per segment)
    """
    flight_id, typecode, traj_path, segments_df, flight_info = args
    
    results = []
    
    try:
        # Load trajectory
        if not os.path.exists(traj_path):
            return results
        
        traj_df = pd.read_parquet(traj_path)
        traj_df = traj_df.sort_values('timestamp').reset_index(drop=True)
        
        # --- INTERPOLATE MISSING DATA ---
        # Some rows (ACARS) miss groundspeed, vertical_rate, track
        # Interpolate from neighboring ADS-B points
        for col in ['groundspeed', 'altitude', 'vertical_rate', 'track']:
            if col in traj_df.columns:
                traj_df[col] = traj_df[col].interpolate(method='linear', limit_direction='both')
                traj_df[col] = traj_df[col].ffill().bfill()
        
        # vertical_rate : remaining NaNs → 0 (level flight)
        if 'vertical_rate' in traj_df.columns:
            traj_df['vertical_rate'] = traj_df['vertical_rate'].fillna(0)
        
        # --- FIX INVALID VALUES ---
        # Negative altitudes (ADS-B errors) → 0
        if 'altitude' in traj_df.columns:
            traj_df.loc[traj_df['altitude'] < 0, 'altitude'] = 0
            traj_df.loc[traj_df['altitude'] > 50000, 'altitude'] = 50000
        
        # Invalid Groundspeed
        if 'groundspeed' in traj_df.columns:
            traj_df.loc[traj_df['groundspeed'] < 0, 'groundspeed'] = 0
            traj_df.loc[traj_df['groundspeed'] > 700, 'groundspeed'] = 700
        
        # --- Estimate fuel flow ---
        mass_estimated = np.nan
        mass_source = 'none'
        fuel_calculated = False
        
        if typecode in ACROPOLE_TYPES:
            # Acropole needs typecode column
            traj_df['typecode'] = typecode
            acropole_result = estimate_fuel_acropole(traj_df)
            # Check if we got valid results (not all NaNs)
            if not np.all(np.isnan(acropole_result)):
                traj_df['fuel_flow'] = acropole_result
                traj_df['fuel_source'] = 'acropole'
                # ACROPOLE handles mass internally, no need to estimate
                mass_estimated = np.nan
                mass_source = 'acropole_internal'
                fuel_calculated = True
        
        # Fallback to OpenAP if Acropole failed or wasn't applicable
        if not fuel_calculated and typecode in OPENAP_TYPES:
            fuel_flow, total_fuel, mass_estimated, mass_source = estimate_fuel_openap_vectorized(traj_df, typecode)
            traj_df['fuel_flow'] = fuel_flow
            traj_df['fuel_source'] = 'openap'
            fuel_calculated = True
            
        if not fuel_calculated:
            traj_df['fuel_flow'] = np.nan
            traj_df['fuel_source'] = 'none'
        
        # --- Flight info from flightlist ---
        origin_icao = flight_info.get('origin_icao', '')
        destination_icao = flight_info.get('destination_icao', '')
        takeoff_time = flight_info.get('takeoff')
        landed_time = flight_info.get('landed')
        
        # Total flight duration
        if takeoff_time and landed_time:
            try:
                flight_duration_min = (pd.to_datetime(landed_time) - pd.to_datetime(takeoff_time)).total_seconds() / 60
            except:
                flight_duration_min = np.nan
        else:
            flight_duration_min = np.nan
        
        # Total segments for this flight
        n_segments_total = len(segments_df)
        
        # --- Process each segment ---
        for seg_idx, (_, seg_row) in enumerate(segments_df.iterrows()):
            # Real columns: idx, flight_id, start, end, fuel_kg
            segment_idx = seg_row['idx']
            seg_start = pd.to_datetime(seg_row['start'])
            seg_end = pd.to_datetime(seg_row['end'])
            
            # Filter segment points
            mask = (traj_df['timestamp'] >= seg_start) & (traj_df['timestamp'] <= seg_end)
            segment_traj = traj_df[mask].copy()
            
            # Calculate features
            segment_info = {
                'flight_id': flight_id,
                'idx': segment_idx,
            }
            
            features = compute_segment_features(segment_traj, segment_info)
            
            # Add segment info
            features['typecode'] = typecode
            features['start'] = seg_start
            features['end'] = seg_end
            
            # Flight info
            features['origin_icao'] = origin_icao
            features['destination_icao'] = destination_icao
            features['flight_duration_min'] = flight_duration_min
            
            # Estimated mass (PRC2024 style)
            features['mass_estimated_kg'] = mass_estimated
            features['mass_source'] = mass_source
            
            # Relative position of segment in flight
            features['segment_position'] = seg_idx / max(n_segments_total - 1, 1)
            features['n_segments_total'] = n_segments_total
            features['is_first_segment'] = 1 if seg_idx == 0 else 0
            features['is_last_segment'] = 1 if seg_idx == n_segments_total - 1 else 0
            
            # Time of day (UTC)
            features['hour_utc'] = seg_start.hour
            features['day_of_week'] = seg_start.dayofweek
            
            # Keep fuel_kg if available (train set)
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
    """Initializes caches in each worker."""
    global _acropole_estimator, _openap_fuelflows, _openap_props
    import warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    
    # Reset caches for this worker
    _acropole_estimator = None
    _openap_fuelflows = {}
    _openap_props = {}


def save_flight_checkpoint(flight_id, features_list, checkpoint_dir):
    """Saves flight features to a checkpoint file."""
    if not features_list:
        return
    checkpoint_path = Path(checkpoint_dir) / f"features_{flight_id}.parquet"
    df = pd.DataFrame(features_list)
    df.to_parquet(checkpoint_path, index=False)


def process_and_save(args_with_checkpoint):
    """Wrapper that processes a flight and saves immediately."""
    import signal
    
    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError("Timeout reached")

    task, checkpoint_dir = args_with_checkpoint
    flight_id = task[0]
    typecode = task[1]
    
    # Set timeout (120 seconds)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(120)
    
    try:
        # Process flight
        features_list = process_single_flight(task)
        
        # Disable alarm if finished in time
        signal.alarm(0)
        
        # Save immediately
        if features_list:
            save_flight_checkpoint(flight_id, features_list, checkpoint_dir)
        
        return flight_id, len(features_list), None
        
    except TimeoutError:
        return flight_id, 0, "TIMEOUT (120s)"
    except Exception as e:
        signal.alarm(0) # Disable alarm on error
        return flight_id, 0, str(e)


def get_completed_flights(checkpoint_dir):
    """Returns the set of already processed flight_ids."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return set()
    
    completed = set()
    for f in checkpoint_path.glob("features_*.parquet"):
        # Extract flight_id from name: features_prc770870642.parquet -> prc770870642
        flight_id = f.stem.replace("features_", "")
        completed.add(flight_id)
    
    return completed


def concat_checkpoints(checkpoint_dir, output_path):
    """Concatenates all checkpoint files into a single file."""
    checkpoint_path = Path(checkpoint_dir)
    
    all_files = list(checkpoint_path.glob("features_*.parquet"))
    if not all_files:
        print("No checkpoint files found!")
        return None
    
    print(f"Concatenating {len(all_files)} files...")
    
    dfs = []
    for f in tqdm(all_files, desc="Reading"):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")
    
    if not dfs:
        return None
    
    features_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure idx is properly typed
    if 'idx' in features_df.columns:
        features_df['idx'] = features_df['idx'].astype(int)
    
    # Sort by flight_id and idx
    if 'flight_id' in features_df.columns and 'idx' in features_df.columns:
        features_df = features_df.sort_values(['flight_id', 'idx']).reset_index(drop=True)
    
    # Save
    features_df.to_parquet(output_path, index=False)
    
    return features_df


def main():
    parser = argparse.ArgumentParser(description='Feature Engineering PRC Challenge')
    parser.add_argument('--trajectories', required=True, help='Directory with trajectory files (prc*.parquet)')
    parser.add_argument('--flightlist', required=True, help='flightlist.parquet file')
    parser.add_argument('--fuel', required=True, help='fuel_train.parquet or fuel_rank_submission.parquet file')
    parser.add_argument('--output', required=True, help='Output features.parquet file')
    parser.add_argument('--workers', type=int, default=cpu_count(), help='Number of workers')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallelism (debug)')
    parser.add_argument('--max-flights', type=int, default=None, help='Limit number of flights (debug)')
    
    # New options for checkpoint/resume
    parser.add_argument('--checkpoint-dir', default=None, 
                        help='Directory to save checkpoints (1 file per flight)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip flights already in checkpoint-dir')
    parser.add_argument('--concat-only', action='store_true',
                        help='Only concatenate existing results')
    parser.add_argument('--ignore-flights', type=str, default="", help='List of flights to ignore (comma separated)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FEATURE ENGINEERING - PRC Data Challenge 2025")
    print("=" * 60)
    print(f"Workers: {args.workers}")
    
    # --- Configure checkpoint ---
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint dir: {checkpoint_dir}")
    else:
        # Create temporary directory next to output file
        checkpoint_dir = Path(args.output).parent / "checkpoints_temp"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint dir (auto): {checkpoint_dir}")
    
    # --- Concat-only mode ---
    if args.concat_only:
        print(f"\nCONCAT-ONLY mode: concatenating checkpoints...")
        features_df = concat_checkpoints(checkpoint_dir, args.output)
        if features_df is not None:
            print(f"\nSaved: {args.output}")
            print(f"Segments: {len(features_df)}")
            print(f"Size: {os.path.getsize(args.output) / 1024**2:.1f} MB")
        return
    
    # --- Load metadata ---
    t0 = datetime.now()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading flightlist: {args.flightlist}")
    flightlist = pd.read_parquet(args.flightlist)
    print(f"  Flights: {len(flightlist)}")
    print(f"  Columns: {flightlist.columns.tolist()}")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading fuel: {args.fuel}")
    fuel_df = pd.read_parquet(args.fuel)
    print(f"  Segments: {len(fuel_df)}")
    
    # Show columns for diagnostics
    print(f"  Fuel columns: {fuel_df.columns.tolist()}")
    
    # --- Check already processed flights (resume mode) ---
    completed_flights = set()
    if args.resume:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning checkpoints...")
        completed_flights = get_completed_flights(checkpoint_dir)
        print(f"RESUME mode: {len(completed_flights)} flights already processed")
    
    # --- Prepare tasks ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Preparing tasks...")
    trajectories_dir = Path(args.trajectories)
    
    # Group segments by flight
    segments_by_flight = fuel_df.groupby('flight_id')
    
    # Create dict with all flightlist info
    flightlist_dict = flightlist.set_index('flight_id').to_dict('index')
    
    tasks = []
    missing_traj = 0
    skipped_completed = 0
    
    for flight_id, segments in segments_by_flight:
        # Skip if already processed (resume mode)
        if flight_id in completed_flights:
            skipped_completed += 1
            continue
        
        # Get flight info
        flight_info = flightlist_dict.get(flight_id, {})
        typecode = flight_info.get('aircraft_type', 'UNKNOWN')
        
        # Search trajectory file
        traj_file = trajectories_dir / f"{flight_id}.parquet"
        
        if not traj_file.exists():
            missing_traj += 1
            continue
        
        tasks.append((flight_id, typecode, str(traj_file), segments, flight_info))
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tasks prepared: {len(tasks)}")
    print(f"Missing trajectories: {missing_traj}")
    if args.resume:
        print(f"Skipped flights (already processed): {skipped_completed}")
    
    if args.max_flights:
        tasks = tasks[:args.max_flights]
        print(f"Limited to {args.max_flights} flights")
    
    if len(tasks) == 0:
        print("\nNo flights to process!")
        if args.resume and completed_flights:
            print("All flights are already processed. Use --concat-only to concatenate.")
        return
    
    # --- Type statistics ---
    type_counts = {}
    for _, typecode, _, _, _ in tasks:
        type_counts[typecode] = type_counts.get(typecode, 0) + 1
    
    acropole_count = sum(v for k, v in type_counts.items() if k in ACROPOLE_TYPES)
    openap_count = sum(v for k, v in type_counts.items() if k in OPENAP_TYPES and k not in ACROPOLE_TYPES)
    unknown_count = sum(v for k, v in type_counts.items() if k not in OPENAP_TYPES)
    
    print(f"\nAircraft types:")
    print(f"  ACROPOLE: {acropole_count} flights ({acropole_count/len(tasks)*100:.1f}%)")
    print(f"  OpenAP: {openap_count} flights ({openap_count/len(tasks)*100:.1f}%)")
    print(f"  Not supported: {unknown_count} flights ({unknown_count/len(tasks)*100:.1f}%)")
    
    # --- Processing with checkpoint saving ---
    print(f"\nProcessing with {args.workers} workers...")
    print(f"Each flight is saved to: {checkpoint_dir}/")
    print("(You can interrupt with Ctrl+C and resume with --resume)")
    
    # Prepare tasks with checkpoint_dir
    tasks_with_checkpoint = [(task, str(checkpoint_dir)) for task in tasks]
    
    n_processed = 0
    n_segments = 0
    errors = []
    
    try:
        if args.no_parallel or args.workers <= 1:
            # Sequential mode (debug or max stability)
            print(f"Sequential mode (Workers={args.workers}) - Max stability")
            
            # Initialize environment (important to load Acropole correctly)
            _init_worker()
            
            for i, task_cp in enumerate(tqdm(tasks_with_checkpoint, desc="Flights (Seq)")):
                flight_id = task_cp[0][0]
                
                # Write current flight for crash debug
                with open("current_flight.txt", "w") as f:
                    f.write(flight_id)
                    f.flush()
                    os.fsync(f.fileno())
                
                flight_id, n_seg, error = process_and_save(task_cp)
                
                n_processed += 1
                n_segments += n_seg
                
                if error:
                    errors.append(f"{flight_id}: {error}")
        else:
            # Parallel mode with timeout per flight
            from multiprocessing import TimeoutError as MPTimeoutError
            
            TIMEOUT_PER_FLIGHT = 120  # 2 minutes max per flight
            
            with Pool(processes=args.workers, initializer=_init_worker) as pool:
                # Submit all jobs
                # Use imap_unordered for reactive progress bar
                # This allows updating the bar as soon as a flight is done, regardless of order
                results_iterator = pool.imap_unordered(process_and_save, tasks_with_checkpoint)
                
                for result in tqdm(results_iterator, total=len(tasks_with_checkpoint), desc="Flights"):
                    flight_id, n_seg, error = result
                    n_processed += 1
                    n_segments += n_seg
                    
                    if error:
                        errors.append(f"{flight_id}: {error}")
                
                if errors:
                    print(f"\n⚠️  Errors on {len(errors)} flights:")
                    for err in errors[:10]:
                        print(f"  - {err}")
                    if len(errors) > 10:
                        print(f"  ... and {len(errors) - 10} others")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  INTERRUPTED - {n_processed} flights processed")
        print(f"To resume: add --resume")
        print(f"To concatenate: add --concat-only")
        return
    
    print(f"\nFlights processed: {n_processed}")
    print(f"Segments calculated: {n_segments}")
    
    # --- Concatenate checkpoints ---
    print(f"\nConcatenating checkpoints...")
    features_df = concat_checkpoints(checkpoint_dir, args.output)
    
    if features_df is None:
        print("Error: no features to concatenate")
        return
    
    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"File: {args.output}")
    print(f"Segments: {len(features_df)}")
    print(f"Columns: {features_df.columns.tolist()}")
    print(f"Size: {os.path.getsize(args.output) / 1024**2:.1f} MB")
    
    if 'fuel_flow_mean' in features_df.columns:
        nan_ff = features_df['fuel_flow_mean'].isna().sum()
        print(f"Fuel flow NaN: {nan_ff} ({nan_ff/len(features_df)*100:.1f}%)")
    
    if 'fuel_estimated_kg' in features_df.columns:
        nan_est = features_df['fuel_estimated_kg'].isna().sum()
        print(f"Estimated fuel NaN: {nan_est} ({nan_est/len(features_df)*100:.1f}%)")
    
    if 'fuel_kg' in features_df.columns and 'fuel_estimated_kg' in features_df.columns:
        valid = features_df.dropna(subset=['fuel_kg', 'fuel_estimated_kg'])
        if len(valid) > 0:
            rmse = np.sqrt(((valid['fuel_kg'] - valid['fuel_estimated_kg'])**2).mean())
            mae = (valid['fuel_kg'] - valid['fuel_estimated_kg']).abs().mean()
            bias = (valid['fuel_estimated_kg'] - valid['fuel_kg']).mean()
            print(f"\nPhysical estimator performance:")
            print(f"  RMSE: {rmse:.1f} kg")
            print(f"  MAE: {mae:.1f} kg")
            print(f"  Bias: {bias:+.1f} kg")
    
    # --- Bias Correction ---
    if 'fuel_kg' in features_df.columns and 'fuel_estimated_kg' in features_df.columns:
        print("\nCalculating and applying bias correction...")
        
        # Calculate mean bias per aircraft type
        # Bias = Real - Estimated
        features_df['bias_raw'] = features_df['fuel_kg'] - features_df['fuel_estimated_kg']
        
        bias_per_type = features_df.groupby('typecode')['bias_raw'].mean().to_dict()
        
        print("Bias factors (kg):")
        for tc, bias in bias_per_type.items():
            print(f"  {tc}: {bias:+.1f}")
            
        # Apply correction
        # Estimated_Corrected = Estimated + Mean_Bias
        features_df['fuel_estimated_corrected'] = features_df.apply(
            lambda row: row['fuel_estimated_kg'] + bias_per_type.get(row['typecode'], 0),
            axis=1
        )
        
        # Save bias factors for future inference
        bias_path = Path(args.output).parent / "bias_factors.pkl"
        import pickle
        with open(bias_path, 'wb') as f:
            pickle.dump(bias_per_type, f)
        print(f"Bias factors saved: {bias_path}")
        
        # Recalculate metrics with correction
        valid = features_df.dropna(subset=['fuel_kg', 'fuel_estimated_corrected'])
        if len(valid) > 0:
            rmse = np.sqrt(((valid['fuel_kg'] - valid['fuel_estimated_corrected'])**2).mean())
            mae = (valid['fuel_kg'] - valid['fuel_estimated_corrected']).abs().mean()
            bias = (valid['fuel_estimated_corrected'] - valid['fuel_kg']).mean()
            print(f"\nPerformance AFTER correction:")
            print(f"  RMSE: {rmse:.1f} kg")
            print(f"  MAE: {mae:.1f} kg")
            print(f"  Bias: {bias:+.1f} kg")
            
        # Replace original column or keep both?
        # For now keep both for comparison, but for submission we'll need to choose
        features_df['fuel_estimated_kg_raw'] = features_df['fuel_estimated_kg']
        features_df['fuel_estimated_kg'] = features_df['fuel_estimated_corrected']
        
        # Cleanup
        features_df = features_df.drop(columns=['bias_raw', 'fuel_estimated_corrected'])
        
        # Save again with correction
        features_df.to_parquet(args.output, index=False)
        print(f"File updated with correction: {args.output}")

    print("\nPreview:")
    print(features_df.head(10))
    
    # Checkpoint info
    n_checkpoints = len(list(checkpoint_dir.glob("features_*.parquet")))
    print(f"\n💾 Checkpoints kept: {n_checkpoints} files in {checkpoint_dir}/")
    print("   (Delete this directory manually if you don't need it anymore)")


if __name__ == "__main__":
    main()