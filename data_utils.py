import pandas as pd
import numpy as np
import os
import sys

def load_and_preprocess_data(path, is_train=True, subsample_ratio=1.0):
    """
    Loads and cleans data for training and inference.
    """
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
        
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    
    # --- 1. HANDLE EMPTY SEGMENTS (GHOSTS) ---
    # Calculate theoretical duration from timestamps
    # Ensure start/end are datetimes
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    
    theoretical_duration = (df['end'] - df['start']).dt.total_seconds()
    
    # If duration_sec is NaN or 0, use theoretical
    # Add feature to flag "Warning, was empty"
    df['is_missing_data'] = (df['n_points'] == 0) | (df['duration_sec'].isna()) | (df['duration_sec'] < 1)
    df['is_missing_data'] = df['is_missing_data'].astype(int)
    
    # Replace null/missing durations
    mask_bad_duration = (df['duration_sec'].isna()) | (df['duration_sec'] < 0.1)
    df.loc[mask_bad_duration, 'duration_sec'] = theoretical_duration[mask_bad_duration]

    # --- 1.5. ADD TEMPORAL FEATURES (CONTEXT) ---
    print("⏳ Adding temporal context (Lag/Lead/Cumul)...")
    
    # Ensure sorted
    df = df.sort_values(['flight_id', 'idx'])
    
    # Groupby for vectorized operations
    g = df.groupby('flight_id')
    
    # 1. Cumulative (Since start of flight)
    df['time_since_takeoff'] = g['duration_sec'].cumsum() - df['duration_sec'] # Time elapsed BEFORE this segment
    if 'distance_km' in df.columns:
        df['dist_flown'] = g['distance_km'].cumsum() - df['distance_km']
    
    # 2. Lag Features (What happened just before)
    # Shift by 1 downwards
    cols_to_lag = ['alt_mean', 'gs_mean', 'vrate_mean', 'phase']
    for col in cols_to_lag:
        if col in df.columns:
            df[f'prev_{col}'] = g[col].shift(1)
            
    # 3. Lead Features (What will happen just after - Anticipation)
    # Shift by -1 (upwards)
    if 'phase' in df.columns:
        df['next_phase'] = g['phase'].shift(-1)
        
    # Fill NaNs created by shift (start/end of flight)
    # For numeric, set -1 or 0
    num_lag_cols = [c for c in df.columns if c.startswith('prev_') and df[c].dtype.kind in 'ifc']
    df[num_lag_cols] = df[num_lag_cols].fillna(0)
    
    # For categorical, set 'NA'
    cat_lag_cols = ['prev_phase', 'next_phase']
    for c in cat_lag_cols:
        if c in df.columns:
            df[c] = df[c].fillna('NA').astype(str)

    # --- 2. OUTLIER FILTERING (TRAIN ONLY) ---
    if is_train:
        # Calculate consumption per second (kg/s)
        # A380 consumes max ~4 kg/s (14t/h). Take a wide margin (20 kg/s)
        # To avoid division by 0, clip duration
        safe_duration = df['duration_sec'].clip(lower=1.0)
        consumption_rate = df['fuel_kg'] / safe_duration
        
        # Physical threshold: 20 kg/s (72 tonnes/hour) -> Impossible even for A380 in climb
        # Only filter if fuel_kg > 10 (to avoid removing small noisy segments)
        mask_outlier = (consumption_rate > 20) & (df['fuel_kg'] > 10)
        n_outliers = mask_outlier.sum()
        
        if n_outliers > 0:
            print(f"🧹 CLEANING: Removing {n_outliers} physical outliers (>20 kg/s)")
            df = df[~mask_outlier].reset_index(drop=True)

    # --- 3. SUBSAMPLING (OPTIONAL) ---
    if is_train and subsample_ratio < 1.0:
        flight_ids = df['flight_id'].unique()
        n_keep = int(len(flight_ids) * subsample_ratio)
        selected_flights = np.random.choice(flight_ids, size=n_keep, replace=False)
        df = df[df['flight_id'].isin(selected_flights)].copy()
        print(f"⚠️ SUBSAMPLING: Dataset reduced to {subsample_ratio*100}% ({len(df)} rows)")

    # --- 4. CATEGORY HANDLING ---
    cat_cols_names = ['aircraft_type', 'fuel_source', 'mass_source', 'origin_icao', 'destination_icao', 'phase', 'typecode', 'prev_phase', 'next_phase']
    for c in cat_cols_names:
        if c in df.columns:
            df[c] = df[c].fillna('UNKNOWN').astype(str).astype('category')
            
    # --- 5. FEATURE SELECTION ---
    cols_exclude = ['flight_id', 'start', 'end', 'fuel_kg', 'flight_date', 'takeoff', 'landed', 'origin_name', 'destination_name', 'timestamp_start', 'timestamp_end', 'idx']
    features = [c for c in df.columns if c not in cols_exclude]
    
    X = df[features]
    y = df['fuel_kg'] if 'fuel_kg' in df.columns else None
    groups = df['flight_id']
    
    cat_cols = [c for c in X.columns if X[c].dtype.name == 'category']
    
    return X, y, groups, cat_cols, df
