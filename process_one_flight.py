import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Import existing modules
from clean_trajectories import clean_single_trajectory
from feature_engineering import process_single_flight, ACROPOLE_TYPES, OPENAP_TYPES

def main():
    flight_id = 'prc770822360'
    raw_path = f'prc_data/flights_train/{flight_id}.parquet'
    clean_path = f'data/temp_{flight_id}.parquet'
    
    print(f"Processing flight {flight_id}...")
    
    # 1. Clean
    print("1. Cleaning trajectory...")
    stats = clean_single_trajectory((raw_path, clean_path))
    print(f"   Cleaned: {stats['n_points_raw']} -> {stats['n_points_clean']} points")
    
    # 2. Prepare args for feature engineering
    print("2. Preparing feature engineering...")
    
    # Load metadata
    flightlist = pd.read_parquet('prc_data/flightlist_train.parquet')
    fuel_df = pd.read_parquet('prc_data/fuel_train.parquet')
    
    flight_info = flightlist[flightlist['flight_id'] == flight_id].iloc[0].to_dict()
    segments_df = fuel_df[fuel_df['flight_id'] == flight_id].copy()
    typecode = flight_info['aircraft_type']
    
    print(f"   Type: {typecode}")
    print(f"   Segments: {len(segments_df)}")
    
    # 3. Run feature engineering
    print("3. Running feature engineering...")
    args = (flight_id, typecode, clean_path, segments_df, flight_info)
    features_list = process_single_flight(args)
    
    if not features_list:
        print("❌ No features generated!")
        return
        
    df_features = pd.DataFrame(features_list)
    
    # 4. Display results
    print("\n=== RESULTS ===")
    cols_to_show = ['idx', 'duration_sec', 'fuel_flow_mean', 'mass_estimated_kg', 'mass_source', 'fuel_estimated_kg', 'fuel_kg']
    
    print(df_features[cols_to_show])
    
    # Check mass source
    print(f"\nMass Source: {df_features['mass_source'].iloc[0]}")
    print(f"Estimated Mass: {df_features['mass_estimated_kg'].iloc[0]} kg")
    
    # Cleanup
    if os.path.exists(clean_path):
        os.remove(clean_path)

if __name__ == "__main__":
    main()
