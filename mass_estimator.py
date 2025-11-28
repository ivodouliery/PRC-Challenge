import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import minimize_scalar
import warnings

# Suppress OpenAP warnings
warnings.filterwarnings('ignore')

try:
    from openap import FuelFlow, prop
except ImportError:
    print("OpenAP not installed")
    exit(1)

def get_openap_model(typecode):
    try:
        return FuelFlow(typecode, use_synonym=True), prop.aircraft(typecode)
    except:
        return None, None

def solve_mass_for_flight(flight_id, df, fuel_actual_kg):
    """
    Finds the initial mass that minimizes the error between calculated and actual fuel.
    """
    if len(df) < 2:
        return None
        
    typecode = df['typecode'].iloc[0]
    ff_model, ac_props = get_openap_model(typecode)
    
    if ff_model is None:
        return None
        
    mtow = ac_props.get('mtow', 75000)
    oew = ac_props.get('oew', mtow * 0.5)
    
    # Prepare vectorized data
    tas = df['groundspeed'].values # Approximation if TAS not available
    if 'airspeed' in df.columns and df['airspeed'].notna().any():
        # Use airspeed where available
        mask = df['airspeed'].notna()
        tas[mask] = df.loc[mask, 'airspeed'].values
        
    alt = df['altitude'].values
    vs = df['vertical_rate'].values
    dt = df['timestamp'].diff().dt.total_seconds().fillna(0).values
    dt[0] = dt[1] if len(dt) > 1 else 1
    
    def calculate_fuel(mass0):
        # Vectorized simulation
        # Note: To be precise, we should recalculate mass at each step
        # But for optimization, we can use an approximation or a fast loop
        
        # Iterative vectorized approach (fast)
        current_mass = mass0
        total_fuel = 0
        
        # We do a single pass with estimated average mass for speed
        # Or better: use OpenAP's enroute function which is vectorized
        
        # For optimization, we keep it simple:
        # Assume mass decreases linearly or do simple integration
        
        # 1. Calculate FF with constant mass (mass0)
        ff = ff_model.enroute(mass=mass0, tas=tas, alt=alt, vs=vs).flatten()
        fuel_consumed = np.sum(ff * dt)
        
        # 2. Refinement: average mass
        mass_avg = mass0 - fuel_consumed / 2
        ff = ff_model.enroute(mass=mass_avg, tas=tas, alt=alt, vs=vs).flatten()
        fuel_final = np.sum(ff * dt)
        
        return fuel_final

    # Objective function
    def objective(mass0):
        fuel_est = calculate_fuel(mass0)
        return abs(fuel_est - fuel_actual_kg)
    
    # Bounded scalar optimization [OEW, MTOW]
    res = minimize_scalar(objective, bounds=(oew, mtow), method='bounded')
    
    if res.success:
        return res.x
    return None

def train_mass_model(fuel_path, flightlist_path, trajectories_dir, output_model_path):
    print(f"Training mass model from {fuel_path}...")
    
    fuel_df = pd.read_parquet(fuel_path)
    flightlist = pd.read_parquet(flightlist_path)
    
    # Map flight_id -> typecode, duration
    flight_meta = flightlist.set_index('flight_id')[['aircraft_type']].to_dict('index')
    
    # Group fuel by flight (sum of segments)
    flight_fuel = fuel_df.groupby('flight_id')['fuel_kg'].sum()
    
    results = []
    
    # Select a sample for training (e.g., 1000 flights or all)
    # To have good coverage (especially B789), we take all or a very large number
    sample_flights = flight_fuel.sample(n=min(12000, len(flight_fuel))).index
    
    print(f"Processing {len(sample_flights)} flights for mass inversion...")
    
    for flight_id in tqdm(sample_flights):
        if flight_id not in flight_meta:
            continue
            
        typecode = flight_meta[flight_id]['aircraft_type']
        fuel_kg = flight_fuel[flight_id]
        
        # Load trajectory
        traj_path = Path(trajectories_dir) / f"{flight_id}.parquet"
        if not traj_path.exists():
            continue
            
        try:
            df = pd.read_parquet(traj_path)
            # Minimal cleaning if necessary (normally already clean)
            if 'groundspeed' not in df.columns: 
                continue
                
            df['typecode'] = typecode
            df = df.sort_values('timestamp')
            
            # Solve mass
            estimated_mass = solve_mass_for_flight(flight_id, df, fuel_kg)
            
            if estimated_mass:
                # Calculate distance and duration
                dist = 0
                if 'latitude' in df.columns:
                    # Approx distance
                    pass # No need to be precise for now
                
                duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
                
                results.append({
                    'typecode': typecode,
                    'duration': duration,
                    'fuel_total': fuel_kg,
                    'estimated_mass': estimated_mass
                })
                
        except Exception as e:
            pass

    # Create model (Lookup table by type + linear regression on duration)
    print(f"Training model on {len(results)} samples...")
    train_df = pd.DataFrame(results)
    
    model = {}
    
    for typecode, group in train_df.groupby('typecode'):
        if len(group) < 5:
            # Not enough data, use simple mean or MTOW ratio
            model[typecode] = {
                'method': 'mean',
                'mass_mean': group['estimated_mass'].mean()
            }
        else:
            # Simple linear regression: Mass = a * Duration + b
            # Or better: Mass = a * Duration + b (because the further we fly, the heavier the fuel load)
            coeffs = np.polyfit(group['duration'], group['estimated_mass'], 1)
            model[typecode] = {
                'method': 'linear',
                'slope': coeffs[0],
                'intercept': coeffs[1]
            }
            
    # Save
    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    # Paths (adapt according to environment)
    train_mass_model(
        fuel_path='prc_data/fuel_train.parquet',
        flightlist_path='prc_data/flightlist_train.parquet',
        trajectories_dir='data/clean_flights_train', # Use real clean files
        output_model_path='data/mass_model.pkl'
    )
