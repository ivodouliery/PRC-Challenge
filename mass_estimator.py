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
    Trouve la masse initiale qui minimise l'erreur entre fuel calculé et fuel réel.
    """
    if len(df) < 2:
        return None
        
    typecode = df['typecode'].iloc[0]
    ff_model, ac_props = get_openap_model(typecode)
    
    if ff_model is None:
        return None
        
    mtow = ac_props.get('mtow', 75000)
    oew = ac_props.get('oew', mtow * 0.5)
    
    # Préparer les données vectorisées
    tas = df['groundspeed'].values # Approximation si TAS pas dispo
    if 'airspeed' in df.columns and df['airspeed'].notna().any():
        # Utiliser airspeed là où dispo
        mask = df['airspeed'].notna()
        tas[mask] = df.loc[mask, 'airspeed'].values
        
    alt = df['altitude'].values
    vs = df['vertical_rate'].values
    dt = df['timestamp'].diff().dt.total_seconds().fillna(0).values
    dt[0] = dt[1] if len(dt) > 1 else 1
    
    def calculate_fuel(mass0):
        # Simulation vectorisée
        # Note: Pour être précis, il faudrait recalculer la masse à chaque pas
        # Mais pour l'optimisation, on peut faire une approximation ou une boucle rapide
        
        # Approche vectorisée itérative (rapide)
        current_mass = mass0
        total_fuel = 0
        
        # On fait une seule passe avec la masse moyenne estimée pour aller vite
        # Ou mieux: on utilise la fonction enroute d'OpenAP qui est vectorisée
        
        # Pour l'optimisation, on va faire simple:
        # On suppose que la masse diminue linéairement ou on fait une intégration simple
        
        # 1. Calculer FF avec masse constante (mass0)
        ff = ff_model.enroute(mass=mass0, tas=tas, alt=alt, vs=vs).flatten()
        fuel_consumed = np.sum(ff * dt)
        
        # 2. Raffinement: masse moyenne
        mass_avg = mass0 - fuel_consumed / 2
        ff = ff_model.enroute(mass=mass_avg, tas=tas, alt=alt, vs=vs).flatten()
        fuel_final = np.sum(ff * dt)
        
        return fuel_final

    # Fonction objective
    def objective(mass0):
        fuel_est = calculate_fuel(mass0)
        return abs(fuel_est - fuel_actual_kg)
    
    # Optimisation scalaire bornée [OEW, MTOW]
    res = minimize_scalar(objective, bounds=(oew, mtow), method='bounded')
    
    if res.success:
        return res.x
    return None

def train_mass_model(fuel_path, flightlist_path, trajectories_dir, output_model_path):
    print(f"Training mass model from {fuel_path}...")
    
    fuel_df = pd.read_parquet(fuel_path)
    flightlist = pd.read_parquet(flightlist_path)
    
    # Mapper flight_id -> typecode, duration
    flight_meta = flightlist.set_index('flight_id')[['aircraft_type']].to_dict('index')
    
    # Grouper fuel par vol (somme des segments)
    flight_fuel = fuel_df.groupby('flight_id')['fuel_kg'].sum()
    
    results = []
    
    # Sélectionner un échantillon pour l'entraînement (ex: 1000 vols ou tous)
    # Pour avoir une bonne couverture (notamment B789), on prend tout ou un très grand nombre
    sample_flights = flight_fuel.sample(n=min(12000, len(flight_fuel))).index
    
    print(f"Processing {len(sample_flights)} flights for mass inversion...")
    
    for flight_id in tqdm(sample_flights):
        if flight_id not in flight_meta:
            continue
            
        typecode = flight_meta[flight_id]['aircraft_type']
        fuel_kg = flight_fuel[flight_id]
        
        # Charger trajectoire
        traj_path = Path(trajectories_dir) / f"{flight_id}.parquet"
        if not traj_path.exists():
            continue
            
        try:
            df = pd.read_parquet(traj_path)
            # Nettoyage minimal si nécessaire (normalement déjà clean)
            if 'groundspeed' not in df.columns: 
                continue
                
            df['typecode'] = typecode
            df = df.sort_values('timestamp')
            
            # Résoudre masse
            estimated_mass = solve_mass_for_flight(flight_id, df, fuel_kg)
            
            if estimated_mass:
                # Calculer distance et durée
                dist = 0
                if 'latitude' in df.columns:
                    # Approx distance
                    pass # Pas besoin d'être précis pour l'instant
                
                duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
                
                results.append({
                    'typecode': typecode,
                    'duration': duration,
                    'fuel_total': fuel_kg,
                    'estimated_mass': estimated_mass
                })
                
        except Exception as e:
            pass

    # Créer le modèle (Lookup table par type + régression linéaire sur durée)
    print(f"Training model on {len(results)} samples...")
    train_df = pd.DataFrame(results)
    
    model = {}
    
    for typecode, group in train_df.groupby('typecode'):
        if len(group) < 5:
            # Pas assez de données, utiliser moyenne simple ou ratio MTOW
            model[typecode] = {
                'method': 'mean',
                'mass_mean': group['estimated_mass'].mean()
            }
        else:
            # Régression linéaire simple: Mass = a * Duration + b
            # Ou mieux: Mass = a * Duration + b (car plus on vole loin, plus on est lourd en fuel)
            coeffs = np.polyfit(group['duration'], group['estimated_mass'], 1)
            model[typecode] = {
                'method': 'linear',
                'slope': coeffs[0],
                'intercept': coeffs[1]
            }
            
    # Sauvegarder
    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    # Chemins (à adapter selon l'environnement)
    train_mass_model(
        fuel_path='prc_data/fuel_train.parquet',
        flightlist_path='prc_data/flightlist_train.parquet',
        trajectories_dir='data/clean_flights_train', # Utiliser les vrais fichiers clean
        output_model_path='data/mass_model.pkl'
    )
