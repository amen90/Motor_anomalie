import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def generate_vibration_data(fault_type, num_samples=300, sample_rate=1000, duration=10):
    """
    Génère des données de vibration simulées pour différents types de défauts
    basées sur les spécifications du PDF du projet.
    """
    # Générer des timestamps pour 10 secondes à 1000 Hz
    time_points = np.linspace(0, duration, num_samples)
    
    data = {
        'timestamp': time_points,
        'X_axis_mg': np.zeros(num_samples),
        'Y_axis_mg': np.zeros(num_samples),
        'Z_axis_mg': np.zeros(num_samples),
        'fault_type': [fault_type] * num_samples,
        'sample_id': list(range(1, num_samples + 1))
    }

    if fault_type == 'Normal':
        # Normal Motor (Baseline)
        # X-axis: -200 to +200 mg, RMS: 10-50 mg
        # Y-axis: -200 to +200 mg, RMS: 10-50 mg  
        # Z-axis: -800 to +800 mg, RMS: 20-80 mg
        base_freq = 50  # Hz - fréquence de base du moteur
        
        # Générer des signaux sinusoïdaux avec du bruit
        data['X_axis_mg'] = 30 * np.sin(2 * np.pi * base_freq * time_points) + \
                           10 * np.random.normal(0, 1, num_samples)
        data['Y_axis_mg'] = 25 * np.sin(2 * np.pi * base_freq * time_points + np.pi/4) + \
                           8 * np.random.normal(0, 1, num_samples)
        data['Z_axis_mg'] = 50 * np.sin(2 * np.pi * base_freq * time_points) + \
                           1000 + 15 * np.random.normal(0, 1, num_samples)  # Inclut la gravité
        
    elif fault_type == 'Imbalance':
        # Imbalance Motor
        # X-axis: -500 to +500 mg, RMS: 50-200 mg
        # Y-axis: -500 to +500 mg, RMS: 50-200 mg
        # Z-axis: -1000 to +1000 mg, RMS: 80-300 mg
        base_freq = 50  # Hz
        
        # Forte composante à 1x la fréquence de rotation (déséquilibre)
        data['X_axis_mg'] = 150 * np.sin(2 * np.pi * base_freq * time_points) + \
                           50 * np.sin(2 * np.pi * 2 * base_freq * time_points) + \
                           30 * np.random.normal(0, 1, num_samples)
        data['Y_axis_mg'] = 120 * np.sin(2 * np.pi * base_freq * time_points + np.pi/2) + \
                           40 * np.sin(2 * np.pi * 2 * base_freq * time_points) + \
                           25 * np.random.normal(0, 1, num_samples)
        data['Z_axis_mg'] = 200 * np.sin(2 * np.pi * base_freq * time_points) + \
                           1000 + 40 * np.random.normal(0, 1, num_samples)
        
    elif fault_type == 'Bearing_Fault':
        # Bearing Fault
        # X-axis: -800 to +800 mg, RMS: 100-400 mg
        # Y-axis: -800 to +800 mg, RMS: 100-400 mg
        # Z-axis: -1200 to +1200 mg, RMS: 150-500 mg
        base_freq = 50  # Hz
        bearing_freq = 157  # Hz - fréquence caractéristique des roulements
        
        # Impacts haute fréquence caractéristiques des défauts de roulement
        data['X_axis_mg'] = 80 * np.sin(2 * np.pi * base_freq * time_points) + \
                           200 * np.sin(2 * np.pi * bearing_freq * time_points) + \
                           50 * np.random.normal(0, 1, num_samples)
        data['Y_axis_mg'] = 70 * np.sin(2 * np.pi * base_freq * time_points) + \
                           180 * np.sin(2 * np.pi * bearing_freq * time_points + np.pi/3) + \
                           45 * np.random.normal(0, 1, num_samples)
        data['Z_axis_mg'] = 100 * np.sin(2 * np.pi * base_freq * time_points) + \
                           250 * np.sin(2 * np.pi * bearing_freq * time_points) + \
                           1000 + 60 * np.random.normal(0, 1, num_samples)
        
    elif fault_type == 'Misalignment':
        # Misalignment
        # X-axis: -600 to +600 mg, RMS: 80-250 mg
        # Y-axis: -600 to +600 mg, RMS: 80-250 mg
        # Z-axis: -1000 to +1000 mg, RMS: 100-350 mg
        base_freq = 50  # Hz
        
        # Forte composante à 2x la fréquence de rotation (désalignement)
        data['X_axis_mg'] = 100 * np.sin(2 * np.pi * base_freq * time_points) + \
                           180 * np.sin(2 * np.pi * 2 * base_freq * time_points) + \
                           35 * np.random.normal(0, 1, num_samples)
        data['Y_axis_mg'] = 90 * np.sin(2 * np.pi * base_freq * time_points + np.pi/6) + \
                           160 * np.sin(2 * np.pi * 2 * base_freq * time_points + np.pi/4) + \
                           30 * np.random.normal(0, 1, num_samples)
        data['Z_axis_mg'] = 120 * np.sin(2 * np.pi * base_freq * time_points) + \
                           200 * np.sin(2 * np.pi * 2 * base_freq * time_points) + \
                           1000 + 45 * np.random.normal(0, 1, num_samples)
    
    return pd.DataFrame(data)

def generate_metadata(fault_type, sample_count):
    """Génère les métadonnées pour chaque échantillon"""
    metadata = {
        "dataset_info": {
            "total_samples": sample_count,
            "fault_type": fault_type,
            "sample_rate_hz": 1000,
            "duration_seconds": 10,
            "axes": ["X", "Y", "Z"],
            "units": "milligravity (mg)"
        },
        "motor_info": {
            "type": "induction",
            "power_rating": "1.5kW",
            "rpm": 1750,
            "manufacturer": "Generic"
        },
        "measurement_conditions": {
            "temperature_c": 25.5,
            "load_percent": 75,
            "sensor_position": "motor_housing_horizontal"
        },
        "fault_details": {
            "description": f"Simulated {fault_type.lower()} condition",
            "severity_level": 2
        },
        "generation_timestamp": datetime.now().isoformat()
    }
    return metadata

# Créer le répertoire de sortie
output_dir = 'example_data'
os.makedirs(output_dir, exist_ok=True)

# Types de défauts à générer
fault_types = ['Normal', 'Imbalance', 'Bearing_Fault', 'Misalignment']

print("Génération des données d'exemple...")

for fault_type in fault_types:
    print(f"Génération des données pour: {fault_type}")
    
    # Générer les données
    vibration_data = generate_vibration_data(fault_type, num_samples=300)
    
    # Sauvegarder en CSV
    csv_filename = f"{fault_type.lower()}_vibration_data.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    vibration_data.to_csv(csv_path, index=False)
    
    # Générer et sauvegarder les métadonnées
    metadata = generate_metadata(fault_type, 300)
    metadata_filename = f"{fault_type.lower()}_metadata.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  - Données sauvegardées: {csv_path}")
    print(f"  - Métadonnées sauvegardées: {metadata_path}")

# Créer un fichier de résumé
summary = {
    "dataset_summary": {
        "total_fault_types": len(fault_types),
        "samples_per_type": 300,
        "total_samples": len(fault_types) * 300,
        "fault_types": fault_types,
        "data_format": "CSV with JSON metadata",
        "generation_date": datetime.now().isoformat()
    }
}

summary_path = os.path.join(output_dir, "dataset_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nGénération terminée!")
print(f"Répertoire de sortie: {output_dir}/")
print(f"Résumé du dataset: {summary_path}")
print(f"Total des fichiers générés: {len(fault_types) * 2 + 1}")