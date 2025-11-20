import os
import csv
import json
from typing import Dict, List, Any
import numpy as np
from src.preprocessing.movement_processor import MovementProcessor
from src.preprocessing.signal_processing import AccelerometerData
from src.visualization.movement_visualizer import MovementVisualizer

def load_json_data(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load and normalize JSON data from file"""
    with open(file_path, "r") as f:
        data = json.load(f)

    normalized = {"LEFT": [], "RIGHT": []}

    if isinstance(data, dict):
        if "imuData" in data:
            # Old format with imuData wrapper
            for d in data["imuData"]:
                side = "LEFT" if "LEFT" in d["deviceId"].upper() else "RIGHT"
                normalized[side].append({
                    "timestamp": d.get("timestamp"),
                    "accelerometer": d.get("accelerometer", {}),
                    "gyroscope": d.get("gyroscope", {})
                })
        elif "izquierda" in data or "derecha" in data:
            # New raw format
            ACC_SCALE = 16384.0
            GYRO_SCALE = 131.0

            for side, raw_key in [("LEFT", "izquierda"), ("RIGHT", "derecha")]:
                if raw_key in data:
                    for d in data[raw_key]:
                        normalized[side].append({
                            "timestamp": d["millis"],
                            "accelerometer": {
                                "x": (d["x"] / ACC_SCALE) * 9.81,
                                "y": (d["y"] / ACC_SCALE) * 9.81,
                                "z": (d["z"] / ACC_SCALE) * 9.81,
                            },
                            "gyroscope": {
                                "x": d["a"] / GYRO_SCALE,
                                "y": d["b"] / GYRO_SCALE,
                                "z": d["g"] / GYRO_SCALE,
                            }
                        })
    return normalized

def process_file(file_path: str, exercise_type: str, visualizer: MovementVisualizer) -> dict:
    """Process a single movement file and generate visualizations"""
    processor = MovementProcessor()
    data = load_json_data(file_path)
    
    # Process both sides (with automatic inactive period trimming enabled by default)
    left_metrics = processor.process_movement_data(data["LEFT"], trim_inactive=True)
    right_metrics = processor.process_movement_data(data["RIGHT"], trim_inactive=True)
    
    # Determine active side
    active_side = "LEFT" if left_metrics.magnitude_mean > right_metrics.magnitude_mean else "RIGHT"
    passive_side = "RIGHT" if active_side == "LEFT" else "LEFT"
    
    # Generate visualizations
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    viz_path = os.path.join("graficas", exercise_type, f"{base_name}.png")
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    
    # Extract AccelerometerData for visualization
    left_acc_data = processor._extract_accelerometer_data(data["LEFT"], 
                                                        np.array([d["timestamp"] for d in data["LEFT"]]))
    right_acc_data = processor._extract_accelerometer_data(data["RIGHT"], 
                                                         np.array([d["timestamp"] for d in data["RIGHT"]]))
    
    # Get peaks for visualization
    left_peaks = processor.signal_processor.detect_peaks(left_acc_data.magnitude)
    right_peaks = processor.signal_processor.detect_peaks(right_acc_data.magnitude)
    
    # Generate comparison plot
    visualizer.plot_movement_comparison(
        left_acc_data, right_acc_data,
        left_peaks, right_peaks,
        f"{exercise_type.title()} - {base_name}",
        viz_path
    )
    
    # Prepare metrics for output
    active_metrics = left_metrics if active_side == "LEFT" else right_metrics
    passive_metrics = right_metrics if active_side == "LEFT" else left_metrics
    
    return {
        'archivo': os.path.basename(file_path),
        'ejercicio': exercise_type,
        'lado_activo': active_side,
        **{f'activo_{k}': v for k, v in active_metrics.__dict__.items()},
        **{f'pasivo_{k}': v for k, v in passive_metrics.__dict__.items()}
    }

def process_all_files():
    """Process all movement files in the data directory"""
    base_dir = './data'
    categories = ['diagnosticados', 'sanos']
    exercises = ['stomp', 'tapping']
    results = []
    visualizer = MovementVisualizer()
    
    for category in categories:
        for exercise in exercises:
            exercise_dir = os.path.join(base_dir, category, exercise)
            if os.path.exists(exercise_dir):
                for file_name in os.listdir(exercise_dir):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(exercise_dir, file_name)
                        try:
                            result = process_file(file_path, exercise, visualizer)
                            result['categoria'] = category
                            results.append(result)
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
    
    return results

def save_results(results: List[dict], output_path: str = 'resultados/resultados.csv'):
    """Save processing results to CSV file"""
    if not results:
        print("⚠️ No hay resultados para guardar.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fields = list(results[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Resultados guardados en: {output_path}")

if __name__ == '__main__':
    results = process_all_files()
    save_results(results)
