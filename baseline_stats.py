import os
import json
import numpy as np
from typing import Dict, List

# Use the new processing pipeline
from src.preprocessing.movement_processor import MovementProcessor
from src.preprocessing.cleaners import recortar_inactividad

CONTROLS_DIR = "./data/sanos"
OUTPUT_FILE = "./baselines/population_baseline.json"


def load_and_normalize(path: str) -> Dict[str, List[dict]]:
    """Load a JSON file and normalize into {'LEFT': [...], 'RIGHT': [...]}.

    Supports legacy `imuData` wrapper or `izquierda`/`derecha` raw arrays.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = {"LEFT": [], "RIGHT": []}

    # legacy imuData format
    if isinstance(data, dict) and "imuData" in data:
        for d in data["imuData"]:
            device = str(d.get("deviceId", "")).upper()
            side = "LEFT" if "LEFT" in device else "RIGHT"
            normalized[side].append({
                "timestamp": d.get("timestamp"),
                "accelerometer": d.get("accelerometer", {}),
                "gyroscope": d.get("gyroscope", {})
            })
        return normalized

    # raw format with 'izquierda'/'derecha'
    if isinstance(data, dict) and ("izquierda" in data or "derecha" in data):
        ACC_SCALE = 16384.0
        GYRO_SCALE = 131.0

        if "izquierda" in data:
            for d in data["izquierda"]:
                normalized["LEFT"].append({
                    "timestamp": d.get("millis"),
                    "accelerometer": {
                        "x": (d.get("x", 0) / ACC_SCALE) * 9.81,
                        "y": (d.get("y", 0) / ACC_SCALE) * 9.81,
                        "z": (d.get("z", 0) / ACC_SCALE) * 9.81,
                    },
                    "gyroscope": {
                        "x": d.get("a", 0) / GYRO_SCALE,
                        "y": d.get("b", 0) / GYRO_SCALE,
                        "z": d.get("g", 0) / GYRO_SCALE,
                    }
                })

        if "derecha" in data:
            for d in data["derecha"]:
                normalized["RIGHT"].append({
                    "timestamp": d.get("millis"),
                    "accelerometer": {
                        "x": (d.get("x", 0) / ACC_SCALE) * 9.81,
                        "y": (d.get("y", 0) / ACC_SCALE) * 9.81,
                        "z": (d.get("z", 0) / ACC_SCALE) * 9.81,
                    },
                    "gyroscope": {
                        "x": d.get("a", 0) / GYRO_SCALE,
                        "y": d.get("b", 0) / GYRO_SCALE,
                        "z": d.get("g", 0) / GYRO_SCALE,
                    }
                })

        return normalized

    raise ValueError(f"Unrecognized JSON format for file: {path}")


def collect_features(controls_dir: str = CONTROLS_DIR) -> Dict[str, List[dict]]:
    """Walk `controls_dir`, process JSON files and return features grouped by exercise.

    For each file we extract left/right MovementMetrics and append both sides to the
    exercise list (so baseline is side-agnostic population data).
    """
    processor = MovementProcessor()
    all_features: Dict[str, List[dict]] = {"tapping": [], "stomp": []}

    if not os.path.exists(controls_dir):
        raise FileNotFoundError(f"Controls directory not found: {controls_dir}")

    for root, _, files in os.walk(controls_dir):
        for file in files:
            if not file.lower().endswith('.json'):
                continue

            path = os.path.join(root, file)
            lower = (root + file).lower()
            # try to infer exercise from path/filename
            if 'tapping' in lower:
                ejercicio = 'tapping'
            elif 'stomp' in lower or 'stom' in lower or 'zapate' in lower:
                ejercicio = 'stomp'
            else:
                # fallback: use filename hints
                ejercicio = 'stomp' if 'stomp' in file.lower() else 'tapping'

            try:
                normalized = load_and_normalize(path)
            except Exception as e:
                print(f"Skipping {file}: could not load/normalize ({e})")
                continue

            # Trim inactivity for each side
            left = recortar_inactividad(normalized.get('LEFT', [])) if normalized.get('LEFT') else []
            right = recortar_inactividad(normalized.get('RIGHT', [])) if normalized.get('RIGHT') else []

            # Process both sides and append metrics if valid
            try:
                if left:
                    lm = processor.process_movement_data(left)
                    all_features[ejercicio].append(vars(lm))
                if right:
                    rm = processor.process_movement_data(right)
                    all_features[ejercicio].append(vars(rm))
                print(f"Processed: {path} -> {ejercicio} (L:{len(left)} R:{len(right)})")
            except Exception as e:
                print(f"Error processing {path}: {e}")

    return all_features


def stats_from_list(feature_list: List[dict]) -> dict:
    """Compute mean/std and percentiles for each numeric feature in feature_list."""
    if not feature_list:
        return {}

    stats = {}
    keys = list(feature_list[0].keys())

    for key in keys:
        # gather numeric values, ignore None
        values = np.array([f.get(key, np.nan) for f in feature_list], dtype=float)
        # filter NaNs
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue

        stats[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'p10': float(np.percentile(values, 10)),
            'p25': float(np.percentile(values, 25)),
            'p50': float(np.percentile(values, 50)),
            'p75': float(np.percentile(values, 75)),
            'p90': float(np.percentile(values, 90))
        }

    return stats


def build_baseline(controls_dir: str = CONTROLS_DIR, output_file: str = OUTPUT_FILE):
    all_feats = collect_features(controls_dir)

    baseline = {}
    for exercise, feats in all_feats.items():
        baseline[exercise] = stats_from_list(feats)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=4, ensure_ascii=False)

    print('=========================================')
    print('  ✅ Population baseline generated')
    print(f'  File: {output_file}')
    print('=========================================')


if __name__ == '__main__':
    build_baseline()
