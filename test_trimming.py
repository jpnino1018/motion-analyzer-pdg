"""
Quick test to visualize the effect of inactive period trimming
"""
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.cleaners import recortar_inactividad

def test_trimming_effect(json_file_path):
    """Load a JSON file and show before/after trimming"""
    
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Simple normalization - handle both formats
    if "imuData" in data:
        left_data_original = [d for d in data["imuData"] if "LEFT" in d.get("deviceId", "").upper()]
    elif "izquierda" in data:
        ACC_SCALE = 16384.0
        left_data_original = []
        for d in data["izquierda"]:
            left_data_original.append({
                "timestamp": d["millis"],
                "accelerometer": {
                    "x": (d["x"] / ACC_SCALE) * 9.81,
                    "y": (d["y"] / ACC_SCALE) * 9.81,
                    "z": (d["z"] / ACC_SCALE) * 9.81,
                }
            })
    else:
        print("Unknown data format")
        return
    left_data_trimmed = recortar_inactividad(left_data_original, umbral=0.5, min_len=50)
    
    # Calculate magnitudes
    def get_magnitudes(data):
        mags = []
        for d in data:
            acc = d.get("accelerometer", {})
            if all(k in acc for k in ("x", "y", "z")):
                mag = (acc["x"]**2 + acc["y"]**2 + acc["z"]**2) ** 0.5
                mags.append(mag)
        return np.array(mags)
    
    mag_original = get_magnitudes(left_data_original)
    mag_trimmed = get_magnitudes(left_data_trimmed)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    
    # Original data
    time_original = np.arange(len(mag_original)) / 100  # Assuming ~100Hz
    ax1.plot(time_original, mag_original, 'b-', alpha=0.7)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5 m/s²)')
    ax1.fill_between(time_original[:len(mag_original)-len(mag_trimmed)], 
                      0, mag_original[:len(mag_original)-len(mag_trimmed)].max(),
                      color='red', alpha=0.2, label='Trimmed region')
    ax1.set_title(f'BEFORE Trimming - {len(mag_original)} samples')
    ax1.set_ylabel('Magnitude (m/s²)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Trimmed data
    time_trimmed = np.arange(len(mag_trimmed)) / 100
    ax2.plot(time_trimmed, mag_trimmed, 'g-', alpha=0.7)
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5 m/s²)')
    ax2.set_title(f'AFTER Trimming - {len(mag_trimmed)} samples (removed {len(mag_original)-len(mag_trimmed)})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Magnitude (m/s²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print stats
    print(f"\n{'='*60}")
    print(f"TRIMMING EFFECT TEST")
    print(f"{'='*60}")
    print(f"Original samples: {len(mag_original)}")
    print(f"Trimmed samples:  {len(mag_trimmed)}")
    print(f"Removed samples:  {len(mag_original)-len(mag_trimmed)} ({100*(1-len(mag_trimmed)/len(mag_original)):.1f}%)")
    print(f"\nOriginal magnitude range: {mag_original.min():.3f} to {mag_original.max():.3f} m/s²")
    print(f"Trimmed magnitude range:  {mag_trimmed.min():.3f} to {mag_trimmed.max():.3f} m/s²")
    print(f"\nOriginal mean: {mag_original.mean():.3f} m/s²")
    print(f"Trimmed mean:  {mag_trimmed.mean():.3f} m/s²")
    print(f"{'='*60}\n")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_trimming.py <path_to_json_file>")
        print("\nExample:")
        print("  python test_trimming.py data/diagnosticados/stomp/20250813201345_25276869-2025-08-13T151235-data.json")
        sys.exit(1)
    
    test_trimming_effect(sys.argv[1])
