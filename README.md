# Motion Data Visualizer and Analyzer

This project is designed to process, analyze, and visualize motion data collected from wearable IMU sensors placed on ankles and feet. The primary goal is to assist in diagnostics and physical assessments by identifying asymmetries and motion patterns in the exercises extracted from MDS-UPRS III.

---

## Features Implemented So Far

# Motion Data Visualizer and Analyzer

This repository processes, analyzes and visualizes IMU motion data (accelerometer + gyroscope) captured from limb-mounted devices. The main aims are:

- Normalize different JSON formats produced by devices
- Trim inactive periods
- Detect movement repetitions and extract features (magnitude, rhythm, fatigue, etc.)
- Visualize movement traces and detected peaks
- Export per-file metrics to CSV for downstream analysis

## Quick start

Install dependencies (preferably in a virtual environment):

```powershell
python -m venv .venv ; .\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit UI (upload a JSON in the app to process & visualize):

```powershell
streamlit run app_improved.py
```

Or process all files programmatically and write a CSV:

```powershell
python main.py
```

Results and plots are written to:
- `resultados/resultados.csv` (summary metrics)
- `graficas/<exercise>/<file>.png` (comparison images)

---

## Preprocessing

Files come in two main JSON shapes; the code normalizes both into a standard structure: `{"LEFT": [...], "RIGHT": [...]}`.

- Supported formats:
  - Legacy format with `imuData`: each entry has `deviceId`, `timestamp`, `accelerometer`, `gyroscope`.
  - Raw sensor format with `izquierda`/`derecha` arrays where counts are converted to physical units.

- Normalization details (`preprocessing/normalization.py`):
  - Converts raw accelerometer counts into m/s² using `ACC_SCALE = 16384.0` (g-counts) and multiplies by 9.81.
  - Converts raw gyro counts using `GYRO_SCALE = 131.0` (°/s).
  - Returns a dict with `LEFT` and `RIGHT` lists; each element: `{timestamp, accelerometer: {x,y,z}, gyroscope: {x,y,z}}`.

- Cleaning inactive segments (`preprocessing/cleaners.py`):
  - `recortar_inactividad(datos, umbral=0.5, min_len=50)` trims the leading samples until a run of `min_len` samples above `umbral` magnitude is found.
  - Important to remove pre/post recording noise and long idle periods.

---

## Movement analysis (feature extraction)

Core responsibilities live in `src/preprocessing/movement_processor.py` and `src/analysis/movement_analysis.py`:

- Signal extraction:
  - Magnitude per sample is computed as |a| = sqrt(x²+y²+z²).
  - Data points are sorted by timestamp and timestamps are normalized to seconds relative to the start of each side.

- Peak detection (`src/preprocessing/signal_processing.py`):
  - Uses `scipy.signal.find_peaks` with configurable filters: `height`, `distance`, `prominence`, `width`.
  - We added a dynamic prominence heuristic so only large, meaningful peaks are selected by default: `prominence = max(0.2, 0.25*(max_mag - min_mag))`.
  - This reduces false positives from small fluctuations while preserving true movement peaks.

- Extracted metrics (`MovementMetrics`):
  - `n_reps`: number of detected repetitions (peaks)
  - `magnitude_mean`, `magnitude_max`: mean/max of peak magnitudes
  - `rep_time_mean`, `rep_time_std`: mean and variability of time between repetitions
  - `fatigue_index`: ratio comparing average first-half vs second-half peak magnitudes
  - `slowdown_rate`: linear slope of rep intervals (positive = slowing)
  - `vertical_amplitude`: peak-to-peak range across X/Y/Z
  - `hesitations`: number of unusually long rep intervals

These metrics are computed in `src/analysis/movement_analysis.py` and returned as a dataclass for easy serialization.

---

## Visualization

- The Streamlit UI is in `app_improved.py` and uses `src/visualization/movement_visualizer.py` for plotting helpers.
- For each file the app shows:
  - Left/right magnitude traces with detected peaks
  - X/Y/Z traces per side
  - A summarized metrics panel (magnitude_mean, magnitude_max, fatigue_index, rep_time_mean, etc.)

Notes on plotting behavior:
- Plots are generated using Matplotlib/Seaborn and displayed in Streamlit.
- If the right side appears compressed or mis-ordered, the code sorts entries by `timestamp` per side and normalizes time to the start of each side.

---

## Tuning peak detection

Recommended knobs:

- `prominence` (preferred): controls how much a peak stands out relative to its neighbors. Default: dynamic `max(0.2, 0.25*range)`.
- `distance`: minimal horizontal separation (in samples) between peaks. Default used in code: 8–10 samples.
- `height`: absolute threshold for peaks (useful when signals are on the same absolute scale).


---


## Example sample entry

```json
{
  "timestamp": 33598,
  "deviceId": "RIGHT-ANKLE",
  "accelerometer": { "x": -0.19, "y": -0.17, "z": 0.21 },
  "gyroscope": { "x": 2.93, "y": -7.5, "z": -35.4 }
}
```
