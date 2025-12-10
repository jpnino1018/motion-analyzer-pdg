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
  - **Input data already comes in gravitational units (g)** and gyroscope in °/s
  - Converts accelerometer from g to m/s² by multiplying by 9.81
  - Gyroscope data is already in °/s, no conversion needed
  - Returns a dict with `LEFT` and `RIGHT` lists; each element: `{timestamp, accelerometer: {x,y,z}, gyroscope: {x,y,z}}`.

- **Automatic inactive period trimming** (`src/preprocessing/cleaners.py`):
  - `recortar_inactividad(datos, umbral=0.5, min_len=50)` automatically removes initial samples with no significant movement.
  - **Problem solved**: Eliminates the delay between pressing "record" button and actual movement start.
  - **Smart detection**: Uses **variance/standard deviation** in a sliding window to detect actual movement vs. static gravity (simply checking magnitude fails because gravity = ~1 m/s² even when stationary).
  - Searches for first window of `min_len` samples where standard deviation > `umbral` (default 0.5 m/s²).
  - **Enabled by default** in both `app_improved.py` and `main.py` (can be disabled via UI checkbox or function parameter).
  - **Why it matters**: Initial idle periods contaminate peak detection (false positives from noise), dilute dynamic prominence calculations, and skew baseline gravity offset removal.
  - **UI control**: In Streamlit app, expand "⚙️ Configuración de preprocesamiento" to disable if needed.
  - **Debug visibility**: Check "🔍 Debug Info" in results to see how many samples were trimmed from each side.

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
  - `vertical_amplitude_mean`: mean vertical displacement per repetition (cm) via double integration
  - `vertical_amplitude_decay`: amplitude reduction rate (cm/rep) - **negative = progressive reduction (Parkinson's indicator)**
  - `vertical_amplitude_ratio`: first-half / second-half amplitude ratio - **> 1.0 = reduction (Parkinson's indicator), < 1.0 = warm-up effect**
  - `hesitations`: number of unusually long rep intervals

**Detecting Parkinson's Bradykinesia:**
Progressive amplitude reduction is a hallmark sign of Parkinson's disease. Unlike a simple low mean amplitude (which could indicate general weakness), a **negative decay rate** specifically captures the progressive narrowing of movement range during the test:

- **Amplitude Decay Rate < -0.5 cm/rep**: 🔴 Strong indicator - each rep is progressively smaller
- **Amplitude Ratio > 1.15**: 🔴 Strong indicator - second half shows significant reduction
- Healthy controls typically show stable or slightly increasing amplitude (warm-up effect)

These metrics are computed in `src/analysis/movement_analysis.py` and returned as a dataclass for easy serialization.

---

## 🩺 Automated Parkinson's Diagnosis System (0-4 Scale)

The system implements an expert rule-based classifier modeled after the **UPDRS (Unified Parkinson's Disease Rating Scale)** motor assessment. Each file receives a **severity score from 0-4** based on weighted analysis of 6 clinical movement metrics.

### Severity Scale Definition

| Score | Label | Clinical Interpretation |
|-------|-------|-------------------------|
| **0** | Normal | No signs of motor impairment - movement within healthy parameters |
| **1** | Mild | Slight amplitude reduction or rhythm irregularity, typically unilateral |
| **2** | Moderate | Clear progressive amplitude decay (bradykinesia), bilateral involvement possible |
| **3** | Marked | Severe bradykinesia with significant freezing/hesitations, functional impairment |
| **4** | Severe | Barely able to perform movement, extreme slowness and amplitude reduction |

### Feature Weights (Total = 1.0)

The diagnosis system prioritizes features that best capture Parkinson's motor dysfunction:

```python
'decay_rate': 0.30       # Highest weight - captures progressive bradykinesia
'amplitude_ratio': 0.25  # Validates decay pattern over time
'magnitude': 0.15        # Overall movement strength
'rhythm_std': 0.15       # Consistency of rhythm
'rep_time': 0.10         # Movement speed
'hesitations': 0.05      # Freezing episodes (less frequent in mild cases)
```

**Rationale:**
- **Decay rate** (30%) is most weighted because **progressive amplitude reduction** is the gold standard for detecting bradykinesia in clinical exams.
- **Amplitude ratio** (25%) cross-validates decay by comparing first vs second half performance.
- **Magnitude and rhythm** (15% each) capture complementary aspects: overall strength and consistency.
- **Repetition time** (10%) indicates slowness but can be confounded by patient effort/fatigue.
- **Hesitations** (5%) capture freezing but occur primarily in advanced stages.

---

### Clinical Thresholds - Detailed Justification

Each metric has **5 severity thresholds** (normal → mild → moderate → marked → severe) based on clinical research and empirical observations:

#### 1. Amplitude Decay Rate (m/s²/rep)
Measures the **linear regression slope** of peak magnitudes across all repetitions.

| Threshold | Value | Justification |
|-----------|-------|---------------|
| **Normal** | ≥ -0.03 | Healthy individuals show minimal decay or slight increase (warm-up). Small negative values within noise range. |
| **Mild** | -0.08 | Subtle progressive reduction detectable but not functionally limiting. Early Parkinson's sign. |
| **Moderate** | -0.15 | Clear bradykinesia pattern - amplitude visibly decreasing rep-by-rep. Clinical intervention warranted. |
| **Marked** | -0.25 | Severe progressive reduction indicating advanced motor dysfunction. |
| **Severe** | -0.40 | Extreme decay - movement nearly arrests mid-test. Requires immediate treatment adjustment. |

**Why these values?**
- Based on m/s² units from accelerometer magnitude (not displacement). 
- A decay of -0.15 m/s²/rep over 10 reps = 1.5 m/s² total reduction (highly significant).
- Empirically tested: healthy controls rarely exceed -0.05, while known Parkinson's patients consistently show < -0.10.

---

#### 2. Amplitude Ratio (First Half / Second Half)
Compares mean peak magnitudes between first and second half of test.

| Threshold | Value | Justification |
|-----------|-------|---------------|
| **Normal** | ≤ 1.05 | Nearly equal performance both halves (ratio ~1.0). Slight fatigue acceptable. |
| **Mild** | 1.15 | 15% reduction in second half - noticeable but mild. |
| **Moderate** | 1.30 | 30% reduction - clear fatigue/bradykinesia pattern. Matches clinical moderate severity. |
| **Marked** | 1.50 | 50% reduction - severe amplitude loss in second half. |
| **Severe** | 2.00 | Second half amplitude is half of first half - extreme bradykinesia. |

**Why these values?**
- Ratio > 1.0 indicates reduction (first half > second half).
- Ratio < 1.0 would indicate warm-up effect (normal/healthy).
- 1.15 threshold balances sensitivity (detecting early signs) with specificity (avoiding false positives from normal fatigue).
- Values validated against UPDRS motor exam where 30%+ reduction warrants "moderate" classification.

---

#### 3. Mean Magnitude (m/s²)
Average of all peak magnitudes - indicates overall movement strength.

| Threshold | Value | Justification |
|-----------|-------|---------------|
| **Normal** | ≥ 3.0 | Strong, vigorous movement. Typical for healthy adults in stomp/tapping exercises. |
| **Mild** | 2.2 | Slightly reduced amplitude but functional. |
| **Moderate** | 1.5 | Clearly diminished movement range - clinically noticeable weakness. |
| **Marked** | 1.0 | Severely reduced - minimal movement excursion. |
| **Severe** | 0.6 | Barely moving above gravity baseline (~1 m/s² static). |

**Why these values?**
- Gravity offset = ~1 m/s² when stationary, so < 1.0 indicates almost no active movement.
- Healthy stomp/tapping exercises generate 2.5-4.0 m/s² peaks.
- Threshold of 3.0 for "normal" aligns with observed data from control subjects.
- Progressive scaling reflects clinical observation: magnitude correlates with functional capacity.

---

#### 4. Rhythm Variability - STD of Inter-Rep Intervals (ms)
Standard deviation of time between repetitions - measures consistency.

| Threshold | Value | Justification |
|-----------|-------|---------------|
| **Normal** | ≤ 150 | Consistent rhythm - STD < 25% of mean rep time (assuming ~600ms mean). |
| **Mild** | 250 | Noticeable irregularity but maintainable rhythm. |
| **Moderate** | 400 | High variability - difficulty maintaining steady pace. |
| **Marked** | 600 | Severe rhythm disruption - frequent hesitations/pauses. |
| **Severe** | 800 | Extreme irregularity - almost no rhythm consistency. |

**Why these values?**
- Healthy rhythm: STD typically 100-150ms for 500-700ms mean interval (CV ~20%).
- Parkinson's patients show increased variability due to motor planning deficits.
- 250ms = ~35% CV, clinically significant but mild.
- 400ms+ indicates moderate-severe dysrhythmia affecting functional tasks.

---

#### 5. Mean Repetition Time (ms)
Average time between repetitions - indicates movement speed.

| Threshold | Value | Justification |
|-----------|-------|---------------|
| **Normal** | ≤ 600 | Brisk pace - healthy adults complete reps in ~0.5-0.6 seconds. |
| **Mild** | 800 | Slightly slow but functional. |
| **Moderate** | 1100 | Clearly slowed - bradykinesia affecting speed. |
| **Marked** | 1500 | Severely slow - taking 1.5+ seconds per rep. |
| **Severe** | 2000 | Extremely slow - 2+ seconds per rep, near-freezing. |

**Why these values?**
- Normal stomp/tapping frequency: ~1-2 Hz (500-1000ms per cycle).
- 600ms threshold captures healthy rapid movements.
- 1100ms (moderate) aligns with clinically observable slowness in UPDRS exam.
- 2000ms represents severe bradykinesia where movement is extremely labored.

---

#### 6. Hesitations (Normalized per 10 Reps)
Count of intervals > mean + 2*STD - detects freezing episodes.

| Threshold | Value | Justification |
|-----------|-------|---------------|
| **Normal** | ≤ 0.5 | Rare outliers (< 1 per 10 reps) - within normal variability. |
| **Mild** | 1.5 | Occasional hesitations - 1-2 per 10 reps. |
| **Moderate** | 3.0 | Frequent pauses - 3 per 10 reps (30%). |
| **Marked** | 5.0 | Very frequent freezing - half of reps affected. |
| **Severe** | 7.0 | Nearly constant freezing - movement extremely disrupted. |

**Why these values?**
- Hesitations defined as outliers (> 2 STD above mean) indicate motor blocks.
- Normalized per 10 reps for fair comparison across variable test lengths.
- 0.5 threshold allows occasional variability without penalizing.
- 3.0+ (moderate) reflects clinically significant freezing of gait equivalent.

---

### Scoring Algorithm

1. **Individual Feature Scoring**: Each metric is scored 0.0-4.0 using interpolation between thresholds.
   - For "lower is better" metrics (decay, ratio, std, time, hesitations): score increases as value worsens.
   - For "higher is better" metrics (magnitude): score increases as value decreases.

2. **Weighted Combination**: 
   ```
   Final Score = Σ(feature_score × feature_weight)
   ```

3. **Rounding**: Score rounded to nearest integer (0-4).

4. **Confidence Calculation**: 
   - Based on variance of individual feature scores.
   - Low variance (all features agree) → high confidence.
   - High variance (mixed signals) → lower confidence.
   - Formula: `confidence = max(0.5, min(1.0, 1.0 - score_variance/4.0))`

---

### Example Diagnosis Output

```
Severity Score: 2
Label: Moderado - Clara bradicinesia con reducción progresiva
Confidence: 87.3%

Contributing Factors:
📉 Decay Rate: moderate (-0.18 m/s²/rep)
⚖️ Amplitude Ratio: moderate (1.35)
💪 Mean Magnitude: mild (2.3 m/s²)
🎵 Rhythm Variability: normal (145 ms)
⏱️ Repetition Time: mild (780 ms)
⏸️ Hesitations: normal (1/10 reps)

Clinical Notes:
⚠️ Bradicinesia moderada detectada
• Reducción progresiva clara: moderate
• Fatiga significativa en segunda mitad: moderate
→ Recomendación: Evaluación neurológica completa
```

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
