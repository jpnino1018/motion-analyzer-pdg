# Motion Data Visualizer and Analyzer

This project is designed to process, analyze, and visualize motion data collected from wearable IMU sensors placed on different body parts. The primary goal is to assist in diagnostics and physical assessments by identifying asymmetries and motion patterns.

---

## Features Implemented So Far

### 1. **Device Data Filtering**
- Only relevant body-mounted devices are considered (e.g., `LEFT-ANKLE`, `RIGHT-ANKLE`, `LEFT-KNEE`, `RIGHT-KNEE`, etc.).
- Data from `BASE-SPINE` or non-limb sensors are excluded to avoid skewing analysis.

**Why it's important:**  
Removing central device data like `BASE-SPINE` helps isolate limb-specific movement, which is crucial for detecting lateral imbalances or dysfunction.

---

### 2. **Automatic Side Classification**
- Data is grouped based on device IDs containing `LEFT` or `RIGHT` regardless of the exact joint (e.g., `LEFT-FOOT`, `LEFT-KNEE`, `LEFT-ANKLE` → grouped under `LEFT`).

**Why it's important:**  
Patients or subjects may wear multiple sensors per limb. Grouping all relevant data under unified sides allows for a more accurate side-to-side comparison.

---

### 3. **Motion Visualization (X, Y, Z Axes)**
- For each limb side (`LEFT`, `RIGHT`), the acceleration values in the X, Y, and Z axes are plotted over time.
- Each plot allows interpretation of motion patterns and detection of irregularities.

**Why it's important:**  
Visual cues are a powerful tool for clinicians and researchers to spot inconsistencies, tremors, or limitations in movement visually, aiding in diagnosis or therapy monitoring.

---

### 4. **Magnitude Comparison**
- The average magnitude of acceleration is computed for each side.
- Helps determine which side has higher physical activity or involvement.

**Why it's important:**  
This quantitative feature can indicate functional differences between limbs, which is essential for assessing rehabilitation progress, asymmetries, or dominance in motor activity.

---

## CSV Exported Variables Explained

When exporting data to CSV, we generate high-level metrics that are useful for interpretation and diagnosis:

| Variable      | Description                                                                 | Why It's Important                                           |
|---------------|-----------------------------------------------------------------------------|--------------------------------------------------------------|
| `side`        | Indicates whether the data corresponds to the LEFT or RIGHT side            | Enables side-by-side comparisons to detect imbalance         |
| `magnitude`   | Average acceleration magnitude for the given side                           | Helps quantify movement intensity and detect hypoactivity     |
| `fatigue`     | Trend of decreasing movement over time (based on slope of magnitudes)       | Useful to identify muscular fatigue or endurance loss        |
| `asymmetry`   | Difference in magnitude between left and right sides                        | Key indicator in gait and postural analysis                  |
| `rhythm`      | Frequency or regularity of movement (steps, oscillations, etc.)             | Abnormal rhythm can suggest tremors, instability, or spasticity |

**Why this matters:**  
These metrics provide a simplified but effective way to detect issues such as limb dominance, fatigue-induced degradation, and uncoordinated movement. They are especially valuable in physical therapy, sports science, and neurological assessments.

---

## Relevance for Diagnostics

These features lay the groundwork for identifying motor impairments, asymmetries, or compensatory behavior in patients with neurological or orthopedic conditions. By combining magnitude analysis and axis-based visualization, the system offers both objective metrics and visual feedback to clinicians.

---

## Example Data Format

```json
{
  "timestamp": 33598,
  "deviceId": "RIGHT-ANKLE",
  "accelerometer": {
    "x": -0.19,
    "y": -0.17,
    "z": 0.21
  },
  "gyroscope": {
    "x": 2.93,
    "y": -7.5,
    "z": -35.4
  }
}

---

## Documentación ampliada

Para una descripción completa de las variables analizadas, el flujo de adquisición de datos (con diagramas) y los protocolos de captura de ejercicios, consulta el documento:

- `docs/01_variables_y_protocolo.md`