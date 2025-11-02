import streamlit as st
import pandas as pd
import os
import tempfile
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.movement_processor import MovementProcessor
from src.preprocessing.signal_processing import AccelerometerData
from src.visualization.movement_visualizer import MovementVisualizer

# Configure matplotlib for Streamlit
plt.style.use('default')
sns.set_style("whitegrid")

DB_FILE = "pacientes.csv"

# =========================
# Data persistence functions
# =========================
def cargar_pacientes():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(columns=["codigo", "nombre", "correo"])

def guardar_pacientes(df):
    df.to_csv(DB_FILE, index=False)

def load_and_process_movement_data(file_path: str, exercise: str):
    """Load and process movement data using new processing modules"""
    processor = MovementProcessor()
    visualizer = MovementVisualizer()
    
    # Load data
    with open(file_path, "r") as f:
        data = json.load(f)

    # Normalize data
    normalized = {"LEFT": [], "RIGHT": []}
    if isinstance(data, dict):
        if "imuData" in data:
            for d in data["imuData"]:
                side = "LEFT" if "LEFT" in d["deviceId"].upper() else "RIGHT"
                normalized[side].append({
                    "timestamp": d.get("timestamp"),
                    "accelerometer": d.get("accelerometer", {}),
                    "gyroscope": d.get("gyroscope", {})
                })
        elif "izquierda" in data or "derecha" in data:
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

    # Process both sides
    left_metrics = processor.process_movement_data(normalized["LEFT"])
    right_metrics = processor.process_movement_data(normalized["RIGHT"])
    
    # Determine active side
    active_side = "LEFT" if left_metrics.magnitude_mean > right_metrics.magnitude_mean else "RIGHT"
    
    # Get data for visualization
    # Separate and sort data by deviceId first
    left_data = []
    right_data = []
    
    if "imuData" in data:
        for d in data["imuData"]:
            if "LEFT" in d["deviceId"].upper():
                left_data.append(d)
            elif "RIGHT" in d["deviceId"].upper():
                right_data.append(d)
    
    # Sort data by timestamp
    left_data.sort(key=lambda x: x["timestamp"])
    right_data.sort(key=lambda x: x["timestamp"])
    
    # Store sorted data back in normalized structure
    normalized["LEFT"] = left_data
    normalized["RIGHT"] = right_data
    
    # Get timestamps after sorting
    left_timestamps = np.array([d["timestamp"] for d in normalized["LEFT"]])
    right_timestamps = np.array([d["timestamp"] for d in normalized["RIGHT"]])
    
    # Debug info
    st.write("Debug Info:")
    st.write(f"Left timestamps range: {left_timestamps[0]} to {left_timestamps[-1]}")
    st.write(f"Right timestamps range: {right_timestamps[0]} to {right_timestamps[-1]}")
    st.write(f"Left data points: {len(left_data)}")
    st.write(f"Right data points: {len(right_data)}")
    
    # Normalize timestamps relative to their start times
    def normalize_timestamps(data, timestamps):
        start_time = timestamps[0]
        normalized_ts = (timestamps - start_time) / 1000.0  # Convert to seconds
        acc_data = []
        
        for d in data:
            acc = d.get('accelerometer', {})
            if all(k in acc for k in ('x', 'y', 'z')):
                acc_data.append(acc)
                
        return normalized_ts, acc_data
    
    left_ts_norm, left_acc = normalize_timestamps(normalized["LEFT"], left_timestamps)
    right_ts_norm, right_acc = normalize_timestamps(normalized["RIGHT"], right_timestamps)
    
    # Create AccelerometerData objects
    left_acc_data = processor._extract_accelerometer_data(normalized["LEFT"], left_timestamps)
    right_acc_data = processor._extract_accelerometer_data(normalized["RIGHT"], right_timestamps)
    
    # Debug magnitudes
    st.write("\nMagnitude ranges:")
    st.write(f"Left magnitude range: {left_acc_data.magnitude.min():.2f} to {left_acc_data.magnitude.max():.2f}")
    st.write(f"Right magnitude range: {right_acc_data.magnitude.min():.2f} to {right_acc_data.magnitude.max():.2f}")
    
    # Detect peaks using a dynamic prominence so only big changes are selected
    left_mags = left_acc_data.magnitude
    right_mags = right_acc_data.magnitude

    def compute_prominence(mags: np.ndarray) -> float:
        if mags.size == 0:
            return 0.2
        r = mags.max() - mags.min()
        return max(0.2, 0.25 * r)

    left_prom = compute_prominence(left_mags)
    right_prom = compute_prominence(right_mags)

    left_peaks = processor.signal_processor.detect_peaks(left_mags,
                                                       prominence=left_prom,
                                                       distance=8)
    right_peaks = processor.signal_processor.detect_peaks(right_mags,
                                                        prominence=right_prom,
                                                        distance=8)
    
    # Prepare metrics for output
    active_metrics = left_metrics if active_side == "LEFT" else right_metrics
    passive_metrics = right_metrics if active_side == "LEFT" else left_metrics
    
    return {
        'file': os.path.basename(file_path),
        'exercise': exercise,
        'active_side': active_side,
        **{f'active_{k}': v for k, v in vars(active_metrics).items()},
        **{f'passive_{k}': v for k, v in vars(passive_metrics).items()},
    }, (left_acc_data, right_acc_data, left_peaks, right_peaks)

# =========================
# Initialize state
# =========================
if "view" not in st.session_state:
    st.session_state.view = "Registro / Búsqueda"
if "paciente" not in st.session_state:
    st.session_state.paciente = None

# Optional: Set page config
st.set_page_config(page_title="Motion-Analyzer", layout="wide")

# Top bar styling
st.markdown(
    """
    <style>
        .top-bar {
            background-color: #1f1f1f;
            padding: 15px 30px;
            width: 100%;
            position: fixed;
            top: 0;
            left: 0;
            z-index: 100;
        }
        .top-bar h1 {
            color: white;
            font-size: 24px;
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
        }
        .spacer {
            height: 70px;
        }
    </style>
    <div class="top-bar">
        <h1>Motion-Analyzer (Feet-Exercises)</h1>
    </div>
    <div class="spacer"></div>
    """,
    unsafe_allow_html=True
)

# =========================
# UI Functions
# =========================
def mostrar_registro_busqueda():
    st.header("Registro o búsqueda de paciente")
    codigo = st.text_input("Código del paciente", key="codigo_busqueda")

    if st.button("Buscar / Crear", key="btn_buscar_crear"):
        pacientes = cargar_pacientes()
        codigo_str = str(codigo).strip()
        pacientes_codigos = pacientes["codigo"].astype(str).str.strip()

        if codigo_str in pacientes_codigos.values:
            paciente = pacientes[pacientes_codigos == codigo_str].iloc[0]
            st.session_state.paciente = paciente.to_dict()
            st.session_state.view = "Subir datos"
            st.rerun()
        else:
            st.session_state.nuevo_codigo = codigo_str
            st.session_state.view = "Registro nuevo"
            st.rerun()

def mostrar_registro_nuevo():
    st.header("Registro de nuevo paciente")
    nombre = st.text_input("Nombre", key="nombre_nuevo")
    correo = st.text_input("Correo", key="correo_nuevo")

    if st.button("Registrar", key="btn_registrar"):
        pacientes = cargar_pacientes()
        nuevo = pd.DataFrame(
            [[st.session_state.nuevo_codigo, nombre, correo]],
            columns=["codigo", "nombre", "correo"]
        )
        pacientes = pd.concat([pacientes, nuevo], ignore_index=True)
        guardar_pacientes(pacientes)

        st.session_state.paciente = nuevo.iloc[0].to_dict()
        st.session_state.pop("nuevo_codigo", None)
        st.session_state.view = "Subir datos"
        st.rerun()

def mostrar_subida_datos():
    paciente = st.session_state.paciente
    st.subheader(f"Paciente: {paciente['nombre']} ({paciente['codigo']})")

    ejercicio = st.radio("Selecciona ejercicio", ["stomp", "tapping"])
    file = st.file_uploader("Sube archivo JSON", type="json")

    if file:
        # Save uploaded file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())

        # Process file with new modules
        results, viz_data = load_and_process_movement_data(temp_path, ejercicio)
        left_acc_data, right_acc_data, left_peaks, right_peaks = viz_data

        # Display results
        st.markdown("### 📊 Resultados del análisis")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Magnitud Promedio (m/s²)", f"{results['active_magnitude_mean']:.2f}")
        col2.metric("Magnitud Máxima (m/s²)", f"{results['active_magnitude_max']:.2f}")
        col3.metric("Fatiga (%)", f"{results['active_fatigue_index']*100:.1f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Tiempo Promedio por Repetición (ms)", f"{results['active_rep_time_mean']:.0f}")
        col5.metric("Variabilidad del Ritmo (ms)", f"{results['active_rep_time_std']:.0f}")
        col6.metric("Cantidad de titubeos", f"{results['active_hesitations']}")

        # Show detailed results
        st.markdown("#### 📄 Datos completos")
        with st.expander("Ver JSON completo"):
            st.json(results)

        # Create visualization
        visualizer = MovementVisualizer()
        
        # Plot movement data
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Left side plots
        time_left = (left_acc_data.timestamps - left_acc_data.timestamps[0]) / 1000  # Convert to seconds
        mag_left = left_acc_data.magnitude
        
        ax1.plot(time_left, mag_left, 'b-', label='Magnitude')
        if len(left_peaks) > 0:
            ax1.plot(time_left[left_peaks], mag_left[left_peaks], 'ro', label='Peaks')
        ax1.set_title('Left Side')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Magnitude (m/s²)')
        ax1.legend()
        
        ax3.plot(time_left, left_acc_data.x, 'r-', label='X', alpha=0.7)
        ax3.plot(time_left, left_acc_data.y, 'g-', label='Y', alpha=0.7)
        ax3.plot(time_left, left_acc_data.z, 'b-', label='Z', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.legend()
        
        # Right side plots
        time_right = (right_acc_data.timestamps - right_acc_data.timestamps[0]) / 1000
        mag_right = right_acc_data.magnitude
        
        ax2.plot(time_right, mag_right, 'b-', label='Magnitude')
        if len(right_peaks) > 0:
            ax2.plot(time_right[right_peaks], mag_right[right_peaks], 'ro', label='Peaks')
        ax2.set_title('Right Side')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Magnitude (m/s²)')
        ax2.legend()
        
        ax4.plot(time_right, right_acc_data.x, 'r-', label='X', alpha=0.7)
        ax4.plot(time_right, right_acc_data.y, 'g-', label='Y', alpha=0.7)
        ax4.plot(time_right, right_acc_data.z, 'b-', label='Z', alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Acceleration (m/s²)')
        ax4.legend()
        
        plt.suptitle(f"{ejercicio.title()} - {results['active_side']} Active")
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close('all')

    if st.button("Cerrar sesión"):
        st.session_state.paciente = None
        st.session_state.view = "Registro / Búsqueda"
        st.rerun()

# =========================
# View Router
# =========================
if st.session_state.view == "Registro / Búsqueda":
    mostrar_registro_busqueda()
elif st.session_state.view == "Registro nuevo":
    mostrar_registro_nuevo()
elif st.session_state.view == "Subir datos":
    mostrar_subida_datos()