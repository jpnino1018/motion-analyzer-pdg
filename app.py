import streamlit as st
import pandas as pd
import os
import tempfile
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from src.preprocessing.movement_processor import MovementProcessor
from src.preprocessing.signal_processing import AccelerometerData
from src.visualization.movement_visualizer import MovementVisualizer
from src.analysis.parkinson_diagnosis import ParkinsonDiagnosisSystem

# Configure matplotlib for Streamlit
plt.style.use('default')
sns.set_style("whitegrid")

# =========================
# Helper Functions
# =========================
def get_base64_image(image_path):
    """Convert image to base64 string for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

def validate_json_format(data):
    """Validate if JSON has the expected format"""
    if not isinstance(data, dict):
        return False, "El archivo JSON debe contener un objeto principal."
    
    # Check for expected format 1: imuData
    if "imuData" in data:
        if not isinstance(data["imuData"], list):
            return False, "El campo 'imuData' debe ser una lista."
        if len(data["imuData"]) == 0:
            return False, "El campo 'imuData' está vacío."
        # Check first item structure
        sample = data["imuData"][0]
        required_fields = ["deviceId", "timestamp", "accelerometer"]
        missing = [f for f in required_fields if f not in sample]
        if missing:
            return False, f"Faltan campos requeridos en imuData: {', '.join(missing)}"
        return True, None
    
    # Check for expected format 2: izquierda/derecha
    if "izquierda" in data or "derecha" in data:
        for side in ["izquierda", "derecha"]:
            if side in data:
                if not isinstance(data[side], list):
                    return False, f"El campo '{side}' debe ser una lista."
                if len(data[side]) == 0:
                    return False, f"El campo '{side}' está vacío."
                # Check first item structure
                sample = data[side][0]
                required_fields = ["millis", "x", "y", "z"]
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    return False, f"Faltan campos requeridos en {side}: {', '.join(missing)}"
        return True, None
    
    return False, "Formato JSON no reconocido."

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Foot Motion Analyzer - i2t",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_and_process_movement_data(file_path: str, exercise: str, trim_inactive: bool = True):
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
            # Data already comes in g units and °/s, just convert g to m/s²
            for side, raw_key in [("LEFT", "izquierda"), ("RIGHT", "derecha")]:
                if raw_key in data:
                    for d in data[raw_key]:
                        normalized[side].append({
                            "timestamp": d["millis"],
                            "accelerometer": {
                                "x": d["x"] * 9.81,  # Already in g, convert to m/s²
                                "y": d["y"] * 9.81,
                                "z": d["z"] * 9.81,
                            },
                            "gyroscope": {
                                "x": d["a"],  # Already in °/s
                                "y": d["b"],
                                "z": d["g"],
                            }
                        })

    # Separate and sort data by side FIRST (critical for correct visualization)
    left_data = []
    right_data = []
    
    if "imuData" in data:
        for d in data["imuData"]:
            if "LEFT" in d["deviceId"].upper():
                left_data.append(d)
            elif "RIGHT" in d["deviceId"].upper():
                right_data.append(d)
    else:
        # Already separated in normalized structure
        left_data = normalized["LEFT"]
        right_data = normalized["RIGHT"]
    
    # Sort by timestamp (fixes the fan pattern issue)
    left_data.sort(key=lambda x: x["timestamp"])
    right_data.sort(key=lambda x: x["timestamp"])
    
    # Store original lengths before trimming
    original_left_len = len(left_data)
    original_right_len = len(right_data)
    
    # Apply trimming if enabled
    from src.preprocessing.cleaners import recortar_inactividad
    if trim_inactive:
        left_data = recortar_inactividad(left_data, umbral=0.5, min_len=50)
        right_data = recortar_inactividad(right_data, umbral=0.5, min_len=50)
    
    # Calculate trimming stats
    left_trimmed_count = original_left_len - len(left_data)
    right_trimmed_count = original_right_len - len(right_data)
    
    # Update normalized with sorted and trimmed data
    normalized["LEFT"] = left_data
    normalized["RIGHT"] = right_data
    
    # Process both sides (now with pre-sorted and pre-trimmed data, so disable internal trimming)
    left_metrics = processor.process_movement_data(normalized["LEFT"], trim_inactive=False)
    right_metrics = processor.process_movement_data(normalized["RIGHT"], trim_inactive=False)
    
    # Determine active side
    active_side = "LEFT" if left_metrics.magnitude_mean > right_metrics.magnitude_mean else "RIGHT"
    
    # Get timestamps after sorting and trimming
    left_timestamps = np.array([d["timestamp"] for d in left_data])
    right_timestamps = np.array([d["timestamp"] for d in right_data])
    
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
    left_acc_data = processor._extract_accelerometer_data(left_data, left_timestamps)
    right_acc_data = processor._extract_accelerometer_data(right_data, right_timestamps)
    
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
        'left_trim_info': {'original': original_left_len, 'trimmed': left_trimmed_count, 'remaining': len(normalized["LEFT"])},
        'right_trim_info': {'original': original_right_len, 'trimmed': right_trimmed_count, 'remaining': len(normalized["RIGHT"])},
        **{f'active_{k}': v for k, v in vars(active_metrics).items()},
        **{f'passive_{k}': v for k, v in vars(passive_metrics).items()},
    }, (left_acc_data, right_acc_data, left_peaks, right_peaks)

# =========================
# Styling
# =========================
st.markdown("""
    <style>
        /* Fixed topbar */
        .fixed-topbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #1a4d2e 0%, #0d3321 100%);
            padding: 1rem 2rem;
            z-index: 999;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .fixed-topbar h1 {
            color: white;
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
        }
        .fixed-topbar p {
            color: white;
            font-size: 0.9rem;
            margin: 0;
            opacity: 0.9;
        }
        
        /* Add padding to main content to account for fixed topbar */
        .main .block-container {
            padding-top: 5rem;
        }
        
        /* Main header styling */
        .main-header {
            background: linear-gradient(135deg, #1a4d2e 0%, #0d3321 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .main-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
        }
        .main-header p {
            color: white;
            font-size: 1.1rem;
            text-align: center;
            margin-top: 0.5rem;
        }
        
        /* Info cards */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #1a4d2e;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        .info-card h3 {
            color: #1a4d2e;
            margin-top: 0;
        }
        
        /* Metric styling */
        .stMetric {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .stMetric [data-testid="stMetricLabel"] {
            color: #1f2937 !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #111827 !important;
            font-weight: 600;
        }
        .stMetric [data-testid="stMetricDelta"] {
            color: #4b5563 !important;
        }
        
        /* File uploader */
        .uploadedFile {
            border: 2px dashed #1a4d2e !important;
            border-radius: 8px;
        }
        
        /* Logo styling in topbar */
        .topbar-logo {
            height: 40px;
            margin: 0 10px;
            vertical-align: middle;
            filter: brightness(0) invert(1);
        }
        .topbar-logos {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        /* Hide Streamlit branding and deployment UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        [data-testid="stToolbar"] {display: none;}
        .stActionButton {display: none;}
        div[data-testid="stDecoration"] {display: none;}
        button[kind="header"] {display: none;}
        
        /* Hide GitHub avatar and fork button */
        [data-testid="stHeader"] {display: none;}
        iframe[title="st.iframe"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Load logos
logo1_base64 = get_base64_image("assets/logo1.png")
logo2_base64 = get_base64_image("assets/logo2.png")

# Build logo HTML
logos_html = ""
if logo1_base64:
    logos_html += f'<img src="data:image/png;base64,{logo1_base64}" class="topbar-logo" alt="Logo 1">'
if logo2_base64:
    logos_html += f'<img src="data:image/png;base64,{logo2_base64}" class="topbar-logo" alt="Logo 2">'

# Fixed topbar
st.markdown(f"""
    <div class="fixed-topbar">
        <div>
            <h1>Motion Analyzer</h1>
        </div>
        <div>
            <p>Parkinson's Assessment System</p>
        </div>
        <div class="topbar-logos">
            {logos_html}
        </div>
    </div>
""", unsafe_allow_html=True)

# =========================
# Main Application
# =========================

# Header
st.markdown("""
    <div class="main-header">
        <h1>Análisis de Taloneo y Tobillos</h1>
        <p>Herramienta de procesamiento y evaluación de enfermedades neuromotoras para movimientos de pie</p>
    </div>
""", unsafe_allow_html=True)


# Exercise selection
col1, col2 = st.columns([1, 3])
with col1:
    ejercicio = st.radio("Tipo de ejercicio", ["stomp", "tapping"], 
                        help="Selecciona el tipo de ejercicio realizado por el paciente")

# Preprocessing controls
with st.expander("Configuración de preprocesamiento"):
    trim_inactive = st.checkbox(
        "Recortar período inactivo inicial",
        value=True,
        help="Elimina automáticamente los primeros segundos sin movimiento antes de procesar"
    )
    if trim_inactive:
        st.info("Se eliminarán los datos iniciales hasta detectar movimiento activo (detecta variación en magnitud, no solo gravedad estática)")

# File upload
st.markdown("### Cargar archivo de datos")
st.markdown("""
Esta aplicación procesa datos de sensores IMU para detectar patrones de movimiento 
asociados con bradicinesia y otras manifestaciones de Parkinson.
""")
file = st.file_uploader("Sube archivo JSON con datos de movimiento", type="json")

if file:
    # Save uploaded file temporarily
    temp_path = os.path.join(tempfile.gettempdir(), file.name)
    with open(temp_path, "wb") as f:
        f.write(file.read())

    try:
        # Load and validate JSON
        with open(temp_path, "r") as f:
            data = json.load(f)
        
        # Validate format
        is_valid, error_message = validate_json_format(data)
        
        if not is_valid:
            # Display styled error popup
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                    border-left: 5px solid #fca5a5;
                ">
                    <h3 style="color: white; margin: 0 0 0.5rem 0;">
                        Formato JSON No Reconocido
                    </h3>
                    <p style="color: white; font-size: 1rem; margin: 0;">
                        {error_message}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Process file with new modules
        with st.spinner("Procesando datos..."):
            results, viz_data = load_and_process_movement_data(temp_path, ejercicio, trim_inactive=trim_inactive)
        left_acc_data, right_acc_data, left_peaks, right_peaks = viz_data
    
    except json.JSONDecodeError as e:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                border-left: 5px solid #fca5a5;
            ">
                <h3 style="color: white; margin: 0 0 0.5rem 0;">
                    Error de JSON
                </h3>
                <p style="color: white; font-size: 1rem; margin: 0;">
                    El archivo no es un JSON válido. Verifica la sintaxis del archivo.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    except Exception as e:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                border-left: 5px solid #fca5a5;
            ">
                <h3 style="color: white; margin: 0 0 0.5rem 0;">
                    Error de Procesamiento
                </h3>
                <p style="color: white; font-size: 1rem; margin: 0;">
                    Ocurrió un error al procesar el archivo. Verifica que el formato sea correcto.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Display results
    st.success("Análisis completado")
    st.markdown("---")
    st.markdown("### Métricas Principales")
    
    # Primary metrics (most relevant for study)
    col1, col2, col3 = st.columns(3)
    col1.metric("Magnitud Promedio (m/s²)", f"{results['active_magnitude_mean']:.2f}")
    col2.metric("Tiempo Promedio por Repetición (ms)", f"{results['active_rep_time_mean']:.0f}")
    col3.metric("Variabilidad del Ritmo (ms)", f"{results['active_rep_time_std']:.0f}")

    # =========================
    # PARKINSON'S DIAGNOSIS
    # =========================
    st.markdown("---")
    st.markdown("### Diagnóstico Automatizado")
    
    # Perform diagnosis
    diagnosis_system = ParkinsonDiagnosisSystem()
    diagnosis = diagnosis_system.diagnose(results)
    
    # Display severity score with color coding
    severity_colors = {
        0: "#10b981",  # Green
        1: "#84cc16",  # Light green
        2: "#f59e0b",  # Orange
        3: "#ef4444",  # Red
        4: "#dc2626"   # Dark red
    }
    severity_color = severity_colors[diagnosis.severity_score]
    
    # Large prominent severity display
    st.markdown(f"""
        <div style="background: {severity_color}; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h1 style="color: white; margin: 0; font-size: 3rem;">{diagnosis.severity_score}</h1>
            <h3 style="color: white; margin: 0.5rem 0 0 0;">{diagnosis.severity_label}</h3>
            <p style="color: white; margin: 0.5rem 0 0 0;">Confianza: {diagnosis.confidence*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Contributing factors breakdown
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### Factores Contribuyentes")
        for factor, value in diagnosis.contributing_factors.items():
            factor_names = {
                'decay_rate': 'Tasa de Reducción',
                'amplitude_ratio': 'Ratio Amplitud',
                'magnitude': 'Magnitud Media',
                'rhythm_variability': 'Variabilidad Ritmo',
                'repetition_time': 'Tiempo Repetición',
                'hesitations': 'Titubeos'
            }
            st.write(f"**{factor_names.get(factor, factor)}:** {value}")
    
    with col_d2:
        st.markdown("#### Notas Clínicas")
        st.info(diagnosis.clinical_notes)
    
    st.markdown("---")

    # Amplitude progression metrics (Parkinson's indicators)
    st.markdown("#### Progresión de Amplitud (Indicadores de Bradicinesia)")
    col_a, col_b, col_c = st.columns(3)
    
    decay_rate = results['active_vertical_amplitude_decay']
    decay_color = "🔴" if decay_rate < -0.15 else "🟡" if decay_rate < -0.05 else "🟢"
    col_a.metric("Tasa de Reducción (m/s²/rep)", 
                f"{decay_color} {decay_rate:.3f}",
                help="Cambio en magnitud de picos por repetición. Negativo = reducción progresiva (Parkinson)")
    
    ratio = results['active_vertical_amplitude_ratio']
    # CORRECTED: ratio > 1.0 means first half is LARGER (reduction in second half)
    ratio_color = "🔴" if ratio > 1.3 else "🟡" if ratio > 1.1 else "🟢"
    col_b.metric("Ratio Primera/Segunda Mitad", 
                f"{ratio_color} {ratio:.2f}",
                help="> 1.1 = reducción significativa en segunda mitad, < 0.9 = aumento (warm-up normal)")
    
    col_c.metric("Interpretación",
                "Posible bradicinesia" if (decay_rate < -0.1 or ratio > 1.15) else "Normal",
                help="Basado en regresión lineal de magnitudes de picos (decay) y comparación primera/segunda mitad (ratio)")

    # Secondary metrics
    st.markdown("#### Métricas Adicionales")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Magnitud Máxima (m/s²)", f"{results['active_magnitude_max']:.2f}")
    col6.metric("Fatiga (%)", f"{results['active_fatigue_index']*100:.1f}")
    col7.metric("Enlentecimiento (ms/rep)", f"{results['active_slowdown_rate']:.2f}")
    col8.metric("Titubeos", f"{results['active_hesitations']}")

    # Show detailed results
    st.markdown("#### Datos Completos")
    with st.expander("Ver JSON completo"):
        st.json(results)

    # Create visualization - only show active side
    st.markdown("### Visualización de Datos")
    
    # Determine which side to plot
    is_left_active = results['active_side'] == 'LEFT'
    active_acc_data = left_acc_data if is_left_active else right_acc_data
    active_peaks = left_peaks if is_left_active else right_peaks
    active_trim_info = results.get('left_trim_info') if is_left_active else results.get('right_trim_info')
    side_name = "Izquierdo" if is_left_active else "Derecho"
    
    # Create 2x2 grid: [Magnitude with peaks, Peak progression] [X/Y/Z traces, Rep intervals]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert time to seconds
    time_active = (active_acc_data.timestamps - active_acc_data.timestamps[0]) / 1000
    mag_active = active_acc_data.magnitude
    
    # Plot 1: Magnitude with detected peaks
    ax1.plot(time_active, mag_active, 'b-', linewidth=1.5, label='Magnitud')
    if len(active_peaks) > 0:
        ax1.plot(time_active[active_peaks], mag_active[active_peaks], 'ro', markersize=8, label=f'Picos ({len(active_peaks)})')
    
    # Add trimming annotation
    if active_trim_info and active_trim_info['trimmed'] > 0:
        trimmed = active_trim_info['trimmed']
        ax1.text(0.02, 0.98, f'{trimmed} muestras recortadas', 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax1.set_title(f'Lado {side_name} - Magnitud con picos detectados', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Magnitud (m/s²)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Peak magnitude progression (decay visualization)
    if len(active_peaks) > 0:
        peak_mags = mag_active[active_peaks]
        rep_numbers = np.arange(1, len(peak_mags) + 1)
        
        ax2.scatter(rep_numbers, peak_mags, s=100, c=rep_numbers, cmap='coolwarm', 
                   edgecolors='black', linewidths=1.5, zorder=3)
        
        # Add trend line (linear regression)
        if len(peak_mags) >= 2:
            z = np.polyfit(rep_numbers, peak_mags, 1)
            p = np.poly1d(z)
            ax2.plot(rep_numbers, p(rep_numbers), "r--", linewidth=2, alpha=0.8, 
                    label=f'Tendencia (pendiente={results["active_vertical_amplitude_decay"]:.3f} m/s²/rep)')
        
        # Shade first/second half for ratio visualization
        mid = len(peak_mags) // 2
        ax2.axvspan(0.5, mid + 0.5, alpha=0.15, color='green', label='Primera mitad')
        ax2.axvspan(mid + 0.5, len(peak_mags) + 0.5, alpha=0.15, color='orange', label='Segunda mitad')
        
        ax2.set_title('Progresión de amplitud de picos (indicador de bradicinesia)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Número de repetición')
        ax2.set_ylabel('Magnitud del pico (m/s²)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(rep_numbers)
    else:
        ax2.text(0.5, 0.5, 'No hay picos detectados', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Progresión de amplitud de picos', fontsize=12, fontweight='bold')
    
    # Plot 3: X/Y/Z acceleration traces
    ax3.plot(time_active, active_acc_data.x, 'r-', label='X', alpha=0.7, linewidth=1.2)
    ax3.plot(time_active, active_acc_data.y, 'g-', label='Y', alpha=0.7, linewidth=1.2)
    ax3.plot(time_active, active_acc_data.z, 'b-', label='Z', alpha=0.7, linewidth=1.2)
    ax3.set_title('Aceleración por ejes', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Aceleración (m/s²)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Repetition intervals (rhythm analysis)
    if len(active_peaks) > 1:
        intervals = np.diff(active_acc_data.timestamps[active_peaks])
        interval_numbers = np.arange(1, len(intervals) + 1)
        
        mean_interval = results['active_rep_time_mean']
        std_interval = results['active_rep_time_std']
        
        ax4.scatter(interval_numbers, intervals, s=80, c='steelblue', 
                   edgecolors='black', linewidths=1.5, zorder=3)
        ax4.axhline(y=mean_interval, color='green', linestyle='--', linewidth=2, 
                   label=f'Media: {mean_interval:.0f} ms')
        ax4.axhline(y=mean_interval + std_interval, color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.7, label=f'±1 SD: {std_interval:.0f} ms')
        ax4.axhline(y=mean_interval - std_interval, color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
        
        # Highlight hesitations (outliers)
        threshold = mean_interval + 2 * std_interval
        hesitation_mask = intervals > threshold
        if np.any(hesitation_mask):
            ax4.scatter(interval_numbers[hesitation_mask], intervals[hesitation_mask], 
                       s=150, c='red', marker='x', linewidths=3, zorder=4, label='Titubeos')
        
        ax4.set_title('Intervalos entre repeticiones (análisis de ritmo)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Intervalo #')
        ax4.set_ylabel('Tiempo entre picos (ms)')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(interval_numbers)
    else:
        ax4.text(0.5, 0.5, 'Insuficientes picos para calcular intervalos', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Intervalos entre repeticiones', fontsize=12, fontweight='bold')
    
    plt.suptitle(f"{ejercicio.title()} - Lado {side_name} (Activo)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    plt.close('all')