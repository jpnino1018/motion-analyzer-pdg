import streamlit as st
import pandas as pd
import os
import tempfile
import json
from main import procesar_archivo
from preprocessing.visualization import visualizar_movimiento_xyz_por_lado

DB_FILE = "pacientes.csv"

# =========================
# Funciones de persistencia
# =========================
def cargar_pacientes():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(columns=["codigo", "nombre", "correo"])

def guardar_pacientes(df):
    df.to_csv(DB_FILE, index=False)

# =========================
# Inicializar estado
# =========================
if "view" not in st.session_state:
    st.session_state.view = "Registro / Búsqueda"
if "paciente" not in st.session_state:
    st.session_state.paciente = None

import streamlit as st

# Optional: Set page config
st.set_page_config(page_title="Motion-Analyzer", layout="wide")

# Top bar
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
# Funciones de UI
# =========================
def mostrar_registro_busqueda():
    st.header("Registro o búsqueda de paciente")
    codigo = st.text_input("Código del paciente", key="codigo_busqueda")

    if st.button("Buscar / Crear", key="btn_buscar_crear"):
        pacientes = cargar_pacientes()

        # Normalizar todo a string y quitar espacios
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
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())

        # Procesar archivo
        resultados = procesar_archivo(temp_path, ejercicio)

        # ------------------------
        # Mostrar resultados
        # ------------------------
        st.markdown("### 📊 Resultados del análisis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Magnitud Promedio (m/s²)", f"{resultados['activo_mag_prom']:.2f}")
        col2.metric("Magnitud Máxima (m/s²)", f"{resultados['activo_mag_max']:.2f}")
        col3.metric("Fatiga (%)", f"{resultados['activo_fatiga']*100:.1f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Tiempo Promedio por Repetición (ms)", f"{resultados['activo_tiempo_prom_rep']:.0f}")
        col5.metric("Variabilidad del Ritmo (ms)", f"{resultados['activo_variabilidad_tiempo']:.0f}")
        col6.metric("Cantidad de titubeos", f"{resultados['activo_titubeos']:.0f}")

        st.markdown("#### 📄 Datos completos")
        with st.expander("Ver JSON completo"):
            st.json(resultados)

        # ------------------------
        # Graficar
        # ------------------------

        # Graficar solo el lado activo
        fig = visualizar_movimiento_xyz_por_lado(temp_path, resultados['lado_activo'])
        if fig:
            st.pyplot(fig)
        else:
            st.warning("No se encontraron datos para graficar.")

    if st.button("Cerrar sesión"):
        st.session_state.paciente = None
        st.session_state.view = "Registro / Búsqueda"
        st.rerun()

# =========================
# Router de vistas
# =========================
if st.session_state.view == "Registro / Búsqueda":
    mostrar_registro_busqueda()
elif st.session_state.view == "Registro nuevo":
    mostrar_registro_nuevo()
elif st.session_state.view == "Subir datos":
    mostrar_subida_datos()
