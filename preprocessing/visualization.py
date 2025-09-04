import json
import matplotlib.pyplot as plt
from preprocessing.cleaners import recortar_inactividad

def visualizar_movimiento_xyz_por_lado(ruta_archivo, lado_activo):
    """
    Grafica las señales de aceleración para el lado activo detectado.

    Parámetros:
        ruta_archivo (str): Ruta al archivo JSON con datos del IMU.
        lado_activo (str): 'LEFT' o 'RIGHT' (determinado previamente).

    Retorna:
        fig (matplotlib.figure.Figure) o None
    """
    with open(ruta_archivo, 'r') as f:
        datos = json.load(f)

    # === Detectar formato viejo con "imuData"
    if isinstance(datos, dict) and "imuData" in datos:
        datos = datos["imuData"]

    # === Detectar formato nuevo con "izquierda"/"derecha"
    elif isinstance(datos, dict) and any(k in datos for k in ["izquierda", "derecha"]):
        # Unir los lados en una sola lista con deviceId
        normalizados = []
        if "izquierda" in datos:
            for d in datos["izquierda"]:
                normalizados.append({
                    "timestamp": d.get("millis", d.get("t", 0)),
                    "x": d["x"] / 16384.0 * 9.81,  # convertir crudos a m/s²
                    "y": d["y"] / 16384.0 * 9.81,
                    "z": d["z"] / 16384.0 * 9.81,
                    "deviceId": "LEFT-ANKLE"
                })
        if "derecha" in datos:
            for d in datos["derecha"]:
                normalizados.append({
                    "timestamp": d.get("millis", d.get("t", 0)),
                    "x": d["x"] / 16384.0 * 9.81,
                    "y": d["y"] / 16384.0 * 9.81,
                    "z": d["z"] / 16384.0 * 9.81,
                    "deviceId": "RIGHT-ANKLE"
                })
        datos = normalizados

    elif not isinstance(datos, list):
        return None  # formato no reconocido

    # === Recortar inactividad antes de graficar ===
    datos = recortar_inactividad(datos)

    # === Inicializar estructura para el lado activo ===
    valores = {'timestamps': [], 'x': [], 'y': [], 'z': []}

    for d in datos:
        device = d.get("deviceId", "")

        # --- Formato viejo ---
        if device.startswith(lado_activo) and "accelerometer" in d:
            acc = d["accelerometer"]
            valores['timestamps'].append(d['timestamp'])
            valores['x'].append(acc['x'])
            valores['y'].append(acc['y'])
            valores['z'].append(acc['z'])

        # --- Formato nuevo ---
        elif device.startswith(lado_activo) and all(k in d for k in ("x", "y", "z")):
            valores['timestamps'].append(d['timestamp'])
            valores['x'].append(d['x'])
            valores['y'].append(d['y'])
            valores['z'].append(d['z'])

    if not valores['timestamps']:
        return None

    # === Crear figura ===
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(valores['timestamps'], valores['x'], label='X', alpha=0.7)
    ax.plot(valores['timestamps'], valores['y'], label='Y', alpha=0.7)
    ax.plot(valores['timestamps'], valores['z'], label='Z', alpha=0.7)

    ax.set_title(f'Movimiento en lado {lado_activo}')
    ax.set_xlabel('Timestamp (ms)')
    ax.set_ylabel('Aceleración (m/s²)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig
