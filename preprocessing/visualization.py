import json
import matplotlib.pyplot as plt

def visualizar_movimiento_xyz_por_lado(ruta_archivo, lado_activo):
    """
    Grafica las señales de aceleración para el lado activo detectado.
    
    Parámetros:
        ruta_archivo (str): Ruta al archivo JSON con datos del IMU.
        lado_activo (str): 'LEFT' o 'RIGHT' (determinado previamente).
    
    Retorna:
        fig (matplotlib.figure.Figure): Figura con la gráfica.
    """
    with open(ruta_archivo, 'r') as f:
        datos = json.load(f)

    # Detectar si hay que acceder a 'imuData'
    if isinstance(datos, dict):
        for k, v in datos.items():
            if isinstance(v, list) and all(isinstance(i, dict) for i in v):
                datos = v
                break
        else:
            return None

    # Inicializar estructura para solo el lado activo
    valores = {'timestamps': [], 'x': [], 'y': [], 'z': []}

    # Filtrar solo datos del lado activo
    for d in datos:
        device = d.get('deviceId', '')
        acc = d.get('accelerometer')

        if device.startswith(lado_activo) and acc:
            valores['timestamps'].append(d['timestamp'])
            valores['x'].append(acc['x'])
            valores['y'].append(acc['y'])
            valores['z'].append(acc['z'])

    if not valores['timestamps']:
        return None

    # Crear figura
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
