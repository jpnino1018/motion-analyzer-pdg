import json
import matplotlib.pyplot as plt

def visualizar_movimiento_xyz_por_lado(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        datos = json.load(f)

    # Detectar si hay que acceder a una clave como 'imuData'
    if isinstance(datos, dict):
        for k, v in datos.items():
            if isinstance(v, list) and all(isinstance(i, dict) for i in v):
                print(f"Usando clave '{k}' como fuente de datos")
                datos = v
                break
        else:
            print("No se encontró una lista válida en el JSON.")
            return

    # Inicializar estructuras para cada lado
    lados = {
        'LEFT': {'timestamps': [], 'x': [], 'y': [], 'z': []},
        'RIGHT': {'timestamps': [], 'x': [], 'y': [], 'z': []}
    }

    # Filtrar y organizar datos
    for d in datos:
        device = d.get('deviceId', '')
        acc = d.get('accelerometer')
        
        if device.startswith('LEFT') and acc:
            lados['LEFT']['timestamps'].append(d['timestamp'])
            lados['LEFT']['x'].append(acc['x'])
            lados['LEFT']['y'].append(acc['y'])
            lados['LEFT']['z'].append(acc['z'])
        elif device.startswith('RIGHT') and acc:
            lados['RIGHT']['timestamps'].append(d['timestamp'])
            lados['RIGHT']['x'].append(acc['x'])
            lados['RIGHT']['y'].append(acc['y'])
            lados['RIGHT']['z'].append(acc['z'])

    # Graficar por lado
    for lado, valores in lados.items():
        if not valores['timestamps']:
            print(f"No hay datos para {lado}")
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(valores['timestamps'], valores['x'], label='X')
        plt.plot(valores['timestamps'], valores['y'], label='Y')
        plt.plot(valores['timestamps'], valores['z'], label='Z')
        plt.title(f'Movimiento en lado {lado}')
        plt.xlabel('Timestamp')
        plt.ylabel('Aceleración')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Ejecutar
visualizar_movimiento_xyz_por_lado('./data/stomp/stomp5.json')
