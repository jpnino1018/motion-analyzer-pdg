import os
import json
import csv
from preprocessing.feature_extraction import resumen_de_movimiento

def cargar_datos(archivo_json):
    with open(archivo_json) as f:
        return json.load(f)['imuData']

def filtrar_datos_por_lado(datos, lado_tag):
    return [d for d in datos if lado_tag in d['deviceId']]

def identificar_lado_dominante(left_data, right_data):
    def mag_prom(data):
        if not data:
            return 0
        vectores_validos = [d['accelerometer'] for d in data if 'accelerometer' in d and all(k in d['accelerometer'] for k in ('x', 'y', 'z'))]
        if not vectores_validos:
            return 0
        return sum((v['x']**2 + v['y']**2 + v['z']**2)**0.5 for v in vectores_validos) / len(vectores_validos)


    left_mag = mag_prom(left_data)
    right_mag = mag_prom(right_data)

    return 'LEFT' if left_mag > right_mag else 'RIGHT'

def calcular_asimetria(val1, val2):
    if val1 == 0 and val2 == 0:
        return 0
    return abs(val1 - val2) / max(val1, val2)

def procesar_archivo(archivo_json, ejercicio):
    datos = cargar_datos(archivo_json)
    left_data = filtrar_datos_por_lado(datos, 'LEFT')
    right_data = filtrar_datos_por_lado(datos, 'RIGHT')

    lado_activo = identificar_lado_dominante(left_data, right_data)
    lado_pasivo = 'RIGHT' if lado_activo == 'LEFT' else 'LEFT'

    datos_activo = left_data if lado_activo == 'LEFT' else right_data
    datos_pasivo = right_data if lado_activo == 'LEFT' else left_data

    features_activo = resumen_de_movimiento(datos_activo)
    features_pasivo = resumen_de_movimiento(datos_pasivo)

    asimetria_mag = calcular_asimetria(features_activo['mag_prom'], features_pasivo['mag_prom'])
    asimetria_ritmo = calcular_asimetria(features_activo['ritmo_prom'], features_pasivo['ritmo_prom'])

    return {
        'archivo': os.path.basename(archivo_json),
        'ejercicio': ejercicio,
        'lado_activo': lado_activo,
        **{f'activo_{k}': v for k, v in features_activo.items()},
        **{f'pasivo_{k}': v for k, v in features_pasivo.items()},
        'asimetria_mag': round(asimetria_mag, 3),
        'asimetria_ritmo': round(asimetria_ritmo, 3)
    }

def procesar_todos_los_archivos():
    base_dir = './data'
    ejercicios = ['stomp', 'tapping']
    resultados = []

    for ejercicio in ejercicios:
        carpeta = os.path.join(base_dir, ejercicio)
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.json'):
                ruta = os.path.join(carpeta, archivo)
                resultados.append(procesar_archivo(ruta, ejercicio))

    return resultados

def guardar_csv(resultados, nombre_csv='resultados/resultados.csv'):
    if not resultados:
        print("⚠️ No hay resultados para guardar.")
        return

    campos = list(resultados[0].keys())
    os.makedirs(os.path.dirname(nombre_csv), exist_ok=True)

    with open(nombre_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"✅ Archivo guardado: {nombre_csv}")

if __name__ == '__main__':
    resultados = procesar_todos_los_archivos()
    guardar_csv(resultados)
