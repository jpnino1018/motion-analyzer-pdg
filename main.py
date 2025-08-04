import os
import json
import csv
from preprocessing.feature_extraction import resumen_de_movimiento

def cargar_datos_desde_archivo(archivo):
    with open(archivo) as f:
        contenido = json.load(f)
        return contenido['imuData']

def separar_lado(datos, lado_tag):
    return [d for d in datos if lado_tag in d['deviceId']]

def procesar_archivo(ruta, ejercicio, archivo_json):
    ruta_completa = os.path.join(ruta, archivo_json)
    datos = cargar_datos_desde_archivo(ruta_completa)

    resultados = []
    for lado_tag in ['LEFT', 'RIGHT']:
        datos_lado = separar_lado(datos, lado_tag)
        features = resumen_de_movimiento(datos_lado)
        resultados.append({
            'archivo': archivo_json,
            'ejercicio': ejercicio,
            'lado': lado_tag,
            **features
        })
    return resultados

def procesar_todos_los_archivos():
    base_dir = './data'
    carpetas = ['stomp', 'tapping']
    resultados_totales = []

    for carpeta in carpetas:
        ruta_carpeta = os.path.join(base_dir, carpeta)
        for archivo in os.listdir(ruta_carpeta):
            if archivo.endswith('.json'):
                resultados = procesar_archivo(ruta_carpeta, carpeta, archivo)
                resultados_totales.extend(resultados)

    return resultados_totales

def guardar_resultados_en_csv(resultados, nombre_csv='resultados.csv'):
    campos = ['archivo', 'ejercicio', 'lado', 'n_peaks', 'mag_prom', 'mag_max', 'ritmo_prom', 'ritmo_var', 'fatiga']
    with open(nombre_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"✅ Resultados guardados en: {nombre_csv}")

if __name__ == '__main__':
    resultados = procesar_todos_los_archivos()
    guardar_resultados_en_csv(resultados)
