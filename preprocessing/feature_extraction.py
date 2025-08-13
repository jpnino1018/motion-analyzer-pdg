import math
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

def calcular_magnitud(acc):
    return math.sqrt(acc['x']**2 + acc['y']**2 + acc['z']**2)

def extraer_magnitudes_y_ejes(datos):
    mags, ts, eje_x, eje_y, eje_z = [], [], [], [], []
    for d in datos:
        acc = d.get('accelerometer')
        if acc and all(k in acc for k in ('x', 'y', 'z')):
            mags.append(calcular_magnitud(acc))
            ts.append(d['timestamp'])
            eje_x.append(acc['x'])
            eje_y.append(acc['y'])
            eje_z.append(acc['z'])
    return mags, ts, eje_x, eje_y, eje_z

def detectar_repeticiones(magnitudes, altura_min=0.2, distancia_min=10, n_reps=10):
    peaks, _ = find_peaks(magnitudes, height=altura_min, distance=distancia_min)
    if len(peaks) > n_reps:
        sorted_peaks = sorted(peaks, key=lambda p: magnitudes[p], reverse=True)[:n_reps]
        sorted_peaks.sort()
        return sorted_peaks
    return list(peaks)

def calcular_tiempos_por_rep(timestamps, peaks):
    return [timestamps[peaks[i+1]] - timestamps[peaks[i]] for i in range(len(peaks)-1)]

def calcular_fatiga(magnitudes, peaks):
    n = len(peaks)
    if n < 4:
        return 0
    primera_mitad = [magnitudes[p] for p in peaks[:n//2]]
    segunda_mitad = [magnitudes[p] for p in peaks[n//2:]]
    prom_1 = np.mean(primera_mitad)
    prom_2 = np.mean(segunda_mitad)
    return (prom_1 - prom_2) / prom_1 if prom_1 != 0 else 0


def calcular_enlentecimiento(tiempos_rep):
    if len(tiempos_rep) < 3:
        return 0
    x = np.arange(len(tiempos_rep))
    pendiente, _, _, _, _ = linregress(x, tiempos_rep)
    return pendiente  # ms por repetición (positivo = enlentecimiento)
def calcular_amplitud_vertical(eje_x, eje_y, eje_z):
    rangos = {
        'x': max(eje_x) - min(eje_x) if eje_x else 0,
        'y': max(eje_y) - min(eje_y) if eje_y else 0,
        'z': max(eje_z) - min(eje_z) if eje_z else 0
    }
    return max(rangos.values())  # Amplitud máxima

def contar_titubeos(tiempos_rep):
    if len(tiempos_rep) < 3:
        return 0
    promedio = np.mean(tiempos_rep)
    std = np.std(tiempos_rep)
    umbral = promedio + 1.5 * std
    return sum(1 for t in tiempos_rep if t > umbral)

def resumen_de_movimiento(datos, n_reps=10):
    magnitudes, timestamps, eje_x, eje_y, eje_z = extraer_magnitudes_y_ejes(datos)
    peaks = detectar_repeticiones(magnitudes, n_reps=n_reps)

    if not peaks:
        return {
            'n_reps': 0,
            'mag_prom': 0,
            'mag_max': 0,
            'tiempo_prom_rep': 0,
            'variabilidad_tiempo': 0,
            'fatiga': 0,
            'enlentecimiento': 0,
            'amplitud_vertical': 0,
            'titubeos': 0
        }

    mag_prom = np.mean([magnitudes[p] for p in peaks])
    mag_max = np.max([magnitudes[p] for p in peaks])

    tiempos_rep = calcular_tiempos_por_rep(timestamps, peaks)
    tiempo_prom_rep = np.mean(tiempos_rep) if tiempos_rep else 0
    variabilidad_tiempo = np.std(tiempos_rep) if tiempos_rep else 0

    fatiga = calcular_fatiga(magnitudes, peaks)
    enlentecimiento = calcular_enlentecimiento(tiempos_rep)
    amplitud_vertical = calcular_amplitud_vertical(eje_x, eje_y, eje_z)
    titubeos = contar_titubeos(tiempos_rep)

    return {
        'n_reps': len(peaks),
        'mag_prom': mag_prom,
        'mag_max': mag_max,
        'tiempo_prom_rep': tiempo_prom_rep,
        'variabilidad_tiempo': variabilidad_tiempo,
        'fatiga': fatiga,
        'enlentecimiento': enlentecimiento,
        'amplitud_vertical': amplitud_vertical,
        'titubeos': titubeos
    }
