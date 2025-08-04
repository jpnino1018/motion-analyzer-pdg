import math
import numpy as np
from scipy.signal import find_peaks


def calcular_magnitud(gyro):
    return math.sqrt(gyro['x']**2 + gyro['y']**2 + gyro['z']**2)


def extraer_magnitudes(datos):
    return [calcular_magnitud(d['gyroscope']) for d in datos], [d['timestamp'] for d in datos]


def detectar_picos(magnitudes, altura_min=0.2, distancia_min=5):
    # Puedes ajustar altura/distancia según tus datos
    peaks, _ = find_peaks(magnitudes, height=altura_min, distance=distancia_min)
    return peaks


def calcular_intervalos(timestamps, peaks):
    return [timestamps[peaks[i+1]] - timestamps[peaks[i]] for i in range(len(peaks)-1)]


def calcular_fatiga(magnitudes, peaks):
    n = len(peaks)
    if n < 6: return 0  # No se puede calcular bien
    primera_mitad = [magnitudes[p] for p in peaks[:n//2]]
    segunda_mitad = [magnitudes[p] for p in peaks[n//2:]]
    prom_1 = np.mean(primera_mitad)
    prom_2 = np.mean(segunda_mitad)
    return (prom_1 - prom_2) / prom_1 if prom_1 != 0 else 0


def resumen_de_movimiento(datos):
    magnitudes, timestamps = extraer_magnitudes(datos)
    peaks = detectar_picos(magnitudes)

    if not peaks.any():
        return {
            'n_peaks': 0,
            'mag_prom': 0,
            'mag_max': 0,
            'ritmo_prom': 0,
            'ritmo_var': 0,
            'fatiga': 0
        }

    mag_prom = np.mean([magnitudes[p] for p in peaks])
    mag_max = np.max([magnitudes[p] for p in peaks])
    
    intervalos = calcular_intervalos(timestamps, peaks)
    ritmo_prom = np.mean(intervalos) if intervalos else 0
    ritmo_var = np.std(intervalos) if intervalos else 0

    fatiga = calcular_fatiga(magnitudes, peaks)

    return {
        'n_peaks': len(peaks),
        'mag_prom': mag_prom,
        'mag_max': mag_max,
        'ritmo_prom': ritmo_prom,
        'ritmo_var': ritmo_var,
        'fatiga': fatiga
    }
