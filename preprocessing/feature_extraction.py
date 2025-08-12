import math
import numpy as np
from scipy.signal import find_peaks

# --- Utilidades de cálculo ---
def calcular_magnitud(gyro):
    """Calcula magnitud vectorial de la señal giroscópica."""
    return math.sqrt(gyro['x']**2 + gyro['y']**2 + gyro['z']**2)

def extraer_magnitudes(datos):
    """Extrae magnitudes y timestamps de los datos crudos."""
    return [calcular_magnitud(d['gyroscope']) for d in datos], [d['timestamp'] for d in datos]

def suavizar_senal(magnitudes, ventana=3):
    """Suaviza la señal usando media móvil."""
    if len(magnitudes) < ventana:
        return magnitudes
    return np.convolve(magnitudes, np.ones(ventana)/ventana, mode='same')

# --- Detección de repeticiones ---
def detectar_repeticiones(magnitudes, altura_min=0.2, distancia_min=10, n_reps=10):
    """
    Detecta picos representando repeticiones.
    Retorna índices de picos y ajusta si hay más de las repeticiones esperadas.
    """
    peaks, _ = find_peaks(magnitudes, height=altura_min, distance=distancia_min)
    
    if len(peaks) > n_reps:
        sorted_peaks = sorted(peaks, key=lambda p: magnitudes[p], reverse=True)[:n_reps]
        sorted_peaks.sort()
        return sorted_peaks
    return list(peaks)

# --- Cálculos de métricas ---
def calcular_tiempos_por_rep(timestamps, peaks):
    """Calcula la duración de cada repetición."""
    return [timestamps[peaks[i+1]] - timestamps[peaks[i]] for i in range(len(peaks)-1)]

def calcular_fatiga(magnitudes, peaks):
    """Calcula índice de fatiga entre la primera y segunda mitad de repeticiones."""
    n = len(peaks)
    if n < 4:
        return 0
    primera_mitad = [magnitudes[p] for p in peaks[:n//2]]
    segunda_mitad = [magnitudes[p] for p in peaks[n//2:]]
    prom_1 = np.mean(primera_mitad)
    prom_2 = np.mean(segunda_mitad)
    return (prom_1 - prom_2) / prom_1 if prom_1 != 0 else 0

# --- Resumen principal ---
def resumen_de_movimiento(datos, n_reps=10):
    if not datos:
        return {
            'n_reps': 0,
            'mag_prom': 0,
            'mag_max': 0,
            'tiempo_prom_rep': 0,
            'variabilidad_tiempo': 0,
            'fatiga': 0
        }

    magnitudes, timestamps = extraer_magnitudes(datos)
    magnitudes_suavizadas = suavizar_senal(magnitudes)
    peaks = detectar_repeticiones(magnitudes_suavizadas, n_reps=n_reps)

    if not peaks:
        return {
            'n_reps': 0,
            'mag_prom': 0,
            'mag_max': 0,
            'tiempo_prom_rep': 0,
            'variabilidad_tiempo': 0,
            'fatiga': 0
        }

    mag_prom = np.mean([magnitudes[p] for p in peaks])
    mag_max = np.max([magnitudes[p] for p in peaks])
    
    tiempos_rep = calcular_tiempos_por_rep(timestamps, peaks)
    tiempo_prom_rep = np.mean(tiempos_rep) if tiempos_rep else 0
    variabilidad_tiempo = np.std(tiempos_rep) if tiempos_rep else 0

    fatiga = calcular_fatiga(magnitudes, peaks)

    return {
        'n_reps': len(peaks),
        'mag_prom': mag_prom,
        'mag_max': mag_max,
        'tiempo_prom_rep': tiempo_prom_rep,
        'variabilidad_tiempo': variabilidad_tiempo,
        'fatiga': fatiga
    }
