import json

def cargar_datos_json(ruta_archivo):
    """
    Normaliza datos desde formato viejo (con 'imuData') o formato nuevo (izquierda/derecha).
    Devuelve siempre un diccionario: {"LEFT": [...], "RIGHT": [...]}
    """
    with open(ruta_archivo, "r") as f:
        data = json.load(f)

    normalizado = {"LEFT": [], "RIGHT": []}

    # ===== Caso 1: viejo con envoltorio 'imuData' =====
    if isinstance(data, dict) and "imuData" in data:
        imu_data = data["imuData"]
        for d in imu_data:
            lado = "LEFT" if "LEFT" in d["deviceId"].upper() else "RIGHT"
            normalizado[lado].append({
                "timestamp": d.get("timestamp"),
                "accelerometer": d.get("accelerometer", {}),
                "gyroscope": d.get("gyroscope", {})
            })
        return normalizado

    # ===== Caso 2: nuevo crudo (izquierda / derecha) =====
    elif isinstance(data, dict) and ("izquierda" in data or "derecha" in data):
        # Conversión desde cuentas crudas
        ACC_SCALE = 16384.0  # cuentas por g (ajustar si tu sensor usa otro)
        GYRO_SCALE = 131.0   # cuentas por °/s

        if "izquierda" in data:
            for d in data["izquierda"]:
                normalizado["LEFT"].append({
                    "timestamp": d["millis"],
                    "accelerometer": {
                        "x": (d["x"] / ACC_SCALE) * 9.81,
                        "y": (d["y"] / ACC_SCALE) * 9.81,
                        "z": (d["z"] / ACC_SCALE) * 9.81,
                    },
                    "gyroscope": {
                        "x": d["a"] / GYRO_SCALE,
                        "y": d["b"] / GYRO_SCALE,
                        "z": d["g"] / GYRO_SCALE,
                    }
                })

        if "derecha" in data:
            for d in data["derecha"]:
                normalizado["RIGHT"].append({
                    "timestamp": d["millis"],
                    "accelerometer": {
                        "x": (d["x"] / ACC_SCALE) * 9.81,
                        "y": (d["y"] / ACC_SCALE) * 9.81,
                        "z": (d["z"] / ACC_SCALE) * 9.81,
                    },
                    "gyroscope": {
                        "x": d["a"] / GYRO_SCALE,
                        "y": d["b"] / GYRO_SCALE,
                        "z": d["g"] / GYRO_SCALE,
                    }
                })
        return normalizado

    else:
        raise ValueError(f"Formato JSON no reconocido. Claves encontradas: {list(data.keys())}")
