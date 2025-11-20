def recortar_inactividad(datos, umbral=0.5, min_len=50):
    """
    Trim the initial segment with no significant movement.
    Uses VARIANCE in magnitude to detect actual movement vs. static gravity.

    Parameters:
        datos (list[dict]): standardized sensor data
        umbral (float): minimum STANDARD DEVIATION of magnitude to consider as movement
        min_len (int): number of consecutive samples above threshold to trigger

    Returns:
        list[dict]: trimmed data starting from first detected movement
    """
    if len(datos) < min_len:
        return datos
    
    mags = []
    for d in datos:
        acc = d.get("accelerometer")
        if acc and all(k in acc for k in ("x", "y", "z")):
            mag = (acc["x"]**2 + acc["y"]**2 + acc["z"]**2) ** 0.5
            mags.append(mag)
        else:
            mags.append(0)

    if len(mags) < min_len * 2:
        return datos

    # Use a sliding window to find where variance increases (actual movement)
    # Static gravity shows low variance, movement shows high variance
    activo_idx = 0
    window_size = min_len
    
    for i in range(len(mags) - window_size):
        window = mags[i:i+window_size]
        variance = sum((x - sum(window)/len(window))**2 for x in window) / len(window)
        std_dev = variance ** 0.5
        
        # Movement detected when standard deviation exceeds threshold
        if std_dev > umbral:
            activo_idx = i
            break

    return datos[activo_idx:]
