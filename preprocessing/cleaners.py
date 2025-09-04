# cleaners.py
def recortar_inactividad(datos, umbral=0.5, min_len=50):
    """
    Trim the initial segment with no significant movement.

    Parameters:
        datos (list[dict]): standardized sensor data
        umbral (float): minimum magnitude (m/s²) to consider as movement
        min_len (int): number of consecutive samples above threshold to trigger

    Returns:
        list[dict]: trimmed data starting from first detected movement
    """
    mags = []
    for d in datos:
        acc = d.get("accelerometer")
        if acc:
            mag = (acc["x"]**2 + acc["y"]**2 + acc["z"]**2) ** 0.5
            mags.append(mag)
        else:
            mags.append(0)

    # Find first index where enough activity happens
    activo_idx = 0
    for i in range(len(mags) - min_len):
        if all(m > umbral for m in mags[i:i+min_len]):
            activo_idx = i
            break

    return datos[activo_idx:]
