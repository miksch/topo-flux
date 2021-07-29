import numpy as np

def sound_to_ts_K(sound, eq_type='gill_manual', e_a=None):
    """
    Convert speed of sound to 
    """

    if eq_type=="gill_manual":
        return (sound ** 2) / 403
    if eq_type=="s_audio":
        return ((sound / 331.3) ** 2 - 1) * 273.15 + 273.15
    else:
        return None