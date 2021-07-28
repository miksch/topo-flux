import numpy as np

def sound_to_ts_K(sound, eq_type='gill_manual', e_a=None):
    """
    Convert speed of sound to 
    """

    if eq_type=="gill_manual":
        return (sound ** 2) / 403
    else:
        return None