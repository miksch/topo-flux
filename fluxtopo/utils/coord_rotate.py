import numpy as np
import pandas as pd
from numba import jit

def rotate_2d(x0, y0, theta):
    """
    Rotate point or vector by theta (+ anticlockwise)
    
    Args:
        x0 (float) : x-component of wind vector (u) [m/s]
        y0 (float) : y-component of wind vector (v) [m/s]
        theta (float) : angle of rotation (+ anticlockwise) [deg]
        
    Returns:
        x1 (float) : rotated x-component of wind vector
        y1 (float) : rotated y-component of wind vector
    """
    
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))
    
    x1 = x0 * cos_theta - y0 * sin_theta
    y1 = x0 * sin_theta + y0 * cos_theta
    
    return x1, y1



def double_rotate(df, u, v, w):
    """
    Perform double rotation method on wind components

    Follows https://www.licor.com/env/support/EddyPro/topics/anemometer-tilt-correction.html#Doublerotationmethod
    """

    # Rotation 1
    rot1_mean = df.mean()

    theta = np.arctan(rot1_mean[v] / rot1_mean[u])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    u_temp = rot1_mean[u] * cos_theta + rot1_mean[v] * sin_theta
    v_temp = -1 * rot1_mean[u] * sin_theta + rot1_mean[v] * cos_theta
    
    # Rotation 2
    phi = np.arctan(rot1_mean[w] / u_temp)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    #df[f"{u}_rot"] = u_temp * cos_phi

    # Store rotations in original dataframe along with theta and phi (also returned in function)


    return df, theta, phi

#@jit
def pitch_correct(wind_vect, return_angle=False):

    """
    Assumes wind vector is [:,3] shape (need to add check on)
    Zeros out w-vector
    """
    u_mean = np.nanmean(wind_vect[:,0])
    w_mean = np.nanmean(wind_vect[:,2])
    denom = np.sqrt(u_mean**2 + w_mean**2)
    u_ele = u_mean / denom
    w_ele = w_mean / denom
    A_pitch = np.array([[u_ele, 0, -w_ele],
                        [0, 1, 0],
                        [w_ele, 0, u_ele]])

    # Apply rotation
    rotated = np.dot(wind_vect, A_pitch)

    return rotated

#@jit 
def yaw_correct(wind_vect, return_angle=False):
    
    """
    Zeros out v-vector
    """

    u_mean = np.nanmean(wind_vect[:,0])
    v_mean = np.nanmean(wind_vect[:,1])
    denom = np.sqrt(u_mean**2 + v_mean**2)
    u_ele = u_mean / denom
    v_ele = v_mean / denom
    A_yaw = np.array([[u_ele, -v_ele, 0],
                      [v_ele, u_ele, 0],
                      [0, 0, 1]])
    
    # Apply rotation
    rotated = np.dot(wind_vect, A_yaw)
    
    return rotated



def yaw_angle(u, v):
    """
    Returns yaw angle given two vectors: u and v
    """

    return np.arctan(u/v)

def pitch_angle(u, v, w):
    """
    Returns pitch angle given angles u, v, and w
    """

    return np.arctan(w/(u**2 + v**2 + w**2))

def _double_rotate_pd_rotations(df):
    """
    Function to rotate winds
    """

    wind_vect0 = df.values
    wind_vect1 = yaw_correct(wind_vect0)
    wind_vect2 = pitch_correct(wind_vect1)

    new_cols = [f'{c}_dr' for c in df.columns]

    return pd.DataFrame(wind_vect2, index=df.index, columns=new_cols)

def double_rotate_pd(df, time_freq, u_meas='Ux', v_meas='Uy', w_meas='Uz', ts_meas='t_sonic'):
    """
    Calculate double rotated velocities
    """
    #Split data into groups dictated by time_freq
    # did .groupby(pd.Grouper(freq='5T', closed='right', label='right'), group_keys=False)
    # in previous code... not sure why
    df_meas_grouped = df.groupby(pd.Grouper(freq='5T', closed='right', label='right'), group_keys=False)
    
    test_grouped = df_meas_grouped.apply(_double_rotate_pd_rotations)

    # Calculate covariances

    return test_grouped


