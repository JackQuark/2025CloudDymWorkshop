########################################
###### attenuation_correction.py #######
######## Author: Wei-Jhih Chen #########
######### Update: 2022/11/11 ###########
########################################

''' Calculate corrected values by the attenuation effect
--------------------
Correction method:
--------------------
Ah(i) = b1 * Kdp(i) ** b2
Zh_corr(i) = Zh(i) + 2 * sum(Ah(0) + Ah(1) + ... + Ah(i)) * delta_r
Adp(i) = d1 * Kdp(i) ** d2
Zdr_corr(i) = Zdr(i) + 2 * sum(Adp(0) + Adp(1) + ... + Adp(i)) * delta_r

--------------------
Functions:
--------------------
- attenuation_correction(Zh , Zdr , Kdp , dr , b1 , b2 , d1 , d2)
- attenuation_correction_C(Zh , Zdr , Kdp , dr) (for C band)
- attenuation_correction_X(Zh , Zdr , Kdp , dr) (for X band)
'''

import copy as cp
import numpy as np
from numpy.ma import masked_array as mama
from typing import Tuple

def attenuation_correction(Zh: np.ndarray , Zdr: np.ndarray , Kdp: np.ndarray , delta_r: float , 
                           b1: float , b2: float , d1: float , d2: float) -> Tuple[np.ndarray, np.ndarray]:
    ''' Attenuation correction by customized parameters: b1 , b2 , d1 , d2

    You can input parameters by yourself, or use attenuation_correction_C or
    attenuation_correction_X function to apply default parameters in each band, 
    and then you would get new values of Zh and Zdr after attenuation correction.

    Please refer to the correction method in the docstring of the module. 

    --------------------
    Args:
        Zh: numpy.ndarray
            (Shape: azimuth * range)
        Zdr: numpy.ndarray
            (Shape: azimuth * range)
        Kdp: numpy.ndarray
            (Shape: azimuth * range)
        delta_r: float
            resolution of range direction (Units: km)
        b1 , b2 , d1 , d2: float
            parameters of correction method (diff from band to band)

    --------------------
    Returns:
        Zh_AC: numpy.ndarray (After correction)
        Zdr_AC: numpy.ndarray (After correction)
    '''
    shp = Zh.shape
    num_rng = shp[-1]
    Zh = Zh.reshape((-1 , num_rng))
    Zdr = Zdr.reshape((-1 , num_rng))
    Kdp = Kdp.reshape((-1 , num_rng))
    num_ray = Zh.shape[0]
    Ah = np.zeros(num_ray)
    Adp = np.zeros(num_ray)
    Zh_AC = cp.copy(Zh)
    Zdr_AC = cp.copy(Zdr)
    
    Kdp = mama(Kdp , Kdp < 0)
    for cnt_rng in range(num_rng):
        Ah += b1 * Kdp[: , cnt_rng] ** b2
        Adp += d1 * Kdp[: , cnt_rng] ** d2
        Zh_AC[: , cnt_rng] += 2 * Ah * delta_r
        Zdr_AC[: , cnt_rng] += 2 * Adp * delta_r

    Zh_AC = Zh_AC.reshape(shp)
    Zdr_AC = Zdr_AC.reshape(shp)
    return Zh_AC , Zdr_AC

def attenuation_correction_C(Zh: np.ndarray , Zdr: np.ndarray , Kdp: np.ndarray , 
                             delta_r: float) -> Tuple[np.ndarray, np.ndarray]:
    ''' Attenuation correction of C band refer to Bringi et al. 1990 (B90).

    b1: 0.08
    b2: 1
    d1: b1 * 0.1125
    d2: 1

    --------------------
    Args:
        Zh: numpy.ndarray
        Zdr: numpy.ndarray
        Kdp: numpy.ndarray
        delta_r: float

        Please refer to the function "attenuation_correction".

    --------------------
    Returns:
        Zh: numpy.ndarray (After correction)
        Zdr: numpy.ndarray (After correction)
    '''
    b1 = 0.08
    b2 = 1
    d1 = 0.009
    d2 = 1
    return attenuation_correction(Zh , Zdr , Kdp , delta_r , b1 , b2 , d1 , d2)


def attenuation_correction_X(Zh: np.ndarray , Zdr: np.ndarray , Kdp: np.ndarray , 
                             delta_r: float) -> Tuple[np.ndarray, np.ndarray]:
    ''' Attenuation correction of X band refer to FURUNO WR2100.

    b1: 0.233
    b2: 1.02
    d1: 0.0298
    d2: 1.293

    --------------------
    Args:
        Zh: numpy.ndarray
        Zdr: numpy.ndarray
        Kdp: numpy.ndarray
        delta_r: float

        Please refer to the function "attenuation_correction".

    --------------------
    Returns:
        Zh: numpy.ndarray (After correction)
        Zdr: numpy.ndarray (After correction)
    '''
    b1 = 0.233
    b2 = 1.02
    d1 = 0.0298
    d2 = 1.293
    return attenuation_correction(Zh , Zdr , Kdp , delta_r , b1 , b2 , d1 , d2)