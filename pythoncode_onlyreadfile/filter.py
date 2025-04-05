########################################
############## filter.py ###############
######## Author: Wei-Jhih Chen #########
######### Update: 2022/11/08 ###########
########################################

''' Filter tools
--------------------
Kdp method:
--------------------
Kdp(i) = d(Phidp(i)) / 2dr

--------------------
Functions:
--------------------
- var_filter(filVar , var , filMin , filMax)
- Zdr_filter(Zdr , var , filNum)
- Kdp_filter(Phidp , rng , smNum)
'''

import numpy as np
import numpy.ma as ma

def var_filter(filVar , var , filMin , filMax):
    ''' General function for one variable to be masked by other variable 
    which exceed the range between the setting maximum and minimum.

    --------------------
    Args:
        filVar: numpy.ndarray
            the variable for "var" to be masked
        var: numpy.ndarray
            the variable masked by "filVar"
        filMin: float
            the lower bound of "filVar" allowed not to mask "var"
        filMax: float
            the higher bound of "filVar" allowed not to mask "var"
    
    --------------------
    Returns:
        var: numpy.ndarray (After be masked)
    '''
    if filMin is None and filMax is None:
        return var
    elif filMin is None:
        return ma.array(var , mask = filVar > filMax , copy = False)
    elif filMax is None:
        return ma.array(var , mask = filVar < filMin , copy = False)
    else:
        return ma.array(var , mask = (filVar < filMin) | (filVar > filMax) , copy = False)

def Zdr_filter(Zdr , var , filNum):
    ''' Use deviations of Zdr to filter other variables

    --------------------
    Args:
        Zdr: numpy.ndarray
            (Shape: azimuth * range)
        var: numpy.ndarray
            the variable masked by "Zdr" (Shape: azimuth * range)
        filNum: int (odd)
            the number for calculating the deviation of Zdr

    --------------------
    Returns:
        Zdr: numpy.ndarray (After be masked)
        var: numpy.ndarray (After be masked)
    '''
    num_azi , num_rng = Zdr.shape
    Zdr = Zdr.filled(np.nan)
    SmZdr = np.empty((num_azi , num_rng))
    SmZdr.fill(np.nan)
    for cnt_rng in range(num_rng - (filNum - 1)):
        SmZdr[: , cnt_rng + int((filNum - 1) / 2)] = np.mean(Zdr[: , cnt_rng : cnt_rng + (filNum - 1)] , axis = 1)
    Zdr_DV = Zdr - SmZdr
    Zdr_SdDV = np.nanstd(Zdr_DV)
    Zdr[np.isnan(SmZdr)] = np.nan
    var[np.isnan(SmZdr)] = np.nan
    Zdr[np.abs(Zdr_DV) > Zdr_SdDV] = np.nan
    var[np.abs(Zdr_DV) > Zdr_SdDV] = np.nan
    return Zdr , var

def Kdp_filter(Phidp , rng , smNum):
    ''' Calculate from Phidp to Kdp

    --------------------
    Args:
        Phidp: numpy.ndarray
            (Shape: azimuth * range)
        rng: numpy.ndarray
            radar range (Shape: range)
        smNum: int (odd)
            the number for smoothing Phidp

    --------------------
    Returns:
        Kdp: numpy.ndarray
    '''
    num_azi , num_rng = Phidp.shape
    Phidp = Phidp.filled(np.nan)
    SmPhidp = np.empty((num_azi , num_rng))
    Kdp = np.empty((num_azi , num_rng))
    SmPhidp.fill(np.nan)
    Kdp.fill(np.nan)
    for cnt_azi in range(num_azi):
        for cnt_rng in range(num_rng - 1):
            if Phidp[cnt_azi , cnt_rng + 1] - Phidp[cnt_azi , cnt_rng] < -180:
                Phidp[cnt_azi , cnt_rng + 1 : ] = Phidp[cnt_azi , cnt_rng + 1 : ] + 360
        for cnt_rng in range(num_rng - (smNum - 1)):
            SmPhidp[cnt_azi , cnt_rng + int((smNum - 1) / 2)] = np.nanmean(Phidp[cnt_azi , cnt_rng : cnt_rng + (smNum - 1)])
    for cnt_rng in range(num_rng - 1):
        Kdp[: , cnt_rng] = (SmPhidp[: , cnt_rng + 1] - SmPhidp[: , cnt_rng]) / (rng[cnt_rng + 1] - rng[cnt_rng]) / 2
    return Kdp