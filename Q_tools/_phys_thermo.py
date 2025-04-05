# _summary_
# ==================================================
import numpy as np
# ==================================================
# Const
Cp = 1004.5# []
Lv = 2.5e6 # [J/kg]
Rd = 287.  # [J/kg/K]
Rv = 461.5 # [J/kg/K]
e0 = 6.112 # [hPa]
g  = 9.81  # [m/s^2]

# ==================================================

def calc_SH(wth):
    return Cp * wth

def calc_LH(wqv):
    return Lv * wqv

def calc_th2temp(th, P):
    """Unit: K / hPa -> K"""
    return th * (P/1000.)**0.286

def calc_temp2th(temp, P):
    """Unit: K / hPa -> K"""
    return temp / (1000./P)**0.286

def calc_cceq(T):
    """Unit: K -> hPa"""
    return 6.112 * np.exp(Lv/Rv * (1/273 - 1/T))

def calc_anti_cceq(es):
    """Unit: hPa -> K"""
    return 1 / ((1/273.15) - (Rv/Lv) * np.log(es/6.112))

def calc_MSE(T, z, qv):
    """Unit: K, m, kg/kg -> J/kg"""
    return Cp * T + Lv * qv + g * z

def calc_th_v(th, qv, ql):
    return th * (1 + 0.608*qv - ql)

def calc_buoyancy(thv, thv_bar, thv_0):
    return g * (thv - thv_bar) / thv_0

del np