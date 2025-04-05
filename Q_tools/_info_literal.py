# path / spots information for 2025CDW
# ==================================================
import os
import numpy as np
# ==================================================

exps_dir  = "/data/mlcloud/mlpbl_2025/b12209017/2025CDW/taiwanvvm_tpe"
exps_name = sorted(os.listdir(exps_dir))
exps_path = [os.path.join(exps_dir, exp_name) for exp_name in exps_name]    

citybound_fpath = "/data/mlcloud/mlpbl_2025/b12209017/2025CDW/DATA/COUNTY_MOI_1130718.shp"
townbound_fpath = "/data/mlcloud/mlpbl_2025/b12209017/2025CDW/DATA/TOWN_MOI_1131028.shp"
topo_fpath  = "/data/mlcloud/mlpbl_2025/b12209017/2025CDW/taiwanvvm_tpe/tpe20150613nor/TOPO.nc"

alat, alon = np.loadtxt("/data/mlcloud/mlpbl_2025/b12209017/2025CDW/DATA_VVM/taiwanvvm_tpe.coords.txt", skiprows=1, unpack=True)
p_zc, m_zc = np.loadtxt("/data/mlcloud/mlpbl_2025/b12209017/2025CDW/DATA_VVM/fort.txt", skiprows=1, unpack=True, usecols=(1,2))

obs_spots = {
    'NTU': (121.539, 25.0145),
    'Xin': (121.525, 24.9595),
    'Tu': (121.445, 24.9735),
    'Wu': (121.55, 24.865)
}

# ==================================================

def genofname(ofname, subdir=None):
    """Generate output filename with automatic numbering to avoid overwriting\n
    ## DO NOT add /"""
    i = 0
    pwd_dir = os.getcwd() + "/"
    if subdir is not None:
        ofname  = subdir + "/" + ofname
    tmp  = ofname.split(".") # name . extension
    ofname = pwd_dir + ofname # -> absolute path
    
    while(os.path.exists(ofname)):
        i += 1
        ofname = pwd_dir + ".".join(tmp[:-1]) + f"({i})." + tmp[-1]    
    return ofname

del np