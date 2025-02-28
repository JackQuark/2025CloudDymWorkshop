# _summary_
# ==================================================
import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd

from scipy.ndimage import binary_erosion

import vvmds as Q
# ==================================================
__filedir__ = os.path.dirname(__file__)
# ==================================================

def main():
    exps_dir  = "/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/taiwanvvm_tpe"
    exps_name = sorted(os.listdir(exps_dir))
    exps_path = [os.path.join(exps_dir, exp_name) for exp_name in exps_name]    

    vvmds = Q.VVMDataset(exps_path[0])
    print(vvmds)

def test():
    exps_dir  = "/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/taiwanvvm_tpe"
    exps_name = sorted(os.listdir(exps_dir))
    exps_path = [os.path.join(exps_dir, exp_name) for exp_name in exps_name]    

    # select a region
    lev        = 70
    Nstep      = 145
    ilat, ilon = 672, 480
    idxrange   = 256
    lon_bound  = slice(ilon, ilon+idxrange)
    lat_bound  = slice(ilat, ilat+idxrange)

    fort_df = pd.read_csv("/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/DATA_VVM/fort.csv", index_col=0,
                          names=['rho', 'th', 'p', 'pi', 'qv'], skiprows=1)
    # selc_lev_idx = [find_closest_idx(fort_df['p'], p) for p in [85000, 50000, 20000]]
    
    
    # selected_fname = ['tpe20090707nor',
    #                   'tpe20110616nor',
    #                   'tpe20140525nor',
    #                   'tpe20150613nor']
    # selected_fpath = [os.path.join(exps_dir, fname) for fname in selected_fname]
    
    # # for exp... in 
    # for name, path in zip(selected_fname, selected_fpath):
    #     archive_path = os.path.join(path, 'archive')
    #     allthefiles = VVMDataset.classify_archive_files(archive_path)         
    #     tmp_empty = np.empty((Nstep, len(selc_lev_idx), idxrange, idxrange), dtype=np.float32)
                 
    #     with xr.Dataset(
    #         {
    #             "u": xr.Variable(['time', 'lev', 'lat', 'lon'], tmp_empty), 
    #             "v": xr.Variable(['time', 'lev', 'lat', 'lon'], tmp_empty)
    #         }
    #     ) as nds:
            
    #         for istep in range(Nstep):
    #             print(f"Loading {name} step {istep}", end='\r')
    #             with xr.open_dataset(os.path.join(archive_path, allthefiles[ncPrefix_Dynamic][istep])) as ds_dyn: 
    #                 nds.variables['u'][istep] = ds_dyn.variables['u'].values[0, selc_lev_idx, lat_bound, lon_bound]    
    #                 nds.variables['v'][istep] = ds_dyn.variables['v'].values[0, selc_lev_idx, lat_bound, lon_bound]
            
    #         nds.to_netcdf(os.path.join("/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/DATA_VVM/L.Dynamic", name+'.'+ncPrefix_Dynamic+'.nc'),
    #                       encoding={
    #                           "u": {"dtype": "float32"},
    #                           "v": {"dtype": "float32"}
    #                       })
            
# ==================================================
from time import perf_counter

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))