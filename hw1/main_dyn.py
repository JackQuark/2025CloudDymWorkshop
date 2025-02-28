# _summary_
# ==================================================
import sys
import os
import numpy as np
import xarray as xr
import netCDF4 as nc

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

from shapely.geometry import box
# ==================================================
# 
__filedir__ = os.path.dirname(__file__)

topo_fpath  = "/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/DATA_VVM/NorthernTW_TOPO.nc"
citybound_fpath = "/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/DATA/COUNTY_MOI_1130718.shp"

ncPrefix_LandSurface   = "C.LandSurface"
ncPrefix_Surface       = "C.Surface"
ncPrefix_Dynamic       = "L.Dynamic"
ncPrefix_Thermodynamic = "L.Thermodynamic"
ncPrefix_Radiation     = "L.Radiation"
ncPrefixes = [ncPrefix_LandSurface, ncPrefix_Surface, ncPrefix_Dynamic, ncPrefix_Thermodynamic, ncPrefix_Radiation]

exps_dir  = "/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/taiwanvvm_tpe"
exps_name = sorted(os.listdir(exps_dir))
exps_path = [os.path.join(exps_dir, exp_name) for exp_name in exps_name]    

# ==================================================
# mpl global settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.

# ==================================================

def nonlinspace(start, end, intervals, bounds):
    N   = len(intervals)
    if N != len(bounds) + 1:
        raise ValueError("len(intervals) should be len(bounds) + 1")
   
    tmp = np.concatenate(([start], bounds, [end]))
    segments = [np.arange(tmp[i], tmp[i+1], intervals[i]) for i in range(N-1)]
    segments.append(np.arange(tmp[-2], tmp[-1]+intervals[-1]/2, intervals[-1]))
    return np.concatenate(segments)

def plot_topo(topo, lat, lon):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf = gpd.read_file(citybound_fpath)
    gdf.plot(ax=ax, facecolor='none', edgecolor='black', zorder=5)
    
    cmap = plt.get_cmap('terrain')
    norm = mplc.BoundaryNorm(np.arange(1, np.nanmax(topo)+1), cmap.N)
    
    CS = ax.pcolormesh(lon, lat, topo, cmap=cmap,norm=norm, zorder=0)
    cbar = plt.colorbar(CS, ax=ax, ticks=norm.boundaries[::2], shrink=0.85, pad=0.05)
    cbar.set_label("topo")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    ax.set_title("tpe{yyyymmdd}nor,\nTaipei Basin Topography", fontsize=10, loc='left')
    return fig, ax

def plot_windfield(U, V, ws, lat, lon):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf = gpd.read_file(citybound_fpath)
    gdf.plot(ax=ax, facecolor='none', edgecolor='black', zorder=5)
    
    cmap = plt.get_cmap('viridis')
    norm = mplc.BoundaryNorm(np.arange(0, np.floor(ws.max())+1, 0.5), cmap.N)
    
    CS = ax.quiver(lon, lat, U, V, ws, scale=70, cmap=cmap, norm=norm, zorder=2)
    cbar = plt.colorbar(CS, ax=ax, shrink=0.85, pad=0.05)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    return fig, ax


def VVM_archive_listdir(exppath):
    archive_path = os.join(exppath, archive_path)
    
    nc_names = sorted(os.listdir(archive_path))
    nc_paths = [os.path.join(archive_path, name) for name in nc_names]

class VVMDataset(object):
    def __init__(self, exppath: str):
        self.exp_name = exppath.split('/')[-1]
        self.exp_path = exppath
        self.archive_path = os.path.join(exppath, 'archive')
        self.nc_names = self.getname_nclistdir(self.archive_path)
        self.nc_paths = self.getpath_nclistdir(self.archive_path)        
        
    def _count_nctype(self):

        for prefix in ncPrefix
            nc_files = [f for f in self.nc_names if f.startswith(self.exp_name + '.' + prefix)]
            if len(nc_files) == 0:
                continue # skip if no this type
            else:
                self.AllTheFiles[prefix] = nc_files
                self.AllTheTypes.append(prefix)
        
            
        
        
    @staticmethod
    def getname_nclistdir(archive_path):
        return sorted(os.listdir(archive_path))
    
    @staticmethod
    def getpath_nclistdir(archive_path):
        return [os.path.join(archive_path, name) for name in VVMDataset.getname_nclistdir(archive_path)]

    


# ==================================================
    
def main():
    # for region select    
    levs = 70
    Nsteps = 145
    idx_lonstart = 123
    idx_latstart = 85
    idx_step  = 64
    idxsl_lon   = slice(idx_lonstart, idx_lonstart+idx_step)
    idxsl_lat   = slice(idx_latstart, idx_latstart+idx_step)
    
    AllTheu = np.empty((Nsteps, levs, idx_step, idx_step))
    AllThev = np.empty((Nsteps, levs, idx_step, idx_step))

    with xr.open_dataset(topo_fpath) as ds:
        lat  = ds.variables['lat'].values[idxsl_lat]
        lon  = ds.variables['lon'].values[idxsl_lon]
        topo = ds.variables['topo'].values[idxsl_lat, idxsl_lon]
        topo = np.where(topo==0, np.nan, topo)
    
    iExp = 0
    vvmds = VVMDataset(exps_path[iExp])
    
    
    return 
    AllThews = np.sqrt(AllTheu**2 + AllThev**2)
    
    selected_time = [36, 72, 108, 144]
    selected_time = np.arange(36, 145, 12)
    
    wsstep = 3
    
    
    for t_idx in selected_time:
        fig, ax = plot_windfield(AllTheu[0, t_idx, ::wsstep, ::wsstep], 
                                AllThev[0, t_idx, ::wsstep, ::wsstep], 
                                AllThews[0, t_idx, ::wsstep, ::wsstep], 
                                lat[::wsstep], lon[::wsstep])
        ax.set_title(f"{uvwind_fname[0]},\nTime: {int(t_idx/6):02d}:00",
                     loc='left', fontsize=10) 
    
    return 
    
    for i, fpath in enumerate(uvwind_fpath):
        for t_idx in selected_time:
            fig, ax = plot_windfield(AllTheu[0, t_idx, ::wsstep, ::wsstep], 
                                    AllThev[0, t_idx, ::wsstep, ::wsstep], 
                                    AllThews[0, t_idx, ::wsstep, ::wsstep], 
                                    lat[::wsstep], lon[::wsstep])

            ax.set_title(f"{uvwind_fname[i]},\nTime: {int(t_idx/6):02d}:00",
                        loc='left', fontsize=10)
            fig.savefig(os.path.join(__filedir__, f"sprec/wind/{uvwind_fname[i]}_{int(t_idx/6):02d}.png"), dpi=300)
    

# ==================================================
from time import perf_counter

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))