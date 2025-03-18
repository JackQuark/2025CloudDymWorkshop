# _summary_
# ==================================================
import sys
import os
import numpy as np
import xarray as xr

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

sys.path.append("/data/mlcloud/mlpbl_2025/b12209017/2025CDW/nctool")
import vvmds
# ==================================================
# 
__filedir__ = os.path.dirname(__file__)

exps_dir  = "/data/mlcloud/mlpbl_2025/b12209017/2025CDW/taiwanvvm_tpe"
exps_name = sorted(os.listdir(exps_dir))
exps_path = [os.path.join(exps_dir, exp_name) for exp_name in exps_name]    

citybound_fpath = "/data/mlcloud/mlpbl_2025/b12209017/2025CDW/DATA/COUNTY_MOI_1130718.shp"
townbound_fpath = "/data/mlcloud/mlpbl_2025/b12209017/2025CDW/DATA/TOWN_MOI_1131028.shp"

topo_fpath  = "/data/mlcloud/mlpbl_2025/b12209017/2025CDW/taiwanvvm_tpe/tpe20150613nor/TOPO.nc"
alat, alon  = np.loadtxt("/data/mlcloud/mlpbl_2025/b12209017/2025CDW/DATA_VVM/taiwanvvm_tpe.coords.txt", skiprows=1, unpack=True)
p_zc, m_zc  = np.loadtxt("/data/mlcloud/mlpbl_2025/b12209017/2025CDW/DATA_VVM/fort.txt", skiprows=1, unpack=True, usecols=(1,2))

obs_spots = {
    'NTU': (121.539, 25.0145),
    'Xin': (121.525, 24.9595),
    'Tu': (121.445, 24.9735),
    'Wu': (121.5682, 24.88368)
}

# ==================================================
# mpl global settings
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.linewidth'] = 1

# ==================================================

def nonlinspace(start, end, intervals, bounds):
    N   = len(intervals)
    if N != len(bounds) + 1:
        raise ValueError("len(intervals) should be len(bounds) + 1")
   
    tmp = np.concatenate(([start], bounds, [end]))
    segments = [np.arange(tmp[i], tmp[i+1], intervals[i]) for i in range(N-1)]
    segments.append(np.arange(tmp[-2], tmp[-1]+intervals[-1]/2, intervals[-1]))
    return np.concatenate(segments)

def sel_region(central_lon: float, central_lat: float, istep: int = 256):    
    cidx_lon = np.argmin(np.abs(alon - central_lon))
    cidx_lat = np.argmin(np.abs(alat - central_lat))
    isel_lon = slice(cidx_lon-istep//2, cidx_lon+istep//2+1)
    isel_lat = slice(cidx_lat-istep//2, cidx_lat+istep//2+1)
    return isel_lon, isel_lat

def _plot_boundaries(ax, boundaries="city"):
    if boundaries == "city":
        gdf = gpd.read_file(citybound_fpath)
    else:
        gdf = gpd.read_file(townbound_fpath)
    gdf.plot(ax=ax, lw=.5, fc='none', ec='dimgray', zorder=1)

def plot_topo(topo: np.ndarray, lat, lon, ax=None, boundaries="city"):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_facecolor('whitesmoke')
        _plot_boundaries(ax, boundaries=boundaries)
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        ax.set_title("tpe{yyyymmdd}nor,\nTaipei Basin Topography", fontsize=10, loc='left')
    
    _topo = np.where(topo == 0, np.nan, topo)
    levels= np.arange(np.nanmin(_topo)-1, np.nanmax(_topo)+1, dtype=int)
    cmap  = plt.get_cmap('terrain')
    norm  = mplc.BoundaryNorm(levels, cmap.N)
    
    cf = ax.contourf(lon, lat, _topo, levels=levels, cmap=cmap,
                     alpha=0.6, zorder=1)
    
    
    cbar = plt.colorbar(cf, ax=ax, ticks=levels[1::2]-.5, shrink=.8, pad=.055)
    cbar.set_ticklabels([f"{i:.0f}" for i in m_zc[levels[1::2]]])
    cbar.set_label("topo [m]")
    cbar.ax.yaxis.set_label_position('left')
    
    if fig is not None:
        return fig, ax
    return ax


def plot_wind(u, v, lon, lat, skip: int = 5, boundaries="city", **kwargs):
    if 'topo' in kwargs:
        fig, ax = plot_topo(kwargs['topo'], lat, lon, boundaries=boundaries)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_facecolor('whitesmoke')
        _plot_boundaries(ax, boundaries=boundaries)
    
    _skip = slice(None, None, skip)
    Q = ax.quiver(lon[_skip], lat[_skip], u[_skip, _skip], v[_skip, _skip], 
                  units = 'width', 
                  color='k', alpha=0.6, zorder=2)
    qk = ax.quiverkey(Q, .95, 1.05, 5, r'$5m/s$', coordinates='axes',
                      labelpos='E', fontproperties={'size': 10})
    
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    return fig, ax

def plot_obsspot(ax, legend: bool):
    # for loc. checking     
    cs = ['lime', 'orange', 'violet', 'r']
    for key, loc in obs_spots.items():
        lon_obs, lat_obs = loc
        ax.plot(lon_obs, lat_obs, 'o', c=cs.pop(0), ms=3, label=key)
    if legend:        
        ax.legend(loc='lower right', facecolor='whitesmoke', 
                bbox_to_anchor=(1, 1), ncol=2)

# ==================================================
    
def main():
    # for region select    
    istep = 128
    central_lon, central_lat = obs_spots['NTU']
    isel_lon, isel_lat = sel_region(central_lon, central_lat, istep)
    
    from functools import partial

    def _preprocess(x, lon_bnds, lat_bnds):
        return x.isel(lon=lon_bnds, lat=lat_bnds)

    partial_func = partial(_preprocess, lon_bnds=isel_lon, lat_bnds=isel_lat)

    # with xr.open_dataset(topo_fpath) as ds_topo:
    #     ds_topo = partial_func(ds_topo)
    #     topo = ds_topo.topo.values.astype(int)
        
    #     lat  = ds_topo.lat
    #     lon  = ds_topo.lon

    # fig, ax = plot_topo(topo, lat, lon)
    # plot_obsspot(ax, legend=True)
    
    # return 

    ds = vvmds.VVMDataset(exps_path[0])    
    # return
    with ds.open_ncdataset('dym', step=None, preprocess=partial_func) as ds_dym:
        u: xr.DataArray = ds_dym.u
        v: xr.DataArray = ds_dym.v

    for i in range(1, len(exps_path)):
        

    return 
    # sel. the topo+1 level
    # ambient_idx = np.where(np.logical_and(topo > 0, topo <= 10), topo, 0)    
    # u = u[ambient_idx, np.arange(lat.size)[:, None], np.arange(lon.size)[:]]p
    v = v[ambient_idx, np.arange(lat.size)[:, None], np.arange(lon.size)[:]]
    u = np.where(ambient_idx == 0, np.nan, u)

    fig, ax = plot_wind(u, v, lon, lat, skip=7, boundaries="town", topo=topo)
    plot_obsspot(ax, legend=True)

# ==================================================
from time import perf_counter

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))