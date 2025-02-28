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

sprec_dir = "/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/DATA_VVM/C.Surface"
sprec_fname = sorted(os.listdir(sprec_dir))
sprec_fpath = [os.path.join(sprec_dir, fname) for fname in sprec_fname]
topo_fpath  = "/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/DATA_VVM/NorthernTW_TOPO.nc"
citybound_fpath = "/data/mlcloud/mlpbl_2025/b12209017/WCD_2025/DATA/COUNTY_MOI_1130718.shp"

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

def plot_sprec(sprec, lat, lon):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf = gpd.read_file(citybound_fpath)
    gdf.plot(ax=ax, facecolor='none', edgecolor='black', zorder=5)
    
    cmvalue = nonlinspace(2.5, 80, (2.5, 5, 10), (15, 50))
    cmcolor = ['#a0fffa','#00cdff','#0096ff',
               '#0069ff','#329600','#32ff00',
               '#ffff00','#ffc800','#ff9600',
               '#ff0000','#c80000','#a00000',
               '#96009b','#c800d2','#ff00f5',]
    cmap = mplc.ListedColormap(cmcolor).with_extremes(under='#FFFFFF', over='#ffc8ff')
    norm = mplc.BoundaryNorm(cmvalue, cmap.N)
    
    CS = ax.pcolormesh(lon, lat, sprec, cmap=cmap, norm=norm, zorder=0)
    cbar = plt.colorbar(CS, ax=ax, shrink=0.85, pad=0.05, extend='both')
    cbar.set_ticks(cmvalue)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    return fig, ax

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

def plot_count(count, lat, lon):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf = gpd.read_file(citybound_fpath)
    gdf.plot(ax=ax, facecolor='none', edgecolor='black', zorder=5)
    
    cmap = plt.get_cmap('Reds')
    norm = mplc.BoundaryNorm(np.arange(0, np.nanmax(count)+1), cmap.N)
    
    CS = ax.pcolormesh(lon, lat, count, zorder=0, cmap=cmap, norm=norm)
    cbar = plt.colorbar(CS, ax=ax, ticks=norm.boundaries, shrink=0.85, pad=0.05)
    cbar.set_label("times [#]")
    
    hotpot_idxes = np.where(count>=9)
    ax.plot(lon[hotpot_idxes[1]], lat[hotpot_idxes[0]], 'o', c='cyan', ms=3, zorder=10, label=">=10 times")
    
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    ax.set_title("tpe{yyyymmdd}nor,\nHeavy Rain Heatmap (>=15mm/hr)", fontsize=10, loc='left')
    ax.legend(loc='upper right')
    return fig, ax

# ==================================================
    
def main():
    # for region select
    idx_start = 50
    idx_step  = 196
    idxsl_lon   = slice(idx_start, idx_start+idx_step)
    idxsl_lat   = slice(idx_start, idx_start+idx_step)
    
    NFile = 40
    NStep = 145
    # hourly
    AllThesprec = np.empty((NFile, 24, idx_step, idx_step), dtype=np.float32)
    
    with xr.open_dataset(topo_fpath) as ds:
        lat  = ds.variables['lat'].values[idxsl_lat]
        lon  = ds.variables['lon'].values[idxsl_lon]
        topo = ds.variables['topo'].values[idxsl_lat, idxsl_lon]
        topo = np.where(topo==0, np.nan, topo)
    
    # Loading hourly sprec data
    for iFile in range(NFile):
        with xr.open_mfdataset(sprec_fpath[iFile]) as ds:
            tmp = ds.variables['sprec'].values[1:, idxsl_lat, idxsl_lon].reshape(-1, 24, lat.size, lon.size)
            AllThesprec[iFile, ...] = np.nansum(tmp, axis=0)
    del tmp
    AllThesprec *= 600
    print("finish data loading...")
    
    # count_heavyrain = np.where(AllThesprec>15, 1, 0)
    # count_heavyrain = np.nansum(count_heavyrain, axis=(0, 1))    
    # fig, ax = plot_count(count_heavyrain, lat, lon)
    # fig.savefig(os.path.join(__filedir__, "sprec/HeavyRainHeatmap.png"))
    
    # for iday in range(NFile):
    #     tot_sprec = np.nansum(AllThesprec[iday, :], axis=0)
    #     fig, ax = plot_sprec(tot_sprec, lat, lon)
    #     ax.set_title(f"{sprec_fname[iday].split('.')[0]}\ndaily total sprec (mm/day)", loc='left')
    #     fig.savefig(os.path.join(__filedir__, f"sprec/daily_totsprec/{sprec_fname[iday].split('.')[0]}.png"))
    #     plt.close(fig)

    # fig, ax = plot_sprec(np.nanmean(AllThesprec.sum(axis=1), axis=(0)), lat, lon)
    # ax.set_title("tpe{yyyymmdd}nor,\nmean of daily sprec (mm/day)", loc='left')
    # fig.savefig(os.path.join(__filedir__, "sprec/daily_totsprec/tpe.nor-40d_mean.png"))
    
    # topo
    # fig, ax = plot_topo(topo[idxsl_lat, idxsl_lon], lat[idxsl_lat], lon[idxsl_lon])
    # fig.savefig(os.path.join(__filedir__, "sprec/TaipeiBasin_topo.png"))
    
# ==================================================
from time import perf_counter

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))