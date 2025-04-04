# _summary_
# ==================================================
import sys
import os
import numpy as np
import xarray as xr

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
# plt.rcParams['axes.linewidth'] = 1.

# ==================================================
# const
Cp = 1004.5
Lv = 2.5e6

def calc_SH(wth):
    return Cp * wth

def calc_LH(wqv):
    return Lv * wqv

def calc_temp(th, P):
    return th * (P/100000.)**0.286

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
    gdf.plot(ax=ax, ls=':', lw=.5, fc='none', ec='dimgray', zorder=1)

def plot_topo(topo: np.ndarray, lat, lon, ax=None, boundaries="city", cbar=True, fill=True):
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
    levels= np.arange(np.nanmin(_topo)-1, np.nanmax(_topo)+1, 1, dtype=int)
    
    ax.contour(lon, lat, _topo, levels=levels[1::4], colors='k', alpha=0.5,
               linewidths=0.5, zorder=5)
    
    if fill:
        cmap  = plt.get_cmap('binary')

        CS = ax.contourf(lon, lat, _topo, levels=levels, cmap=cmap,
                        alpha=0.6, zorder=1)
    
        if cbar:
            axins = inset_axes(
                ax,
                width="2.5%",
                height="45%",
                loc="upper left",
                bbox_to_anchor=(1.05, 0., 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=.5,
            )
            
            cbar = plt.colorbar(CS, cax=axins, ticks=levels[1::4]-.5, shrink=.3, pad=.025)
            cbar.set_ticklabels([f"{i:.0f}" for i in m_zc[levels[1::4]]])
            cbar.set_label("topo [m]")
            cbar.ax.yaxis.set_label_position('left')

    return ax

def plot_sprec(sprec, lat, lon, topo=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.set_facecolor('whitesmoke')
        _plot_boundaries(ax, boundaries="city")
    if topo is not None:
        plot_topo(topo, lat, lon, ax=ax)
    
    cmvalue = nonlinspace(2.5, 80, (2.5, 5, 10), (15, 50))
    # cmvalue = nonlinspace(1, 30, (1, 2.5, 5), (10, 20))
    cmcolor = ['#a0fffa','#00cdff','#0096ff',
               '#0069ff','#329600','#32ff00',
               '#ffff00','#ffc800','#ff9600',
               '#ff0000','#c80000','#a00000',
               '#96009b','#c800d2','#ff00f5',]
    
    cmap = mplc.ListedColormap(cmcolor).with_extremes(under='none', over='#ffc8ff')
    norm = mplc.BoundaryNorm(cmvalue, cmap.N)
    
    CS = ax.pcolormesh(lon, lat, sprec, cmap=cmap, norm=norm, zorder=3)
    
    axins = inset_axes(
        ax,
        width="2.5%",
        height="45%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=.5,
    )
    
    cbar = plt.colorbar(CS, cax=axins, shrink=0.3, extend='both')
    cbar.set_ticks(cmvalue[::3])
    cbar.set_label("sprec [mm]")
    cbar.ax.yaxis.set_label_position('left')
    
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_aspect('equal')
    return ax

def plot_dym(u, v, lon, lat, skip: int = 5, boundaries="city", ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
    if 'topo' in kwargs:
        plot_topo(kwargs['topo'], lat, lon, boundaries=boundaries, ax=ax, cbar=False)

    # _plot_boundaries(ax, boundaries=boundaries)
    # ax.set_facecolor('whitesmoke')
    ax.set_aspect('equal')
    _skip = slice(None, None, skip)
    Q = ax.quiver(lon[_skip], lat[_skip], u[_skip, _skip], v[_skip, _skip], 
                  units='width', scale=120, 
                  color='k', alpha=0.6, zorder=10)
    qk = ax.quiverkey(Q, .95, 1.05, 5, r'$5m/s$', coordinates='axes',
                      labelpos='E', fontproperties={'size': 10})
    
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    return ax

# ==================================================

def main():
    # region select
    region_range = 96
    central_lon, central_lat = obs_spots['Wu']
    isel_lon, isel_lat = sel_region(central_lon, central_lat, region_range)    
    
    from functools import partial

    def _preprocess(x, lon_bnds, lat_bnds):
        return x.isel(lon=lon_bnds, lat=lat_bnds)

    partial_func = partial(_preprocess, lon_bnds=isel_lon, lat_bnds=isel_lat)

    with xr.open_dataset(topo_fpath) as ds_topo:
        ds_topo = partial_func(ds_topo)
        topo = ds_topo.variables['topo'].squeeze().to_numpy().astype(int)
        lat  = ds_topo.variables['lat']
        lon  = ds_topo.variables['lon']       

    ax = plot_topo(topo, lat, lon, boundaries="city", cbar=True, fill=True)
    isel_lon, isel_lat = sel_region(central_lon, central_lat, 1)
    ax.plot(alon[isel_lon], alat[isel_lat], 'rx', ms=10, mew=2)

    sel_case  = "tpe20110802nor"
    isel_case = exps_name.index(sel_case)
    exp_path  = exps_path[isel_case]
    sel_step  = slice(60, 109)
    
    ds = vvmds.VVMDataset(exp_path)
    with ds.open_ncdataset('surf', step=sel_step, preprocess=partial_func) as ds_surf:
        hourly_sprec = ds_surf.variables['sprec'] * 600
        wth = ds_surf.variables['wth'].squeeze() * Cp
        wqv = ds_surf.variables['wqv'].squeeze() * Lv
    with ds.open_ncdataset('thermo', step=sel_step, preprocess=partial_func) as ds_thermo:
        th = ds_thermo['th'].squeeze()
        qv = ds_thermo['qv'].squeeze()
        
    temp = calc_temp(th, p_zc[None, :])

    print("data loading complete.")

    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax_time  = np.arange(hourly_sprec.shape[0])
    
    # axs[0].plot(temp[:, topo], 'k-', lw=1)
    # axtw0 = axs[0].twinx()
    
    axs[0].plot(temp[:, topo], 'k-', lw=1)
    axs[0].set_title("Temperature [K]")
    axs[1].plot(wth, color='tab:red')
    axs[1].plot(wqv, color='tab:blue')
    axs[1].set_title("SH / LH [w/$m^2$]")
    axtime = np.arange(hourly_sprec.shape[0])
    axs[1].set_xlabel("Time [UTC+8]")
    for ax in axs:
        ax.set_xticks(axtime[::6])
        ax.set_xticklabels(np.arange(10, 19, 1), rotation=45)     
        ax.set_xlim(0, hourly_sprec.shape[0]-1)
        
    # axtw0 = axs[0].twinx()
    # axtw0.plot(ax_time, ambient_qv, color='tab:red')
    # axs[0].set_ylabel("Temperature [K]")
    # axtw0.set_ylabel("RH [%]", color='tab:red')
    
    return 
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    
    sel_lev = slice(topo, 20)
    L_th, = axs[0].plot(temp[0, sel_lev], m_zc[sel_lev], 'k-', lw=1)
    L_qv, = axs[1].plot(qv[0, sel_lev], m_zc[sel_lev], 'k-', lw=1)
    
    for ax in axs:
        ax.set_ylim(m_zc[sel_lev][0], m_zc[sel_lev][-1])
    
    return 
    def update_plot(i):
        print(f"{i}", end="\r")
        L_th.set_data(th[i, sel_lev], m_zc[sel_lev])
        L_qv.set_data(qv[i, sel_lev], m_zc[sel_lev])
        
    anim = FuncAnimation(fig, update_plot, frames=range(hourly_sprec.shape[0]))
    writer = FFMpegWriter(fps=10)
    anim.save("./anim.mp4", writer=writer)
    
    return
    
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

# ==================================================
from time import perf_counter

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))