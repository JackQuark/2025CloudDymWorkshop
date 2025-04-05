# _summary_
# ==================================================
import sys
import os
import numpy     as np
import xarray    as xr
import netCDF4   as nc

import matplotlib.pyplot  as plt
from matplotlib.axes      import Axes
from matplotlib.gridspec  import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.append("/data/mlcloud/mlpbl_2025/b12209017/2025CDW")
from Q_tools import *
# ==================================================

def sel_region(central_lon, central_lat, istep) -> tuple[int, int] | tuple[slice, slice]:
    cidx_lon = np.argmin(np.abs(alon - central_lon))
    cidx_lat = np.argmin(np.abs(alat - central_lat))
    if istep == 1:
        return cidx_lon, cidx_lat
    isel_lon = slice(cidx_lon-istep//2, cidx_lon+istep//2+1)
    isel_lat = slice(cidx_lat-istep//2, cidx_lat+istep//2+1)
    return isel_lon, isel_lat

def loc_to_idx(lon, lat, lon_bnds=None, lat_bnds=None):
    if lon_bnds is None:
        lon_bnds = slice(None)
    if lat_bnds is None:
        lat_bnds = slice(None)
    return np.argmin(np.abs(alon[lon_bnds] - lon)), np.argmin(np.abs(alat[lat_bnds] - lat))

def idx_to_loc(idx_lon, idx_lat, lon_bnds=None, lat_bnds=None):
    if lon_bnds is None:
        lon_bnds = slice(None)
    if lat_bnds is None:
        lat_bnds = slice(None)
    return alon[lon_bnds][idx_lon], alat[lat_bnds][idx_lat]

def sel_bresenham_line(x1, y1, x2, y2):
    """use index pls."""
    dots = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        dots.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x1 = x1 + sx
        if e2 < dx:
            err = err + dx
            y1 = y1 + sy

    return np.array(dots)

def show_sel_spot(topo, lon, lat, spotsx, spotsy):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.set_aspect('equal')
    plot_boundaries(ax, "city")
    plot_topo(ax, topo, lon, lat, fill=True, cbar=True)
    ax.plot(spotsx, spotsy, '--', c='r')
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    return fig, ax

def uvwind_projection(u, v, phi):
    """give phi of the line you want to project u and v on."""
    return u*np.cos(phi) + v*np.sin(phi)
    
# ===============================
from functools import partial
def _preprocess(x, lon_bnds, lat_bnds, lev_bnds=None):
    if not isinstance(lon_bnds, (int, slice)) or not isinstance(lat_bnds, (int, slice)):
        raise ValueError("lon_bnds and lat_bnds should be int or slice")
    if lev_bnds is None:
        return x.isel(lon=lon_bnds, lat=lat_bnds)
    else: 
        return x.isel(lon=lon_bnds, lat=lat_bnds, lev=lev_bnds)

def main():
    sel_case_name = "tpe20110802nor"
    sel_case_path = exps_path[exps_name.index(sel_case_name)]
    ds_case       = VVMDataset(sel_case_path)
    istep = 96
    isel_lon, isel_lat = sel_region(*obs_spots["Wu"], istep)
    
    # for region select
    partial_func = partial(_preprocess, lon_bnds=isel_lon, lat_bnds=isel_lat)
    # for section select
    tot_tstep = np.arange(ds_case.Nsteps)
    sel_tstep = slice(60, 108)

    with xr.open_dataset(topo_fpath) as ds:
        ds   = ds.isel(lon=isel_lon, lat=isel_lat)
        lat  = ds.lat
        lon  = ds.lon
        topo = ds.variables['topo'].astype(int)
    
    # print(loc_to_idx(121.55, 24.865, isel_lon, isel_lat))
    # print(idx_to_loc(*loc_to_idx(121.55, 24.865, isel_lon, isel_lat), isel_lon, isel_lat))
    # res = sel_bresenham_line(19, 56, 59, 56) # horizonal section
    # res = sel_bresenham_line(44, 40, 44, 80) # vertical section
    res = sel_bresenham_line(57, 30, 35, 75)
    # fig, ax = show_sel_spot(topo, lon, lat, *idx_to_loc(res[:, 0], res[:, 1], isel_lon, isel_lat))
    # ax.plot(*idx_to_loc(*loc_to_idx(121.55, 24.865, isel_lon, isel_lat), isel_lon, isel_lat), 'bo')
    # fig.savefig("section4.png")
    # return 
    phi = np.arctan2(res[-1, 1] - res[0, 1], res[-1, 0] - res[0, 0])
    
    _topo = np.diag(topo[res[:, 1], res[:, 0]])
    with ds_case.open_ncdataset("surf", step=sel_tstep, preprocess=partial_func) as ds:
        sprec = ds.sprec.isel(lon=xr.DataArray(res[:, 0]), lat=xr.DataArray(res[:, 1])) * 600

    isel_lev  = slice(None, 55)
    partial_func = partial(_preprocess, lon_bnds=isel_lon, lat_bnds=isel_lat, lev_bnds=isel_lev)
    with ds_case.open_ncdataset("dym", step=sel_tstep, preprocess=partial_func) as ds:
        u = ds.u.isel(lon=xr.DataArray(res[:, 0]), lat=xr.DataArray(res[:, 1]))
        v = ds.v.isel(lon=xr.DataArray(res[:, 0]), lat=xr.DataArray(res[:, 1]))
        V_h = uvwind_projection(u, v, phi)
        w = ds.w.isel(lon=xr.DataArray(res[:, 0]), lat=xr.DataArray(res[:, 1]))
    
    with ds_case.open_ncdataset("thermo", step=sel_tstep, preprocess=partial_func) as ds:
        th = ds.th.isel(lon=xr.DataArray(res[:, 0]), lat=xr.DataArray(res[:, 1]))
        qv = ds.qv.isel(lon=xr.DataArray(res[:, 0]), lat=xr.DataArray(res[:, 1]))
    
    with ds_case.open_ncdataset("radar", step=sel_tstep, preprocess=partial_func) as ds:
        ze = ds.ze.isel(lon=xr.DataArray(res[:, 0]), lat=xr.DataArray(res[:, 1]))
    del ds    
    print("complete data loading")
    
    _m_zc = m_zc[isel_lev]
    _p_zc = p_zc[isel_lev]
    Temp = calc_th2temp(th, _p_zc[None, :, None]/100)
    MSE = calc_MSE(Temp, _m_zc[None, :, None], qv)

    fig = plt.figure(figsize=(8, 6))
    axs: list[Axes] = []
    gs  = GridSpec(4, 4, figure=fig)
    axs.append(fig.add_subplot(gs[:3, :]))
    axs.append(fig.add_subplot(gs[3, :]))
    fig.subplots_adjust(hspace=.3)

    _xskip = slice(None, None, 2)
    _zskip = slice(None, None, 3)
    _t    = 0
    _X    = np.arange(res.shape[0])
    Q     = axs[0].quiver(_X[_xskip], _m_zc[_zskip], 
                          V_h.isel(time=_t, lev=_zskip, dim_0=_xskip), 
                          w.isel(time=_t, lev=_zskip, dim_0=_xskip), 
                          color='k', scale=200)
    axs[0].quiverkey(Q, 1.075, 0.95, 10, r'$10m/s$', coordinates='axes',
                     labelpos='E', fontproperties={'size': 10})
    
    axs[0].bar(_X, _m_zc[_topo], width=1, fc='dimgray')
    axs[0].set_ylim(0, _m_zc.max()+300)
    levels = np.linspace(0, 100, 21)
    norm   = plt.Normalize(0, 100)
    CS   = axs[0].pcolormesh(_X, _m_zc, ze.isel(time=_t), cmap='turbo', norm=norm, alpha=0.6, zorder=0)
    cbar = fig.colorbar(CS, ax=axs, shrink=0.75)
    cbar.set_ticks(levels[::2])
    cbar.set_label(r"dBZ [$mm^6/m^3$]")
    cbar.ax.yaxis.set_label_position('left')
    
    axs[0].set_yticklabels([f"{int(t/1000)}" for t in axs[0].get_yticks()])
    axs[0].set_title(f"{sel_case_name}, @ LST\ncross-section MSE / wind", 
                     loc="left")
    axs[0].set_ylabel("Height [km]")
    
    L_sprec, = axs[1].plot(sprec.isel(time=_t), 'b--')
    axs[1].set_ylabel("sprec [mm/10mins]")
    axs[1].set_xlim(0, res.shape[0]-1)
    axs[1].set_ylim(0, 30)
    axs[0].set_xticks([18], labels=["Wu-Lai"])
    axs[1].axvline(18, c='gray', ls='--')
    axs[1].set_xticks(_X[::10])
    axs[1].set_xticklabels([x//2 for x in _X[::10]])
    axs[1].set_xlabel("[km]")
    print("fig. init complete")
    return 
    def update_plot(i):
        print(f"{i}", end='\r')
        axs[0].set_title(f"{sel_case_name}, @ LST {f'{i//6+10:02d}:{i%6*10:02d}'}\ncross-section dBZ / wind", loc='left')
        L_sprec.set_ydata(sprec.isel(time=i))
        Q.set_UVC(V_h[i, _zskip, _xskip], w[i, _zskip, _xskip])
        CS.set_array(ze.isel(time=i))
    
    anim   = FuncAnimation(fig, update_plot, frames=sprec.shape[0])
    writer = FFMpegWriter(fps=5)
    anim.save(genofname("test.mp4"), writer=writer)

# ==================================================
from time import perf_counter

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))
