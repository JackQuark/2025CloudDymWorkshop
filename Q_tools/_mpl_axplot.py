# _summary_
# ==================================================
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib.axes     import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ._info_literal import *
# ==================================================
__all__ = [
    'nonlinspace',
    'plot_boundaries',
    'plot_topo',
    'plot_sprec',
    'fig_allax_setting'
]

def nonlinspace(start, end, intervals, bounds):
    N   = len(intervals)
    if N != len(bounds) + 1:
        raise ValueError("len(intervals) should be len(bounds) + 1")
   
    tmp = np.concatenate(([start], bounds, [end]))
    segments = [np.arange(tmp[i], tmp[i+1], intervals[i]) for i in range(N-1)]
    segments.append(np.arange(tmp[-2], tmp[-1]+intervals[-1]/2, intervals[-1]))
    return np.concatenate(segments)

def plot_boundaries(ax, boundaries="city"):
    if boundaries == "city":
        gdf = gpd.read_file(citybound_fpath)
    else:
        gdf = gpd.read_file(townbound_fpath)
    gdf.plot(ax=ax, ls=':', lw=1, fc='whitesmoke', ec='dimgray', zorder=0)

def plot_topo(ax, topo, lon, lat, fill=False, cbar=True):    
    _topo = np.where(topo == 0, np.nan, topo)
    levels= np.arange(np.nanmin(_topo)-1, np.nanmax(_topo)+1, 1, dtype=int)
    
    ax.contour(lon, lat, _topo, levels=levels[1::4], colors='k', alpha=0.5,
               linewidths=0.5, zorder=5)
    
    if fill:
        cmap = plt.get_cmap('binary')
        CS = ax.contourf(lon, lat, _topo, levels=levels, cmap=cmap,
                        alpha=0.6, zorder=0)
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

def plot_sprec(ax, sprec, lon, lat):    
    cmvalue = nonlinspace(2.5, 80, (2.5, 5, 10), (15, 50))
    cmcolor = ['#a0fffa','#00cdff','#0096ff',
               '#0069ff','#329600','#32ff00',
               '#ffff00','#ffc800','#ff9600',
               '#ff0000','#c80000','#a00000',
               '#96009b','#c800d2','#ff00f5',]
    cmap = mplc.ListedColormap(cmcolor).with_extremes(under='none', over='#ffc8ff')
    norm = mplc.BoundaryNorm(cmvalue, cmap.N)
    
    CS = ax.pcolormesh(lon, lat, sprec, cmap=cmap, norm=norm, 
                       zorder=5)
    
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
    
    return ax  

def fig_allax_setting(fig, lon, lat):
    for ax in fig.axes:
        if ax.__class__.__name__ == 'AxesHostAxes': continue
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
