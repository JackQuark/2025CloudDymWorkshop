# Radar vertical profile
# ==================================================
import numpy as np
import matplotlib.pyplot as plt
# ==================================================

def main():
    Range     = np.loadtxt("data/Range.dat")
    Elevation = np.loadtxt("data/Elevation.dat")
    Azimuth   = np.loadtxt("data/Azimuth.dat")
    DBZ       = np.loadtxt("data/DBZ.dat")
    # count the number of data point in each Elevation
    ELE_type, ELE_counts = np.unique(Elevation, return_counts=True)
    # ===== Classify Data by Elevation =====
    alldata = {
    	"DBZ": {},
    	"ELE": {},
    	"AZI": {}
    }
    pre_idx = 0
    for i, n in enumerate(ELE_counts):
        alldata["DBZ"][f"{i+1:d}"] = DBZ[pre_idx:pre_idx+n, :]
        alldata["ELE"][f"{i+1:d}"] = Elevation[pre_idx:pre_idx+n]
        alldata["AZI"][f"{i+1:d}"] = Azimuth[pre_idx:pre_idx+n]
        pre_idx += n
    # ===== Classify Data by Elevation =====

    def sel_AZI_data(sel_AZI: int | float, data: dict):
    	# mark the index of selected AZI in each elevation
        sel_AZI_idxes = []
        sel_AZI_data  = []
        for i in range(ELE_type.size):
            # search the index of selected AZI in each elevation
            sel_AZI_idxes.append(np.argmin(np.abs(data['AZI'][f"{i+1:d}"] - sel_AZI)))
            sel_AZI_data.append(data['DBZ'][f"{i+1:d}"][sel_AZI_idxes[-1]])
        sel_AZI_data = np.array(sel_AZI_data)
        return np.where(sel_AZI_data<0, np.nan, sel_AZI_data)
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=40)
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])

    c_lon, c_lat = (121.5393, 25.0181) # Radar pos.
    x = np.cos(np.deg2rad(ELE_type))[:, None] * Range[None, :]
    z = np.sin(np.deg2rad(ELE_type))[:, None] * Range[None, :]

    sel_AZI = 0
    ax.pcolormesh(x, z, sel_AZI_data(sel_AZI, alldata) , cmap=cmap, norm=norm, alpha=0.5)
    sel_AZI = 180
    ax.pcolormesh(-x, z, sel_AZI_data(sel_AZI, alldata), cmap=cmap, norm=norm, alpha=0.5)

    cbar = plt.colorbar(m, ax=ax, shrink=0.6)
    cbar.set_label('dBZ')
    cbar.ax.yaxis.set_label_position('left')

    x_km  = np.linspace(-30, 30, 1004*2-1)
    ax.set_title("NTU Radar @ LST 2024-07-31 07:48:00")
    ax.set_xticks(np.linspace(-30, 30, 9))
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('Height [km]')
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 10)

# ==================================================
from time import perf_counter
if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))