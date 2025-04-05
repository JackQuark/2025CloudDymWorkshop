# Radar vertical profile
# ==================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
# ==================================================
def calc_cosrule(s1, s2, theta, unit='rad'):
    """Law of Cosines: length of 3rd side """
    if unit == 'deg':
        theta = np.deg2rad(theta)
    return np.sqrt(s1**2 + s2**2 - 2*s1*s2*np.cos(theta))

def calc_anti_cosrule(s1, s2, s3, unit='rad'):
    """Law of Cosines: angle between s1, s2"""
    res = np.arccos((s1**2 + s2**2 - s3**2) / (2*s1*s2))
    if unit == 'deg':
        return np.rad2deg(res)
    return res

def calc_AZI2polar(AZI):
    """"""
    if isinstance(AZI, (int, float)):
        if AZI <= 90:
            return 90 - AZI
        else:
            return 450 - AZI
    elif hasattr(AZI, '__iter__'):
        if np.any(AZI // 360 >= 1):
            AZI = AZI % 360
        return np.where(AZI <= 90, 90 - AZI, 450 - AZI)

    else: raise TypeError("Invalid input type")

def calc_polar2AZI(theta):
    """"""
    if isinstance(theta, (int, float)):
        if theta <= 90:
            return 90 - theta
        else:
            return 450 - theta
    elif hasattr(theta, '__iter__'):
        theta = np.array(theta)
        return np.where(theta <= 90, 90 - theta, 450 - theta)
    
    else: raise TypeError("Invalid input type")

def calc_dist_2pts(*args):
    """calc. distance between two pts\n
    Args:
        x1, y1, x2, y2
        or 
        (x1, y1), (x2, y2)
    Returns:
        _type_: _description_
    """
    if len(args) == 2:
        x1, y1 = args[0]
        x2, y2 = args[1]
    elif len(args) == 4:
        x1, y1, x2, y2 = args
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def calc_dist_2loc(lon1, lat1, lon2, lat2):
    """calc. \n
    return:
        dx: float, distance of `lat2-lat1`
        dy: float, distance of `lat2-lat1`
        dist: float, distance between two points
    """
    R = 6371 # Earth's radius (m)
    dist_perlon = R * np.cos(np.deg2rad(max(lat1, lat2))) * 2 * np.pi / 360
    dist_perlat = R * np.pi / 180
    dx = (lon2 - lon1) * dist_perlon
    dy = (lat2 - lat1) * dist_perlat
    return dx, dy, np.sqrt(dx**2 + dy**2)

def gen_line_1pt1slope(pt, slope, length, cpt_ratio=0.5):
    """return 3pts (start_pt, pt, end_pt)
    """
    ndim = len(pt)
    if ndim != 2: raise ValueError("pt must be 2D")
    theta = np.arctan(slope) # rad
    ds1 = cpt_ratio     * length
    ds2 = (1-cpt_ratio) * length

    start_pt = (pt[0] - ds1*np.cos(theta), pt[1] - ds1*np.sin(theta))
    end_pt   = (pt[0] + ds2*np.cos(theta), pt[1] + ds2*np.sin(theta))
    return start_pt, pt, end_pt

def find_closest_pt(A, B, P):
    """find the closest point of P to line AB\n"""
    x1, y1 = A
    x2, y2 = B
    x0, y0 = P

    AB = np.array([x2 - x1, y2 - y1])
    AP = np.array([x0 - x1, y0 - y1])
    t = np.dot(AP, AB) / np.dot(AB, AB)
    return (x1 + t * AB[0], y1 + t * AB[1])

def main():
    Range     = np.loadtxt("data/Range.dat") * 1000    
    Elevation = np.loadtxt("data/Elevation.dat")
    Azimuth   = np.loadtxt("data/Azimuth.dat")
    DBZ       = np.loadtxt("data/DBZ.dat")
    DBZ = np.where(DBZ <= 0, np.nan, DBZ)
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

    c_lon, c_lat = (121.5393, 25.0181) # Radar pos.
    dx, dy, dist = calc_dist_2loc(c_lon, c_lat, c_lon, 25.1)

    # ===== test polar coord data pts =====
    r = Range / 1000
    theta = np.linspace(0, 2*np.pi, 36, endpoint=False)
    xs = r[None, ::50] * np.sin(theta)[:, None]
    ys = r[None, ::50] * np.cos(theta)[:, None]

    plt.rcParams['font.size'] = 10
    plt.rcParams['lines.linewidth'] = 1.
    plt.rcParams['lines.markersize'] = 3.
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.scatter(xs, ys, s=1)

    # ===== sel. secant line =====
    sel_pt = (0, 20)
    sel_AZI = 30
    sel_AZI_T = calc_AZI2polar(sel_AZI)
    ax.plot(*sel_pt, 'ro')
    tmp = gen_line_1pt1slope(sel_pt, np.tan(np.deg2rad(sel_AZI_T)), 200, 0.5)
    tmp = np.array(tmp)
    ax.plot(tmp[:, 0], tmp[:, 1], 'r-', zorder=0)
    
    # q: foot of perpendicular from (0, 0) to line AB
    q = find_closest_pt(tmp[0], tmp[-1], (0, 0))
    # angle of Oq vec. in polar coord.
    q_angle = np.rad2deg(np.arctan2(q[1], q[0]))
    
    s1 = calc_dist_2pts(q, (0, 0))
    s2 = r[-1]
    s3 = np.sqrt(s2**2 - s1**2)

    # theta: angle between Oq and Oa vec. where a is one of the contact point of AB
    theta = calc_anti_cosrule(s1, s2, s3, unit='deg')
    max_height = s1 * np.tan(np.deg2rad(ELE_type[-1]))

    def test_bdLine():
        ax.plot([0, s1*np.cos(np.deg2rad(q_angle))], 
                [0, s1*np.sin(np.deg2rad(q_angle))], 'g-', zorder=0)
        ax.plot([0, s2*np.cos(np.deg2rad(q_angle+theta))], 
                [0, s2*np.sin(np.deg2rad(q_angle+theta))], 'b-', zorder=0)
        ax.plot([0, s2*np.cos(np.deg2rad(q_angle-theta))], 
                [0, s2*np.sin(np.deg2rad(q_angle-theta))], 'b-', zorder=0)
    test_bdLine()

    angle_bd_polar = (q_angle - theta, q_angle + theta)
    angle_bd_AZI   = sorted(calc_polar2AZI(angle_bd_polar))

    bddata = {
        "DBZ": {},
        "ELE": {},
        "AZI": {},
        "polar": {},
        "dist_2_section": {},
        "sel_DBZ": {}
    }

    # fig, axs = plt.subplots(6, 2, figsize=(8, 10), sharey=True)
    # fig.subplots_adjust(hspace=0.5, wspace=0.2)
    for i in range(12):
        k = i+1
        
        if abs(angle_bd_AZI[0] - angle_bd_AZI[1]) >= 180:
            cond = np.logical_or(alldata['AZI'][f"{k:d}"] <= angle_bd_AZI[0], alldata['AZI'][f"{k:d}"] >= angle_bd_AZI[1])
        else:
            cond = np.logical_and(alldata['AZI'][f"{k:d}"] >= angle_bd_AZI[0], alldata['AZI'][f"{k:d}"] <= angle_bd_AZI[1])
    
        bddata['DBZ'  ][f"{k:d}"] = alldata['DBZ'][f"{k:d}"][cond]
        bddata['ELE'  ][f"{k:d}"] = alldata['ELE'][f"{k:d}"][cond]
        bddata['AZI'  ][f"{k:d}"] = alldata['AZI'][f"{k:d}"][cond]
        bddata['polar'][f"{k:d}"] = calc_AZI2polar(bddata['AZI'][f"{k:d}"])

        tmp_angle = bddata['polar'][f"{k:d}"] - q_angle
        dist_2_secant = s1 / np.cos(np.deg2rad(tmp_angle))
        
        bddata['dist_2_section'][f"{k:d}"] = dist_2_secant / np.cos(np.deg2rad(ELE_type[i]))

        tmp = np.argmin(bddata['dist_2_section'][f"{k:d}"][:, None] - r, axis=1)
        bddata['sel_DBZ'][f"{k:d}"] = np.take_along_axis(bddata['DBZ'][f"{k:d}"], tmp[:, None], axis=1)[:, 0]
        
        # # test dist to secant line
        # for j in range(0, dist_2_secant.size, 10):
        #     ax.plot([0, dist_2_secant[j]*np.cos(np.deg2rad(bddata['polar'][f"{k:d}"][j]))], 
        #             [0, dist_2_secant[j]*np.sin(np.deg2rad(bddata['polar'][f"{k:d}"][j]))], 'g--', zorder=0)
        # return 

    maxN_AZI = np.max([arr.shape for arr in bddata['AZI'].values()])
    totN_ELE = len(ELE_type)
    sec_AZI = np.zeros((totN_ELE, maxN_AZI))
    sec_DBZ = np.zeros((totN_ELE, maxN_AZI))
    sec_dist2sec = np.zeros((totN_ELE, maxN_AZI))
    sec_ELE = np.copy(ELE_type)
    for i in range(totN_ELE):
        sec_AZI[i, :bddata['AZI'][f"{i+1:d}"].size] = bddata['AZI'][f"{i+1:d}"]
        sec_DBZ[i, :bddata['AZI'][f"{i+1:d}"].size] = bddata['sel_DBZ'][f"{i+1:d}"]
        sec_dist2sec[i, :bddata['AZI'][f"{i+1:d}"].size] = bddata['dist_2_section'][f"{i+1:d}"]
        
    sec_xcoord = sec_dist2sec * np.cos(np.deg2rad(sec_ELE)[:, None]) * np.sin(np.deg2rad(calc_AZI2polar(sec_AZI) - q_angle))
    sec_zcoord = sec_dist2sec * np.sin(np.deg2rad(sec_ELE)[:, None])
    
    print(sec_xcoord.max(), sec_xcoord.min(), sec_zcoord.max(), sec_zcoord.min())

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    ax2.scatter(sec_xcoord, sec_zcoord, s=1, c=sec_DBZ, cmap='jet')
    ax2.set_xlim(-45, 45)
    ax2.set_ylim(0, 18)

    return 
    fig, axs = plt.subplots(6, 2, figsize=(8, 10), sharey=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    for i in range(6):
        for j in range(2):
            k = i*2+j
            axs[i, j].plot(alldata['AZI'][f"{k+1:d}"])
            if abs(angle_bd_AZI[0] - angle_bd_AZI[1]) >= 180:
                cond = np.logical_or(alldata['AZI'][f"{k+1:d}"] <= angle_bd_AZI[0], alldata['AZI'][f"{k+1:d}"] >= angle_bd_AZI[1])
            else:
                cond = np.logical_and(alldata['AZI'][f"{k+1:d}"] >= angle_bd_AZI[0], alldata['AZI'][f"{k+1:d}"] <= angle_bd_AZI[1])
            axs[i, j].plot(np.where(cond, alldata['AZI'][f"{k+1:d}"], np.nan), 'r.')
            axs[i, j].set_title(f"ELE {ELE_type[k]:.1f}")
            axs[i, j].set_xlim(0, alldata['AZI'][f"{k+1:d}"].size-1)
            axs[i, j].set_xticks([alldata['AZI'][f"{k+1:d}"].size-1], labels=[f"{alldata['AZI'][f"{k+1:d}"].size:d}"])            
    axs[i, j].set_yticks(np.linspace(0, 360, 5))

# ==================================================
from time import perf_counter
if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))