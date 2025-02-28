import sys
sys.path.append("/data/mlcloud/mlpbl_2025/b12209017/2025CDW/nctool")
import vvmds

ds = vvmds.VVMDataset("/data2/VVM/taiwanvvm_tpe/tpe20060623nor")
print(ds)

"""output:
VVM Dataset Info. tpe20060623nor
time steps: 0-144
nc types: ['C.LandSurface', 'C.LandSurface', 'C.Surface', 'L.Dynamic', 'L.Radiation', 'L.Thermodynamic']
"""

# you can select a specific time step or a range of time steps (slice | list | np.ndarray)
with ds.open_ncdataset('dym', step=slice(0, 10)) as ds_dym:
    print(ds_dym)
    print(ds_dym.u.shape)
    print(ds_dym.v.shape)
    
"""output:
Dimensions:  (time: 10, lev: 70, lat: 1024, lon: 1024)
Coordinates:
  * time     (time) datetime64[ns] 80B 1900-01-01T23:40:00 ... 1900-01-01T01:...
  * lev      (lev) float64 560B 0.0 0.05 0.15 0.25 ... 16.6 17.46 18.35 19.26
  * lat      (lat) float64 8kB -2.3 -2.296 -2.291 -2.287 ... 2.291 2.296 2.3
  * lon      (lon) float64 8kB -2.3 -2.296 -2.291 -2.287 ... 2.291 2.296 2.3
    ...
    eta      (time, lev, lat, lon) float32 3GB dask.array<chunksize=(1, 70, 1024, 1024), meta=np.ndarray>
    zeta     (time, lev, lat, lon) float32 3GB dask.array<chunksize=(1, 70, 1024, 1024), meta=np.ndarray>
...
(10, 70, 1024, 1024)
(10, 70, 1024, 1024) 
"""

ilat, ilon = 672, 480
idxrange   = 256
lon_bound  = (ilon, ilon+idxrange)
lat_bound  = (ilat, ilat+idxrange)

# use preprocess function to select a region of interest
from functools import partial
def _preprocess(x, lon_bnds, lat_bnds):
    return x.isel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))

partial_func = partial(_preprocess, lon_bnds=lon_bound, lat_bnds=lat_bound)

# step deafult is None, all steps will be selected
with ds.open_ncdataset('surf', preprocess=partial_func) as ds_lsurf:
    print(ds_lsurf)
    
"""output:
Dimensions:  (time: 144, lat: 256, lon: 256, lev: 70)
Coordinates:
  * time     (time) datetime64[ns] 1kB 1900-01-01T23:50:00 ... 1900-01-01T23:...
  * lat      (lat) float64 2kB 0.7218 0.7263 0.7308 0.7353 ... 1.86 1.864 1.869
    ...
    wth      (time, lat, lon) float32 38MB dask.array<chunksize=(1, 256, 256), meta=np.ndarray>
    wqv      (time, lat, lon) float32 38MB dask.array<chunksize=(1, 256, 256), meta=np.ndarray>
    ...
"""