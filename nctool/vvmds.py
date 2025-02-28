# _summary_
# ==================================================
import os
import numpy as np
import xarray as xr

from time import perf_counter
# ==================================================
__all__ = ['VVMDataset']
# ==================================================

__filedir__ = os.path.dirname(__file__)

ncPrefix_LandSurface   = "C.LandSurface"
ncPrefix_Surface       = "C.Surface"
ncPrefix_Dynamic       = "L.Dynamic"
ncPrefix_Thermodynamic = "L.Thermodynamic"
ncPrefix_Radiation     = "L.Radiation"
ncPrefixes = [ncPrefix_LandSurface, ncPrefix_Surface, ncPrefix_Dynamic, ncPrefix_Thermodynamic, ncPrefix_Radiation]
ncTypes = [s.split('.')[1] for s in ncPrefixes]

exps_dir  = "/data2/VVM/taiwanvvm_tpe"
exps_name = sorted(os.listdir(exps_dir))
exps_path = [os.path.join(exps_dir, exp_name) for exp_name in exps_name]

# ==================================================

class VVMDataset(object):
    def __init__(self, exp_path: str):
        if os.path.exists(exp_path) == False:
            raise ValueError("Experiment path not found: " + exp_path)
        
        self.exp_name = exp_path.split('/')[-1]
        self.exp_path = exp_path
        self.archive_path = os.path.join(exp_path, 'archive')
        self.nc_names: list[str] = self.getname_nclistdir(self.archive_path)
        self.nc_paths: list[str] = self.getpath_nclistdir(self.archive_path)        
        
        self.Nsteps = int(self.nc_names[-1].split('-')[-1].split('.')[0])
        self._tempname()
        
    def _tempname(self):
        self.nc_types = []
        self.AllThenc = {}        
        tot_Nfiles = len(self.nc_names)
        i = 0
        while (True):
            if i >= tot_Nfiles:
                break
            
            current_prefix = self.nc_names[i].split('-')[0][len(self.exp_name)+1:]
            if current_prefix in ncPrefixes:
                self.nc_types.append(current_prefix)
                self.AllThenc[current_prefix] = self.nc_paths[i:i+self.Nsteps]           
                i += self.Nsteps
            else:
                raise ValueError("Unknow prefix: " + current_prefix)
    
    def open_ncdataset(self, nctype: str, step: int = None, **kwargs):
        """open VVM nc file as xarray dataset\n
        **kwargs: additional args to `xr.open_dataset` or `xr.open_mfdataset`\n
        Parameters
        ----------
        nctype : str
            - "lsurf": LandSurface
            - "surf": Surface
            - "dym": Dynamic
            - "thermo": Thermodynamic
            - "rad": Radiation
        step : int or slice or list, optional
            - int: select one time step
            - slice: select a range of time steps
            - list: select a list of time steps
            - None: select all time steps
        **kwargs : additional args to `xr.open_dataset` or `xr.open_mfdataset`
        """
        sel_prefix = self._type_abbr_to_prefix(nctype)
        if not sel_prefix in self.nc_types: raise ValueError("Invalid nctype: " + nctype)
        if step is None: step = slice(0, self.Nsteps)
        
        if isinstance(step, int):
            return xr.open_dataset(self._getpath_selectednc(sel_prefix, step), **kwargs)
        else:
            return xr.open_mfdataset(self._getpath_selectednc(sel_prefix, step), combine='nested', concat_dim="time", **kwargs)
    
    def _getpath_selectednc(self, sel_prefix: str, tstep: int = None):
        if isinstance(tstep, int):
            return self.AllThenc[sel_prefix][tstep]
        elif isinstance(tstep, slice):
            return self.AllThenc[sel_prefix][tstep]
        elif isinstance(tstep, list):
            return [self.AllThenc[sel_prefix][i] for i in tstep]
        elif isinstance(tstep, np.ndarray):
            return [self.AllThenc[sel_prefix][i] for i in tstep]
        elif tstep is None:
            return self.AllThenc[sel_prefix]

    def _type_abbr_to_prefix(self, type_abbr: str):
        """"""
        match type_abbr:
            case "lsurf":
                return ncPrefix_LandSurface
            case "surf":
                return ncPrefix_Surface
            case "dym":
                return ncPrefix_Dynamic
            case "thermo":
                return ncPrefix_Thermodynamic
            case "rad":
                return ncPrefix_Radiation
            case '-':
                msg = (
                    "Invalid nctype: {}\n".format(type_abbr) + 
                    "Available types:\n"
                    "  lsurf: LandSurface\n"
                    "  surf: Surface\n"
                    "  dym: Dynamic\n"
                    "  thermo: Thermodynamic\n"
                    "  rad: Radiation"
                )    
                raise ValueError(msg)  

    @staticmethod
    def getname_nclistdir(archive_path):
        return sorted(os.listdir(archive_path))
    
    @staticmethod
    def getpath_nclistdir(archive_path):
        return [os.path.join(archive_path, name) for name in VVMDataset.getname_nclistdir(archive_path)]

    def __str__(self):
        msg = (
            "VVM Dataset Info. {}\n".format(self.exp_name)
        +   "time steps: 0-{}\n".format(self.Nsteps)
        +   "nc types: {}\n".format(self.nc_types)
        )
        return msg
    
# ==================================================


def main():
    ilat, ilon = 672, 480
    idxrange   = 256
    lon_bound  = (ilon, ilon+idxrange)
    lat_bound  = (ilat, ilat+idxrange)
    
    from functools import partial
    def _preprocess(x, lon_bnds, lat_bnds):
        return x.isel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))
    
    partial_func = partial(_preprocess, lon_bnds=lon_bound, lat_bnds=lat_bound)
    
    vvmds = VVMDataset(exps_path[0])    
    with vvmds.open_ncdataset('lsurf', step=slice(0, 5), preprocess=partial_func) as ds_lsurf:
        print(ds_lsurf)

# ==================================================

if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))