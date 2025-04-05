########################################
######## Author: Wei-Jhih Chen #########
######### Update: 2022/12/13 ###########
####### Modified: Yu-Hung Chang ########
######### Update: 2025/02/21 ###########
########################################
# preserve attenuation correction, remove filtering

import glob
import datetime as dt
from datetime import datetime as dtdt
from read_furuno_wr2100_archive import *
from attenuation_correction import *

import matplotlib.pyplot as plt

##### Date Setting #####
CASE_DATE = '20240731'
CASE_START = dtdt(2024 , 7 , 31 , 7 , 48 , 0)
CASE_END = dtdt(2024 , 7 , 31 , 7 , 48 , 0)

##### File Name Setting #####
PRODUCT_ID = '0092'
INEXT = '[.gz]'
INDIR = '/data/mlcloud/mlpbl_2025/NTUradar/rawdata/20240731/' 
#INDIR = '/data/poyen/NTU_radar/raw/'
INPATHS = glob.glob(INDIR+PRODUCT_ID+'_'+CASE_DATE+'_*.gz')
SEL_AZI_NUM = []
SEL_AZI = []
scan_type='HSQ'

##### Parameters Setting #####
VAR_IN = ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH']
VAR_SELECT = ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL' , 'WIDTH' , 'DBZ_AC' , 'ZDR_AC']
VAR = {'DBZ': {'name': 'DZ' , 'plotname': 'Z$_{HH}$' , 'units': 'dBZ' , 'data': None} , 
       'ZDR': {'name': 'ZD' , 'plotname': 'Z$_{DR}$' , 'units': 'dB' , 'data': None} , 
       'PHIDP': {'name': 'PH' , 'plotname': r'$\phi$$_{DP}$' , 'units': 'Deg.' , 'data': None} , 
       'KDP': {'name': 'KD' , 'plotname': 'K$_{DP}$' , 'units': 'Deg. km$^{-1}$' , 'data': None} , 
       'RHOHV': {'name': 'RH' , 'plotname': r'$\rho$$_{HV}$' , 'units': '' , 'data': None} , 
       'VEL': {'name': 'VR' , 'plotname': 'V$_R$' , 'units': 'm s$^{-1}$' , 'data': None} , 
       'WIDTH': {'name': 'SW' , 'plotname': 'SW' , 'units': 'm s$^{-1}$' , 'data': None} , 
       'RRR': {'name': 'RR' , 'plotname': 'RainRate' , 'units': 'mm hr$^{-1}$' , 'data': None} , 
       'QC_INFO': {'name': 'QC' , 'plotname': 'QC Info' , 'units': '' , 'data': None} , 
       'DBZ_AC': {'name': 'DZac' , 'plotname': 'Z$_{HH}$ (AC)' , 'units': 'dBZ' , 'data': None} , 
       'ZDR_AC': {'name': 'ZDac' , 'plotname': 'Z$_{DR}$ (AC)' , 'units': 'dB' , 'data': None}}
FILTER = {'DBZ': {'min': 0 , 'max': None , 'var': ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'RHOHV' , 'VEL']} , 
          'RHOHV': {'min': 0.7 , 'max': 1.1 , 'var': ['DBZ' , 'ZDR' , 'PHIDP' , 'KDP' , 'VEL']}}
INVALID = -999

def main():
  ##### Load #####
  SEL_TIMES = find_volume_scan_times(INPATHS , CASE_START , CASE_END)
  for sel_time in SEL_TIMES:
    if(scan_type=='HSQ'):  
      ##### Load HSQ #####
      # read data
      (datetimes , NULL , NULL , NULL , 
        LATITUDE , LONGITUDE , ALTITUDE , 
        NULL , NULL , Fixed_angle , 
        Sweep_start_ray_index , Sweep_end_ray_index , 
        Range , Azimuth , Elevation , Fields) = read_volume_scan(INDIR , PRODUCT_ID , sel_time , INEXT)
      # setting
      datetimeLST = datetimes['data'][0] + dt.timedelta(hours = 8)
      Fixed_angle = Fixed_angle['data']
      Sweep_start_ray_index = Sweep_start_ray_index['data']
      Sweep_end_ray_index = Sweep_end_ray_index['data']
      Range = Range['data']
      Azimuth = Azimuth['data']
      Elevation = Elevation['data']
      for var in VAR_IN:
          VAR[var]['data'] = Fields[var]['data']
      # Invalid Value
      for var in VAR_IN:
          VAR[var]['data'] = ma.array(VAR[var]['data'] , mask = VAR[var]['data'] == INVALID , copy = False)
      VAR['DBZ_AC']['data'] , VAR['ZDR_AC']['data'] = attenuation_correction_X(VAR['DBZ']['data'] , VAR['ZDR']['data'] , VAR['KDP']['data'] , Range[1] - Range[0])

    elif(scan_type=='RHI'):
      ##### Load RHI #####    
      INFILES = find_volume_scan_files(INDIR , PRODUCT_ID , sel_time , INEXT)
      NUM_FILE_LIST = SEL_AZI_NUM if SEL_AZI_NUM else range(len(INFILES))
      for cnt_file in NUM_FILE_LIST:
          # read data
          (datetime , metadata , INSTRUMENT_PARAMETERS , SCAN_TYPE , 
          LATITUDE , LONGITUDE , ALTITUDE , 
          sweep_number , sweep_mode , aziFix , 
          sweep_start_ray_index , sweep_end_ray_index , 
          RANGE , azimuth , ELEVATION , fields) = reader_corrected_by_radar_constant(INFILES[cnt_file])
          # setting
          RANGE = RANGE['data']
          ELEVATION = ELEVATION['data']
          datetimeLST = datetime['data'] + dt.timedelta(hours = 8)
          # azimuth        
          if SEL_AZI:
              if not([True for azi in SEL_AZI if abs(aziFix['data'] - azi) <= 1]):
                  print(f"Skip Azimuth: {aziFix['data']:.2f}^o")
                  continue
          # get data
          for var in VAR_IN:
              VAR[var]['data'] = fields[var]['data']
          # Invalid Value
          VAR['DBZ_AC']['data'] , VAR['ZDR_AC']['data'] = attenuation_correction_X(VAR['DBZ']['data'] , VAR['ZDR']['data'] , VAR['KDP']['data'] , RANGE[1] - RANGE[0])
  print("data loading complete")
  
  for key, val in VAR.items():
      if key in VAR_SELECT:
        print(key, val['data'].shape)

  return 
  np.savetxt("data/Range.dat", Range, fmt='%f')
  np.savetxt("data/Elevation.dat", Elevation, fmt='%f')
  np.savetxt("data/Azimuth.dat", Azimuth, fmt='%f')
  np.savetxt("data/DBZ.dat", DBZ, fmt='%f')

# ==================================================
from time import perf_counter
if __name__ == '__main__':
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))