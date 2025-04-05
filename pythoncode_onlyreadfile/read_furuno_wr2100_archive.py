# -*- coding:utf-8 -*-

########################################
#### read_furuno_wr2100_archive.py #####
######## Author: Wei-Jhih Chen #########
######### Update: 2022/12/13 ###########
########################################

import gzip
import struct
import numpy as np
import numpy.ma as ma
from pathlib import Path
from typing import Any
from datetime import datetime as dtdt

INSTRUMENT_NAME = 'FURUNO WR2100'  # Instrument Name
DATA_TYPE = 'binary'

def reader(inPath: Path) -> tuple[dict[str , dtdt] , dict[str , str] , dict[str , float] , str , 
                            dict[str , float] , dict[str , float] , dict[str , float] , 
                            dict[str , int] , dict[str , str] , dict[str , float] , 
                            dict[str , int] , dict[str , int] , 
                            dict[str , np.ndarray] , dict[str , np.ndarray] , dict[str , np.ndarray] , dict[str , dict[str , Any]]]:
    if str(inPath)[-3:] == '.gz':
        
        scan_type = str(inPath)[-6:-3]
        sweep_number = {'data': int(str(inPath)[-12:-10])}
        sweep_mode = {'data': scan_type}
        file = gzip.open(inPath , 'rb')
    else:
        scan_type = str(inPath)[-3:]
        sweep_number = {'data': int(str(inPath)[-7:-4])}
        sweep_mode = {'data': scan_type}
        file = open(inPath , 'rb')

    data = file.read()
    num_byte = len(data)

    ########## Header ##########
    cnt_byte = 0
    size_of_header = struct.unpack('<H' , data[0 : 2])[0]                               # Size of header; Units: byte
    version_of_data_format = struct.unpack('<H' , data[2 : 4])[0]                       # Production type information & Version of data format; Range: 0-99
    year_log = struct.unpack('<H' , data[4 : 6])[0]                                     # DPU Log time: year
    month_log = struct.unpack('<H' , data[6 : 8])[0]                                    # DPU Log time: month
    day_log = struct.unpack('<H' , data[8 : 10])[0]                                     # DPU Log time: day
    hour_log = struct.unpack('<H' , data[10 : 12])[0]                                   # DPU Log time: hour
    minute_log = struct.unpack('<H' , data[12 : 14])[0]                                 # DPU Log time: minute
    second_log = struct.unpack('<H' , data[14 : 16])[0]                                 # DPU Log time: second
    latitude_degree = struct.unpack('<h' , data[16 : 18])[0]                            # Units: degree (N+S-)
    latitude_minute = struct.unpack('<H' , data[18 : 20])[0]                            # Units: minute
    latitude_second = struct.unpack('<H' , data[20 : 22])[0] / 1000                     # Units: second
    longitude_degree = struct.unpack('<h' , data[22 : 24])[0]                           # Units: degree (E+W-)
    longitude_minute = struct.unpack('<H' , data[24 : 26])[0]                           # Units: minute
    longitude_second = struct.unpack('<H' , data[26 : 28])[0] / 1000                    # Units: second
    antenna_altitude_upper = struct.unpack('<H' , data[28 : 30])[0]                     # Range: 0-65535
    antenna_altitude_lower = struct.unpack('<H' , data[30 : 32])[0]                     # Range: 0-9999
    antenna_rotation_speed_azimuth = struct.unpack('<H' , data[32 : 34])[0] / 10        # Units: rpm
    prf1 = struct.unpack('<H' , data[34 : 36])[0] / 10                                  # Units: Hz
    prf2 = struct.unpack('<H' , data[36 : 38])[0] / 10                                  # Units: Hz
    noise_level_pulse_modulation_H = struct.unpack('<h' , data[38 : 40])[0] / 100       # Units: dBm
    noise_level_frequency_modulation_H = struct.unpack('<h' , data[40 : 42])[0] / 100   # Units: dBm
    total_number_of_sweep = struct.unpack('<H' , data[42 : 44])[0]                      # Units: qty
    number_of_range_direction_data = struct.unpack('<H' , data[44 : 46])[0]             # Units: qty
    resolution_of_range_direction = struct.unpack('<H' , data[46 : 48])[0] / 100000     # Units: km
    constant_radar_mantissa_H = struct.unpack('<l' , data[48 : 52])[0]                  # Range: -999999999-999999999
    constant_radar_characteristic_H = struct.unpack('<h' , data[52 : 54])[0]            # Range: -32768-32767
    constant_radar_mantissa_V = struct.unpack('<l' , data[54 : 58])[0]                  # Range: -999999999-999999999
    constant_radar_characteristic_V = struct.unpack('<h' , data[58 : 60])[0]            # Range: -32768-32767
    azimuth_offset = struct.unpack('<H' , data[60 : 62])[0] / 100                       # Offset value of North and radar direction of origin
    if size_of_header == 80:
        year_record = struct.unpack('<H' , data[62 : 64])[0]                            # Record UTC time: year
        month_record = struct.unpack('<H' , data[64 : 66])[0]                           # Record UTC time: month
        day_record = struct.unpack('<H' , data[66 : 68])[0]                             # Record UTC time: day
        hour_record = struct.unpack('<H' , data[68 : 70])[0]                            # Record UTC time: hour
        minute_record = struct.unpack('<H' , data[70 : 72])[0]                          # Record UTC time: minute
        second_record = struct.unpack('<H' , data[72 : 74])[0]                          # Record UTC time: second
        record_item = np.unpackbits(bytearray(data[74 : 76]) , bitorder = 'little')     # bit0: Rain, bit1: Zhh, bit2: V, bit3: Zdr, bit4: Kdp, bit5: phi-dp, bit6: rho-hv, bit7: W, bit8: quality information, bit9-15: reserved
        tx_pulse_blind_area = struct.unpack('<H' , data[76 : 78])[0] / 1000             # Units: km
        tx_pulse_specification = struct.unpack('<H' , data[78 : 80])[0]

    datetime = dtdt(year_record , month_record , day_record , hour_record , minute_record , second_record)
    radar_constant_H = constant_radar_mantissa_H * 10 ** constant_radar_characteristic_H
    radar_constant_V = constant_radar_mantissa_V * 10 ** constant_radar_characteristic_V

    ########## Observation data ##########
    cnt_byte = size_of_header
    num_ray = total_number_of_sweep
    num_rng = number_of_range_direction_data
    invalid_number = -999

    azimuth = np.zeros([num_ray] , dtype = np.float32)
    azi_element = np.array([] , dtype = np.float32)
    azi_count = np.array([] , dtype = np.ushort)
    elevation = np.zeros([num_ray] , dtype = np.float32)
    rain = np.zeros([num_ray , num_rng] , dtype = np.float32)
    zhh = np.zeros([num_ray , num_rng] , dtype = np.float32)
    v = np.zeros([num_ray , num_rng] , dtype = np.float32)
    zdr = np.zeros([num_ray , num_rng] , dtype = np.float32)
    kdp = np.zeros([num_ray , num_rng] , dtype = np.float32)
    phidp = np.zeros([num_ray , num_rng] , dtype = np.float32)
    rhohv = np.zeros([num_ray , num_rng] , dtype = np.float32)
    w = np.zeros([num_ray , num_rng] , dtype = np.float32)
    qc = np.zeros([num_ray , num_rng] , dtype = np.ushort)
    # quality_information = np.zeros([num_ray , num_rng] , dtype = np.uint8)
    # signal_shading = np.zeros([num_ray , num_rng] , dtype = np.uint8)
    # signal_extinction = np.zeros([num_ray , num_rng] , dtype = np.uint8)
    # clutter_reference = np.zeros([num_ray , num_rng] , dtype = np.uint8)
    # pulse_blind_area = np.zeros([num_ray , num_rng] , dtype = np.uint8)
    # sector_blank = np.zeros([num_ray , num_rng] , dtype = np.uint8)
    # fix_1 = np.zeros([num_ray , num_rng] , dtype = np.uint8)
    for cnt_ray in np.arange(num_ray):
        ########## Observation angularity information ##########
        information_ID = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]                                         # Units: byte
        azimuth[cnt_ray] = (struct.unpack('<H' , data[cnt_byte + 2 : cnt_byte + 4])[0] / 100 + azimuth_offset) % 360    # Range: 0-359.99; Units: degree; Angle from initial position of ATU; Initial position: 0 deg
        if cnt_ray == 0:
            azi_element = np.append(azi_element , azimuth[cnt_ray])
            azi_count = np.append(azi_count , 1)
        elif azimuth[cnt_ray] != azi_element[-1]:
            azi_element = np.append(azi_element , azimuth[cnt_ray])
            azi_count = np.append(azi_count , 1)
        else:
            azi_count[-1] += 1
        elevation[cnt_ray] = struct.unpack('<h' , data[cnt_byte + 4 : cnt_byte + 6])[0] / 100                           # Range: -3.00-180.00; Horizontal: 0 deg, Elevation: +, Dip:-

        ########## Observation data ##########
        cnt_byte += information_ID
        observed_data_size = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]                 # Units: byte
        cnt_byte += 2

        if record_item[0] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                rain[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]    # Range: 0-65535; Invalid: 0
                cnt_byte += 2
        if record_item[1] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                zhh[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]     # Range: 0-65535; Invalid: 0
                cnt_byte += 2
        if record_item[2] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                v[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]       # Range: 0-65535; Invalid: 0
                cnt_byte += 2
        if record_item[3] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                zdr[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]     # Range: 0-65535; Invalid: 0
                cnt_byte += 2
        if record_item[4] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                kdp[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]     # Range: 0-65535; Invalid: 0
                cnt_byte += 2
        if record_item[5] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                phidp[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]   # Range: 0-65535; Invalid: 0
                cnt_byte += 2
        if record_item[6] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                rhohv[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]   # Range: 0-65535; Invalid: 0
                cnt_byte += 2
        if record_item[7] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                w[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]       # Range: 0-65535; Invalid: 0
                cnt_byte += 2
        if record_item[8] == 1:
            for cnt_rng in np.arange(0 , num_rng):
                qc[cnt_ray , cnt_rng] = struct.unpack('<H' , data[cnt_byte : cnt_byte + 2])[0]      # Range: 0-65535; Invalid: 0
                # bits = np.unpackbits(bytearray(data[cnt_byte : cnt_byte + 2]) , bitorder = 'little')
                # quality_information[cnt_ray , cnt_rng] = bits[3] + bits[4] * 2 + bits[5] * 2 ** 2
                # signal_shading[cnt_ray , cnt_rng] = bits[0]
                # signal_extinction[cnt_ray , cnt_rng] = bits[1]
                # clutter_reference[cnt_ray , cnt_rng] = bits[2]
                # pulse_blind_area[cnt_ray , cnt_rng] = bits[6]
                # sector_blank[cnt_ray , cnt_rng] = bits[7]
                # fix_1[cnt_ray , cnt_rng] = bits[8]
                cnt_byte += 2
    
    # azi_mean = np.mean(azi)                                                   # Not avoid transition data
    fixed_angle = azi_element[np.argmax(azi_count)]                             # Avoid transition data

    rain[rain == 0] = invalid_number
    rain = (ma.masked_values(rain , invalid_number) - 32768) / 100              # Rain Range: -327.67-327.67; Invalid: -999; Units: mm/h
    zhh[zhh == 0] = invalid_number
    zhh = (ma.masked_values(zhh , invalid_number) - 32768) / 100                # Zhh Range: -327.67-327.67; Invalid: -999; Units: dBz
    v[v == 0] = invalid_number
    v = (ma.masked_values(v , invalid_number) - 32768) / 100                    # V Range: -327.67-327.67; Invalid: -999; Units: m/s
    zdr[zdr == 0] = invalid_number
    zdr = (ma.masked_values(zdr , invalid_number) - 32768) / 100                # Zdr Range: -327.67-327.67; Invalid: -999; Units: dB
    kdp[kdp == 0] = invalid_number
    kdp = (ma.masked_values(kdp , invalid_number) - 32768) / 100                # Kdp Range: -327.67-327.67; Invalid: -999; Units: deg/km
    phidp[phidp == 0] = invalid_number
    phidp = (ma.masked_values(phidp , invalid_number) - 32768) * 360 / 65535    # Phidp Range: -179.9972-179.9972; Invalid: -999; Units: deg
    rhohv[rhohv == 0] = invalid_number
    rhohv = (ma.masked_values(rhohv , invalid_number) - 1) * 2 / 65534          # Rhohv Range: 0.0-2.0; Invalid: -999; Units:
    w[w == 0] = invalid_number
    w = (ma.masked_values(w , invalid_number) - 1) / 100                        # W Range: 0.00-655.34; Invalid: -999; Units: m/s

    sweep_start_ray_index = 0
    sweep_end_ray_index = total_number_of_sweep - 1

    ########## Output data ##########
    datetime = {'data': datetime , 'units': 'UTC'}  # UTC time
    range = {'data': np.arange(0 , number_of_range_direction_data * resolution_of_range_direction , resolution_of_range_direction) , 'units': 'km'}
    metadata = {'data_type': DATA_TYPE , 'instrument_name': INSTRUMENT_NAME}

    latitude = {'data': latitude_degree + latitude_minute / 60 + latitude_second / 3600}
    longitude = {'data': longitude_degree + longitude_minute / 60 + longitude_second / 3600}
    altitude = {'data': (antenna_altitude_upper * 10000 + antenna_altitude_lower) / 100000 , 'units': 'km'} # Units: km
    
    fields = {'DBZ': {'data': zhh , 'units': 'dBz'} , 
              'VEL': {'data': v , 'units': 'm/s'} , 
              'ZDR': {'data': zdr , 'units': 'dB'} , 
              'KDP': {'data': kdp , 'units': 'deg/km'} , 
              'PHIDP': {'data': phidp , 'units': 'deg'} , 
              'RHOHV': {'data': rhohv , 'units': ''} , 
              'WIDTH': {'data': w , 'units': 'm/s'} , 
              'QC_INFO': {'data': qc , 'units': ''} , 
              'RRR': {'data': rain , 'units': 'mm/h'} , }
            #   'QUALITY_INFO': {'data': quality_information , 'units': 'bit'} , 
            #   'SIG_SHADING': {'data': signal_shading , 'units': 'bit'} , 
            #   'SIG_EXTINCTION': {'data': signal_extinction , 'units': 'bit'} , 
            #   'CLUTTER_REF': {'data': clutter_reference , 'units': 'bit'} , 
            #   'PULSE_BLIND': {'data': pulse_blind_area , 'units': 'bit'} , 
            #   'SECTOR_BLANK': {'data': sector_blank , 'units': 'bit'} , 
            #   'FIX': {'data': fix_1 , 'units': 'bit'}}

    sweep_start_ray_index = {'data': sweep_start_ray_index}
    sweep_end_ray_index = {'data': sweep_end_ray_index}
    fixed_angle = {'data': fixed_angle}
    azimuth = {'data': azimuth}
    elevation = {'data': elevation}
    if tx_pulse_specification == 1 or tx_pulse_specification == 2 or tx_pulse_specification == 3:       pulse_width = 50        # us (Q0N , long)
    elif tx_pulse_specification == 4 or tx_pulse_specification == 7:                                    pulse_width = 0.5       # us (P0N , short)
    elif tx_pulse_specification == 5 or tx_pulse_specification == 8:                                    pulse_width = 0.66      # us (P0N , short)
    elif tx_pulse_specification == 6 or tx_pulse_specification == 9 or tx_pulse_specification == 10:    pulse_width = 1         # us (P0N , short)
    instrument_parameters = {'antenna_rotation_speed_azimuth': antenna_rotation_speed_azimuth , 
                             'prf1': prf1 , 'prf2': prf2 , 
                             'noise_level_pulse_modulation_H': noise_level_pulse_modulation_H , 
                             'noise_level_frequency_modulation_H': noise_level_frequency_modulation_H , 
                             'Tx_pulse_blind_area': tx_pulse_blind_area , 
                             'Tx_pulse_specification': tx_pulse_specification , 
                             'radar_constant_H': radar_constant_H , 
                             'radar_constant_V': radar_constant_V , 
                             'pulse_width': pulse_width , 
                             }

    return (datetime , metadata , instrument_parameters , scan_type , 
            latitude , longitude , altitude , 
            sweep_number , sweep_mode , fixed_angle , 
            sweep_start_ray_index , sweep_end_ray_index , 
            range , azimuth , elevation , fields)

def correct_Zh_Zdr_by_radar_constant(RC_H_ori , RC_V_ori , RC_H_corr , RC_V_corr , Zh , Zdr):
    Zv = 10 * np.log10(10 ** ((Zh - Zdr) / 10))
    Zv = 10 * np.log10((10 ** (Zv / 10) * RC_V_ori / RC_V_corr))
    Zh = 10 * np.log10((10 ** (Zh / 10) * RC_H_ori / RC_H_corr))
    Zdr = 10 * np.log10(10 ** ((Zh - Zv) / 10))
    return Zh , Zdr

def reader_corrected_by_radar_constant(inPath: Path) -> tuple[dict[str , dtdt] , dict[str , str] , dict[str , float] , str , 
        dict[str , float] , dict[str , float] , dict[str , float] , 
        dict[str , int] , dict[str , str] , dict[str , float] , 
        dict[str , int] , dict[str , int] , 
        dict[str , np.ndarray] , dict[str , np.ndarray] , dict[str , np.ndarray] , dict[str , dict[str , Any]]]:

    ########## RADAR CONSTANT CALIBRATION ##########
    # Origin: 1.7027899999999999e-16(H) , 1.9105599999999997e-16(V)
    # After: 6.680812773836028e-15(H,V))
    RADAR_CONSTANT = lambda wavelength , loss , power_transmission , antenna_gain , beamwidth_H , beamwidth_V , pulse_width , K_2: \
                 (np.pi ** 5 * 10 ** -17 * power_transmission * antenna_gain ** 2 * beamwidth_H * beamwidth_V * pulse_width * K_2) / \
                 (6.75 * 2 ** 14 * np.log(2) * wavelength ** 2 * loss) / 1000
    WAVELENGTH = 3.19           # cm
    LOSS_H = 10 ** (4 / 10)     # log10 to ratio
    LOSS_V = 10 ** (4 / 10)     # log10 to ratio
    POWER_TRANSMISSION_H = 100  # W
    POWER_TRANSMISSION_V = 100  # W
    GAIN_H = 10 ** (34.0 / 10)  # log10 to ratio
    GAIN_V = 10 ** (34.0 / 10)  # log10 to ratio
    BEAMWIDTH_H = 2.7           # degree
    BEAMWIDTH_V = 2.7           # degree
    K_2 = 0.93
    
    (datetime , metadata , instrument_parameters , scan_type , 
     latitude , longitude , altitude , 
     sweep_number , sweep_mode , fixed_angle , 
     sweep_start_ray_index , sweep_end_ray_index , 
     range , azimuth , elevation , fields) = reader(inPath)

    RC_H = RADAR_CONSTANT(
        WAVELENGTH , LOSS_H , POWER_TRANSMISSION_H , GAIN_H , 
        BEAMWIDTH_H , BEAMWIDTH_V , instrument_parameters['pulse_width'] , K_2
    )
    RC_V = RADAR_CONSTANT(
        WAVELENGTH , LOSS_V , POWER_TRANSMISSION_V , GAIN_V , 
        BEAMWIDTH_H , BEAMWIDTH_V , instrument_parameters['pulse_width'] , K_2
    )
    fields['DBZ']['data'] , fields['ZDR']['data'] = correct_Zh_Zdr_by_radar_constant(
        instrument_parameters['radar_constant_H'] , instrument_parameters['radar_constant_V'] , 
        RC_H , RC_V , fields['DBZ']['data'] , fields['ZDR']['data']
    )

    instrument_parameters['radar_constant_H'] = RC_H
    instrument_parameters['radar_constant_V'] = RC_V

    return (datetime , metadata , instrument_parameters , scan_type , 
            latitude , longitude , altitude , 
            sweep_number , sweep_mode , fixed_angle , 
            sweep_start_ray_index , sweep_end_ray_index , 
            range , azimuth , elevation , fields)

def find_volume_scan_times(paths , start = None , end = None):
    datetimes = []
    names = [str(path) for path in paths]
    if not names:
        return datetimes
    start = start.timestamp() if start is not None else start
    end = end.timestamp() if end is not None else end
    dateStr = names[0].split('_')[2]
    timeStrs = sorted(list(set([name.split('_')[3] for name in names])))
    dts = [dtdt.strptime(dateStr + timeStr , '%Y%m%d%H%M%S').timestamp() for timeStr in timeStrs]
    for dt in dts:
        if start is None and end is None:
            datetimes.append(dtdt.fromtimestamp(dt))
        else:
            if dt >= start and dt <= end:
                datetimes.append(dtdt.fromtimestamp(dt))
    return datetimes

def find_volume_scan_files(dir: str , id: str , dt: dtdt , ext: str) -> list:
    return sorted(Path(dir).glob(f"{id}_{dtdt.strftime(dt , '%Y%m%d_%H%M%S')}_*{ext}"))

def read_volume_scan(dir: str , id: str , dt: dtdt , ext: str) -> tuple[dict[str , dtdt] , dict[str , str] , dict[str , float] , str , 
        dict[str , float] , dict[str , float] , dict[str , float] , 
        dict[str , int] , dict[str , str] , dict[str , float] , 
        dict[str , int] , dict[str , int] , 
        dict[str , np.ndarray] , dict[str , np.ndarray] , dict[str , np.ndarray] , dict[str , dict[str , Any]]]:

    files = find_volume_scan_files(dir , id , dt , ext)
    num_file = len(files)
    datetimes = {}
    datetimes['data'] = np.empty((num_file) , dtype = dtdt)
    sweep_numbers = np.zeros((num_file) , dtype = int)
    fixed_angles = np.zeros((num_file))
    sweep_start_ray_indexes = np.zeros((num_file) , dtype = int)
    sweep_end_ray_indexes = np.zeros((num_file) , dtype = int)

    (datetime , metadata , instrument_parameters , scan_type , 
     latitude , longitude , altitude , 
     sweep_number , sweep_mode , fixed_angle , 
     sweep_start_ray_index , sweep_end_ray_index , 
     range , azimuth , elevation , fields) = reader_corrected_by_radar_constant(files[0])

    datetimes['data'][0] = datetime['data']
    datetimes['units'] = datetime['units']
    sweep_numbers[0] = sweep_number['data']
    fixed_angles[0] = fixed_angle['data']
    sweep_start_ray_indexes[0] = sweep_start_ray_index['data']
    sweep_end_ray_indexes[0] = sweep_end_ray_index['data']
    azimuths = azimuth['data']
    elevations = elevation['data']
    Fields = {}
    for key in fields:
        Fields[key] = {}
        Fields[key]['data'] = fields[key]['data']
        Fields[key]['units'] = fields[key]['units']

    for cnt_file in np.arange(1 , num_file):
        (datetime , NULL , NULL , NULL , 
         NULL , NULL , NULL , 
         sweep_number , NULL , fixed_angle , 
         sweep_start_ray_index , sweep_end_ray_index , 
         NULL , azimuth , elevation , fields) = reader_corrected_by_radar_constant(files[cnt_file])

        datetimes['data'][cnt_file] = datetime['data']
        sweep_numbers[cnt_file] = sweep_number['data']
        fixed_angles[cnt_file] = fixed_angle['data']
        sweep_start_ray_indexes[cnt_file] = sweep_end_ray_indexes[cnt_file - 1] + 1
        sweep_end_ray_indexes[cnt_file] = sweep_end_ray_indexes[cnt_file - 1] + sweep_end_ray_index['data'] + 1
        azimuths = np.hstack((azimuths , azimuth['data']))
        elevations = np.hstack((elevations , elevation['data']))
        for key in fields:
            Fields[key]['data'] = np.vstack((Fields[key]['data'] , fields[key]['data']))

    sweep_numbers = {'data': sweep_numbers}
    fixed_angles = {'data': fixed_angles}
    sweep_start_ray_indexes = {'data': sweep_start_ray_indexes}
    sweep_end_ray_indexes = {'data': sweep_end_ray_indexes}
    azimuths = {'data': azimuths}
    elevations = {'data': elevations}

    return (datetimes , metadata , instrument_parameters , scan_type , 
            latitude , longitude , altitude , 
            sweep_numbers , sweep_mode , fixed_angles , 
            sweep_start_ray_indexes , sweep_end_ray_indexes , 
            range , azimuths , elevations , Fields)
