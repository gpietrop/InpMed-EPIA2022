import torch
import netCDF4 as nc
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from hyperparameter import *
from make_datasets import find_index, to_depth, read_date_time_sat

constant_latitude = 111  # 1° of latitude corresponds to 111 km
constant_longitude = 111  # 1° of latitude corresponds to 111 km
float_path = "../FLOAT_BIO/"

path_results = "analysis_result/float_analysis/"

lat_min, lat_max = latitude_interval
lon_min, lon_max = longitude_interval
depth_min, depth_max = depth_interval
year_min, year_max = year_interval
w_res, h_res, d_res = resolution
w = int((lat_max - lat_min) * constant_latitude / w_res + 1)
h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
d = int((depth_max - depth_min) / d_res + 1)

season_parallelepiped = {"winter": torch.zeros(batch, number_channel, d, h, w),
                         "spring": torch.zeros(batch, number_channel, d, h, w),
                         "summer": torch.zeros(batch, number_channel, d, h, w),
                         "autumn": torch.zeros(batch, number_channel, d, h, w), }

months = ["0" + str(month) for month in range(1, 10)] + [str(month) for month in range(10, 52)]

dict_season = {"winter": ["01", "02", "03"],
               "spring": ["04", "05", "06"],
               "summer": ["07", "08", "09"],
               "autumn": ["10", "11", "12"]}

list_data = pd.read_csv(float_path + 'data/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()
list_datetime = pd.read_csv(float_path + 'data/Float_Index.txt', header=None).to_numpy()[:, 3].tolist()

for season in dict_season.keys():
    path_season = path_results + "/" + season
    if not os.path.exists(path_season):
        os.mkdir(path_season)
    for i in range(np.size(list_data)):  # indexing on list_data and list_datetime also

        path_current_float = float_path + "data/" + list_data[i]
        ds = nc.Dataset(path_current_float)

        var_list = []
        for var in ds.variables:
            var_list.append(var)

        datetime = list_datetime[i]
        month = datetime[4:6]
        if month not in dict_season[season]:
            continue
        time = read_date_time_sat(datetime)
        if not year_min < time < year_max:
            continue

        lat = float(ds['LATITUDE'][:].data)  # single value
        lon = float(ds['LONGITUDE'][:].data)  # single value

        lat_index = find_index(lat, latitude_interval, w)
        lon_index = find_index(lon, longitude_interval, h)

        pres_list = ds['PRES'][:].data[0]  # list of value
        depth_list = []
        for pres in pres_list:
            depth_list.append(to_depth(pres, lat))

        temp = ds['TEMP'][:].data[0]  # list of value
        salinity = ds['PSAL'][:].data[0]
        if 'DOXY' in var_list:
            doxy = ds['DOXY'][:].data[0]
        if 'CHLA' in var_list:
            chla = ds['CHLA'][:].data[0]

        if lat_max > lat > lat_min:
            if lon_max > lon > lon_min:
                for depth in depth_list:
                    if depth_max > depth > depth_min:
                        depth_index = find_index(depth, depth_interval, d)
                        channel_index = np.where(depth_list == depth)[0][0]

                        temp_v, salinity_v = temp[channel_index], salinity[channel_index]

                        if -3 < temp_v < 40:
                            season_parallelepiped[season][0, 0, depth_index, lon_index, lat_index] += 1

                        if 2 < salinity_v < 41:
                            season_parallelepiped[season][0, 1, depth_index, lon_index, lat_index] += 1

                        if 'DOXY' in var_list:
                            doxy_v = doxy[channel_index]
                            if -5 < doxy_v < 600:
                                season_parallelepiped[season][0, 2, depth_index, lon_index, lat_index] += 1

                        if 'CHLA' in var_list:
                            chla_v = chla[channel_index]
                            if -5 < chla_v < 600:
                                season_parallelepiped[season][0, 3, depth_index, lon_index, lat_index] += 1

    for depth_index in range(0, d):
        cmap = plt.get_cmap('Greys')
        plt.imshow(season_parallelepiped[season][0, 0, depth_index, :, :], cmap=cmap)
        plt.colorbar()
        plt.savefig(path_season + "/float_distribution_" + str(depth_index) + ".png")
        # plt.show()
        plt.close()
