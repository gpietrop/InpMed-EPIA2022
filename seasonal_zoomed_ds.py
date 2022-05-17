import os
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from get_dataset import get_list_model_tensor
from completion import CompletionN
from utils import generate_input_mask
from normalization import Normalization
from mean_pixel_value import MV_pixel
from make_datasets import find_index
from hyperparameter import latitude_interval, longitude_interval, depth_interval, resolution

sns.set(context='notebook', style='whitegrid')

epoch_float_tot, lr_float = 200, 0.001
epoch_float = 50

kind_of_study = "one_degree_ds"
name_model = "model_step3_ep_800"

model_considered = 'model2015/' + name_model
path_model = os.getcwd() + '/model/' + model_considered + '.pt'
path_model_float = os.getcwd() + '/result2/' + name_model + '/' + str(epoch_float_tot) + '/' + str(lr_float) \
                   + '/model_' + str(epoch_float) + '.pt'

dict_channel = {'temperature': 0, 'salinity': 1, 'oxygen': 2, 'chla': 3, 'ppn': 4}

for variable in list(dict_channel.keys()):
    snaperiod = 25

    constant_latitude = 111  # 1° of latitude corresponds to 111 km
    constant_longitude = 111  # 1° of latitude corresponds to 111 km
    lat_min, lat_max = latitude_interval
    lon_min, lon_max = longitude_interval
    depth_min, depth_max = depth_interval
    w_res, h_res, d_res = resolution

    w = int((lat_max - lat_min) * constant_latitude / w_res + 1) - 2
    h = int((lon_max - lon_min) * constant_longitude / h_res + 1)
    d_d = int((depth_max - depth_min) / d_res + 1) - 1
    d = d_d - 1

    latitude_interval = (lat_min + (lat_max - lat_min) / w, lat_max - (lat_max - lat_min) / w)
    depth_interval = (depth_min + (depth_max - depth_min) / d, depth_max - (depth_max - depth_min) / d)
    depth_interval_d = (depth_min, depth_max - (depth_max - depth_min) / d_d)

    hole_min_d, hole_max_d = 10, 20
    hole_min_h, hole_max_h = 30, 50
    hole_min_w, hole_max_w = 30, 50

    mvp_dataset = get_list_model_tensor()
    mvp_dataset, mean_model, std_model = Normalization(mvp_dataset)
    mean_value_pixel = MV_pixel(mvp_dataset)  # compute the mean of the channel of the training set
    mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, 5, 1, 1, 1))

    model = CompletionN()
    model.load_state_dict(torch.load(path_model))  # network trained only with model information
    model.eval()

    model_float = CompletionN()
    model_float.load_state_dict(torch.load(path_model_float))  # network adjusted with float information
    model_float.eval()

    path_fig = os.getcwd() + '/analysis_result/' + kind_of_study + '/' + name_model[23:] + '/'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    path_fig = path_fig + str(epoch_float) + '_' + str(lr_float) + '/'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    path_fig = path_fig + variable
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)

    months = ["0" + str(month) for month in range(1, 10)] + [str(month) for month in range(10, 52)]
    dict_season = {"winter": months[0:12],
                   "spring": months[12:25],
                   "summer": months[25:38],
                   "autumn": months[38:-1]}

    unkn_phys = torch.zeros([1, 5, 30, 65, 73])
    unkn_model = torch.zeros([1, 5, 29, 65, 73])
    unkn_float = torch.zeros([1, 5, 29, 65, 73])

    for season in dict_season.keys():
        for month in dict_season[season]:  # iteration among months
            number_months = len(dict_season[season])
            if month[-1] == "0":
                month = month[:-1]
            datetime = "2015." + month
            data_tensor = os.getcwd() + '/tensor/model2015_n/datetime_' + str(datetime) + '.pt'
            data_tensor = torch.load(data_tensor)

            # TEST ON THE MODEL'S MODEL AND THE FLOAT MODEL WITH SAME HOLE
            with torch.no_grad():
                training_mask = generate_input_mask(
                    shape=(data_tensor.shape[0], 1, data_tensor.shape[2], data_tensor.shape[3], data_tensor.shape[4]),
                    hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
                data_tensor_mask = data_tensor - data_tensor * training_mask + mean_value_pixel * training_mask
                input = torch.cat((data_tensor_mask, training_mask), dim=1)

                model_result = model(input.float())
                float_result = model_float(input.float())

            mean_unkn = mean_model[0, dict_channel[variable], 0, 0, 0]
            std_unkn = std_model[0, dict_channel[variable], 0, 0, 0]

            unkn_phys_ = data_tensor * std_unkn + mean_unkn
            unkn_model_ = model_result * std_unkn + mean_unkn
            unkn_float_ = float_result * std_unkn + mean_unkn

            unkn_phys += unkn_phys_
            unkn_model += unkn_model_
            unkn_float += unkn_float_

        unkn_phys = unkn_phys / number_months
        unkn_model = unkn_model / number_months
        unkn_float = unkn_float / number_months

        diff_float_model = unkn_model - unkn_float
        diff_float_model[unkn_phys[:, :, :-1, :, :] < 5] = 0

        diff_model_phys = unkn_model - unkn_phys[:, :, :-1, :, :]
        diff_model_phys[unkn_phys[:, :, :-1, :, :] < 5] = 0

        diff_phys_float = unkn_phys[:, :, :-1, :, :] - unkn_float
        diff_phys_float[unkn_phys[:, :, :-1, :, :] < 5] = 0

        path_fm = path_fig + "/float-model/"
        if not os.path.exists(path_fm):
            os.mkdir(path_fm)

        path_pf = path_fig + "/float-phys/"
        if not os.path.exists(path_pf):
            os.mkdir(path_pf)

        path_mf = path_fig + "/model-phys/"
        if not os.path.exists(path_mf):
            os.mkdir(path_mf)

        path_fm_month = path_fm + "/winter/"
        if not os.path.exists(path_fm_month):
            os.mkdir(path_fm_month)

        path_pf_month = path_pf + "/winter/"
        if not os.path.exists(path_pf_month):
            os.mkdir(path_pf_month)

        path_mf_month = path_mf + "/" + season + "/"
        if not os.path.exists(path_mf_month):
            os.mkdir(path_mf_month)

        for depth_index in range(0, 1):  # d # iteration among depth

            if variable == "temperature":
                pf_min, pf_max = -3, 3
                fm_min, fm_max = -3, 3

            lat_range = range(0, 75, 5)
            lon_range = range(0, 70, 5)

            if depth_index == 0:
                cmap = plt.get_cmap('Greys')

                plt.imshow(unkn_phys[0, dict_channel[variable], depth_index, :, :], cmap=cmap)
                for y_line in lon_range:
                    plt.plot([0, 73], [y_line, y_line], lw=2, c="r")
                for x_line in lat_range:
                    plt.plot([x_line, x_line], [0, 65], lw=2, c="r")

                plt.colorbar()
                plt.suptitle("week: " + str(month) + " - depth index: " + str(depth_index))
                plt.title("Discretization grid" + variable)
                plt.savefig(path_fm_month + "/discretization_grid.png")
                plt.savefig(path_pf_month + "/discretization_grid.png")
                plt.savefig(path_mf_month + "/discretization_grid.png")
                # plt.show()
                plt.close()

            lat_range = range(0, 65, 5)
            lon_range = range(0, 75, 5)

            for from_index_lat in lat_range:
                for from_index_lon in lon_range:
                    to_index_lat = from_index_lat + 5
                    to_index_lon = from_index_lon + 5

                    path_fm_range = path_fm_month + "/" + str(from_index_lat) + "_" + str(from_index_lon)
                    if not os.path.exists(path_fm_range):
                        os.mkdir(path_fm_range)
                    path_pf_range = path_pf_month + "/" + str(from_index_lat) + "_" + str(from_index_lon)
                    if not os.path.exists(path_pf_range):
                        os.mkdir(path_pf_range)
                    path_mf_range = path_mf_month + "/" + str(from_index_lat) + "_" + str(from_index_lon)
                    if not os.path.exists(path_mf_range):
                        os.mkdir(path_mf_range)

                    cmap = plt.get_cmap('bwr')
                    plt.imshow(diff_float_model[0, dict_channel[variable], depth_index, from_index_lat:to_index_lat,
                               from_index_lon:to_index_lon], cmap=cmap, vmin=fm_min, vmax=fm_max)
                    plt.colorbar()
                    plt.suptitle("week: " + str(month) + " - depth index: " + str(depth_index))
                    plt.title("CNN+GAN+float - CNN+GAN sec. diff. for " + variable)
                    plt.savefig(
                        path_fm_range + "/fm_" + variable + "_week_" + str(month) + "_depth_" + str(depth_index) + ".png")
                    plt.close()

                    cmap = plt.get_cmap('bwr')
                    plt.imshow(diff_phys_float[0, dict_channel[variable], depth_index, from_index_lat:to_index_lat,
                               from_index_lon:to_index_lon], cmap=cmap, vmin=pf_min, vmax=pf_max)
                    plt.colorbar()
                    plt.suptitle("week: " + str(month) + " - depth index: " + str(depth_index))
                    plt.title("CNN+GAN+float - physical model sec. diff. for " + variable)
                    plt.savefig(
                        path_pf_range + "/pf_" + variable + "_week_" + str(month) + "_depth_" + str(depth_index) + ".png")
                    plt.close()

                    cmap = plt.get_cmap('bwr')
                    plt.imshow(diff_model_phys[0, dict_channel[variable], depth_index, from_index_lat:to_index_lat,
                               from_index_lon:to_index_lon], cmap=cmap, vmin=pf_min,
                               vmax=pf_max)
                    plt.colorbar()
                    plt.suptitle("week: " + str(month) + " - depth index: " + str(depth_index))
                    plt.title("CNN+GAN - physical model sec. diff. for " + variable)
                    plt.savefig(
                        path_mf_range + "/mp_" + variable + "_week_" + str(month) + "_depth_" + str(depth_index) + ".png")
                    plt.close()
