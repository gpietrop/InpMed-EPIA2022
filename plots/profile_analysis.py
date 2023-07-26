import os
import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.get_dataset import get_list_model_tensor
from architecture.completion import CompletionN
from utils.utils import generate_input_mask
from utils.utils import Normalization
from utils.utils import MV_pixel
from hyperparameter import latitude_interval, longitude_interval, depth_interval, resolution, number_channel

sns.set(context='notebook', style='whitegrid')
matplotlib.rc('font', **{'size': 8, 'weight': 'bold'})

epoch_float_tot, lr_float = 200, 0.001
epoch_float = 100

name_model = "model_step3_ep_800"
# name_model = "phase3_ep_875"

paper_path = os.getcwd() + "/paper_fig/" + name_model
if not os.path.exists(paper_path):
    os.mkdir(paper_path)
if not os.path.exists(paper_path + "/mean+std/"):
    os.mkdir(paper_path + "/mean+std/")
model_considered = 'model2015/' + name_model
path_model = os.getcwd() + '/model/' + model_considered + '.pt'
path_model_float = os.getcwd() + '/result2/' + name_model + '/' + str(epoch_float_tot) + '/' + str(lr_float) \
                   + '/model_' + str(epoch_float) + '.pt'

if not os.path.exists(path_model_float):
    flag_float = False
    print(path_model_float)
else:
    flag_float = True

dict_channel = {'temperature': 0, 'salinity': 1, 'oxygen': 2, 'chla': 3, "ppn": 4}
dict_threshold = {"temperature": 3, "salinity": 10, "oxygen": 50, "chla": 0, "ppn": -30}
dict_unit = {"temperature": " degrees °C", "salinity": " mg/Kg", "oxygen": " mol", "chla": " mg/Kg",
             "ppn": " gC/m^2/yr"}
dict_limit_plot = {"temperature": [10, 25], "salinity": [32, 45], "oxygen": [150, 300], "chla": [-0.05, 0.6],
                   "ppn": [-0.5, 10]}

for variable in ["ppn"]:
    snaperiod = 25
    print("plotting " + variable + " vertical profile")

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
    mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, number_channel, 1, 1, 1))

    model = CompletionN()
    model.load_state_dict(torch.load(path_model))  # network trained only with model information
    model.eval()

    if flag_float:
        model_float = CompletionN()
        model_float.load_state_dict(torch.load(path_model_float))  # network adjusted with float information
        model_float.eval()

    path_fig = os.getcwd() + '/analysis_result/profile/'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    path_fig = os.getcwd() + '/analysis_result/profile/' + name_model + '/'
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    if flag_float:
        path_fig = path_fig + str(epoch_float) + '_' + str(lr_float) + '/'
        if not os.path.exists(path_fig):
            os.mkdir(path_fig)
    path_fig = path_fig + variable
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    if not os.path.exists(path_fig + '/mean/'):
        os.mkdir(path_fig + '/mean/')
    if not os.path.exists(path_fig + '/std/'):
        os.mkdir(path_fig + '/std/')
    if not os.path.exists(path_fig + '/mean+std/'):
        os.mkdir(path_fig + '/mean+std/')

    months = ["0" + str(month) for month in range(1, 10)] + [str(month) for month in range(10, 52)]

    for month in months:  # iteration among months
        if month[-1] == "0":
            month = month[:-1]
            # month = "12"
        datetime = "2015." + month
        data_tensor = os.getcwd() + '/tensor/model2015_n/datetime_' + str(
            datetime) + '.pt'  # get the data_tensor correspondent to the datetime of emodnet sample to feed the nn (
        # NORMALIZED!!)
        data_tensor = torch.load(data_tensor)

        # TEST ON THE MODEL'S MODEL AND THE FLOAT MODEL WITH SAME HOLE
        with torch.no_grad():
            training_mask = generate_input_mask(
                shape=(data_tensor.shape[0], 1, data_tensor.shape[2], data_tensor.shape[3], data_tensor.shape[4]),
                hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
            data_tensor_mask = data_tensor - data_tensor * training_mask + mean_value_pixel * training_mask
            input = torch.cat((data_tensor_mask, training_mask), dim=1)

            model_result = model(input.float())
            if flag_float:
                float_result = model_float(input.float())

        mean_unkn = mean_model[0, dict_channel[variable], 0, 0, 0]
        std_unkn = std_model[0, dict_channel[variable], 0, 0, 0]

        means_phys, means_mod = [], []
        if flag_float:
            means_flo = []
        std_phys, std_mod = [], []
        if flag_float:
            std_flo = []

        for depth_index in range(0, d):  # iteration among depth
            unkn_phys = data_tensor[:, dict_channel[variable], depth_index, :, :]
            unkn_model = model_result[:, dict_channel[variable], depth_index, :, :]
            if flag_float:
                unkn_float = float_result[:, dict_channel[variable], depth_index, :, :]

            unkn_phys = unkn_phys * std_unkn + mean_unkn
            unkn_phys[unkn_phys < dict_threshold[variable]] = torch.nan

            unkn_model = unkn_model * std_unkn + mean_unkn
            unkn_model[unkn_model < dict_threshold[variable]] = torch.nan

            if flag_float:
                unkn_float = unkn_float * std_unkn + mean_unkn
                unkn_float[unkn_float < dict_threshold[variable]] = torch.nan
                if depth_index > 8:
                    unkn_float[unkn_float < 0.3] = 0

            means_phys.append(torch.nanmean(unkn_phys))
            means_mod.append(torch.nanmean(unkn_model))
            if flag_float:
                means_flo.append(torch.nanmean(unkn_float))

            std_phys.append(np.nanstd(unkn_phys.numpy()))
            std_mod.append(np.nanstd(unkn_model.numpy()))
            if flag_float:
                std_flo.append(np.nanstd(unkn_float.numpy()))

        if flag_float:
            zip_result = zip(means_mod, std_mod, means_flo, std_flo, means_phys, std_phys)
            zip_result = [x for x in zip_result if x[4] > dict_threshold[variable]]
            if zip_result:
                means_mod, std_mod, means_flo, std_flo, means_phys, std_phys = zip(*zip_result)
            else:
                continue

        else:
            zip_result = zip(means_mod, std_mod, means_phys, std_phys)
            zip_result = [x for x in zip_result if x[2] > dict_threshold[variable]]
            if zip_result:
                means_mod, std_mod, means_phys, std_phys = zip(*zip_result)
            else:
                continue

        mk_size = 6
        ls = '--'
        lw = 1
        color_phys, mk_phys = "slategray", "v"
        color_model, mk_model = "orangered", "o"
        color_float, mk_float = "forestgreen", "*"

        # MEAN
        # depth_val = range(0, 600, 20)[1:]
        depth_val = np.linspace(0, 600, len(means_phys))
        plt.plot(means_phys[::-1],
                 depth_val[::-1],
                 color=color_phys,
                 linestyle=ls,
                 linewidth=lw,
                 marker=mk_phys,
                 markersize=mk_size,
                 markerfacecolor="w",
                 alpha=0.8)
        plt.plot(means_mod[::-1],
                 depth_val[::-1],
                 color=color_model,
                 linestyle=ls,
                 linewidth=lw,
                 marker=mk_model,
                 markersize=mk_size,
                 markerfacecolor="w",
                 alpha=0.8)
        if flag_float:
            plt.plot(means_flo[::-1],
                     depth_val[::-1],
                     color=color_float,
                     linestyle=ls,
                     linewidth=lw,
                     marker=mk_float,
                     markersize=mk_size,
                     markerfacecolor="w",
                     alpha=0.8)
            plt.legend(["MedBFM", "EmuMed", "InpMed"], prop={'size': 8})
        else:
            plt.legend(["MedBFM", "EmuMed"], prop={'size': 8})

        plt.gca().invert_yaxis()
        plt.xlabel(variable + dict_unit[variable])
        plt.ylabel("depth (m)")
        plt.savefig(path_fig + '/mean/' + variable + '_pro_mean_' + month + '.png')
        plt.close()

        # STD
        plt.plot(std_phys[::-1],
                 depth_val[::-1],
                 color=color_phys,
                 linestyle=ls,
                 linewidth=lw,
                 marker=mk_phys,
                 markersize=mk_size,
                 markerfacecolor="w",
                 alpha=0.8)
        plt.plot(std_mod[::-1],
                 depth_val[::-1],
                 color=color_model,
                 linestyle=ls,
                 linewidth=lw,
                 marker=mk_model,
                 markersize=mk_size,
                 markerfacecolor="w",
                 alpha=0.8)
        if flag_float:
            plt.plot(std_flo[::-1],
                     depth_val[::-1],
                     color=color_float,
                     linestyle=ls,
                     linewidth=lw,
                     marker=mk_float,
                     markersize=mk_size,
                     markerfacecolor="w",
                     alpha=0.8)
            plt.legend(["MedBFM", "EmuMed", "InpMed"], prop={'size': 8})
        else:
            plt.legend(["MedBFM", "EmuMed"], prop={'size': 8})

        plt.gca().invert_yaxis()
        plt.xlabel(variable + dict_unit[variable])
        plt.ylabel("depth (m)")
        plt.savefig(path_fig + '/std/' + variable + '_pro_std_' + month + '.png')
        plt.close()

        # MEAN + STD
        plt.plot(means_phys[::-1],
                 depth_val[::-1],
                 color=color_phys,
                 linestyle=ls,
                 linewidth=lw,
                 # markerfacecolor="w",
                 alpha=0.8,
                 label="MedBFM")
        plt.plot(means_phys[::-1],
                 depth_val[::-1],
                 color=color_phys,
                 marker=mk_phys,
                 markersize=mk_size,
                 # markerfacecolor="w",
                 alpha=0.4)

        plt.plot(means_mod[::-1],
                 depth_val[::-1],
                 color=color_model,
                 linestyle=ls,
                 linewidth=lw,
                 # markerfacecolor="w",
                 alpha=0.8,
                 label='EmuMed')
        plt.plot(means_mod[::-1],
                 depth_val[::-1],
                 color=color_model,
                 marker=mk_model,
                 markersize=mk_size,
                 # markerfacecolor="w",
                 alpha=0.4)

        if flag_float:
            plt.plot(means_flo[::-1],
                     depth_val[::-1],
                     color=color_float,
                     linestyle=ls,
                     linewidth=lw,
                     alpha=0.8,
                     label="InpMed")
            plt.plot(means_flo[::-1],
                     depth_val[::-1],
                     color=color_float,
                     marker=mk_float,
                     markersize=mk_size,
                     # markerfacecolor="w",
                     alpha=0.4)
            # plt.fill_betweenx(depth_val[::-1],
            #                  (np.array(means_flo) - np.array(std_flo) / 2)[::-1],
            #                  (np.array(means_flo) + np.array(std_flo) / 2)[::-1],
            #                  color=color_float,
            #                  alpha=0.2
            #                  )
            plt.legend(prop={'size': 8})
        else:
            plt.legend(prop={'size': 8})

        plt.gca().invert_yaxis()
        plt.xlim(dict_limit_plot[variable])
        plt.xlabel(variable + dict_unit[variable])
        plt.ylabel("depth (m)")
        plt.savefig(path_fig + '/mean+std/' + variable + '_pro_mean_std_' + month + '.png')
        if month in ["2", "35"]:
            plt.savefig(paper_path + '/mean+std/' + variable + '_pro_mean_std_' + month + '.png')
        plt.close()
