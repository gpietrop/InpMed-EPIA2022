import os
import random
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from get_dataset import get_list_model_tensor
from completion import CompletionN
from utils import generate_input_mask
from normalization import Normalization
from mean_pixel_value import MV_pixel
from make_datasets import find_index
from hyperparameter import latitude_interval, longitude_interval, depth_interval, resolution, number_channel

sns.set(context="notebook", style="whitegrid")

epoch_float, lr_float = 25, 0.0001

name_model = "model_step1_ep_1975"

paper_path = os.getcwd() + "/paper_fig"
model_considered = "model2015/" + name_model
path_model = os.getcwd() + "/model/" + model_considered + ".pt"
path_model_float = os.getcwd() + "/result2/" + name_model + "/" + str(epoch_float) + "/" + str(lr_float) + "/model.pt"

if not os.path.exists(path_model_float):
    flag_float = False
else:
    flag_float = True

dict_channel = {"temperature": 0, "salinity": 1, "oxygen": 2, "chla": 3, "ppn": 4}
dict_threshold = {"temperature": 3, "salinity": 10, "oxygen": 50, "chla": 0, "ppn": -30}

for variable in dict_channel:  # list(dict_channel.keys()):
    snaperiod = 25
    print("plotting " + variable + " vertical profile")

    constant_latitude = 111  # 1° of latitude corresponds to 111 km
    constant_longitude = 111  # 1° of latitude corresponds to 111 km
    lat_min, lat_max = latitude_interval
    lon_min, lon_max = longitude_interval
    depth_min, depth_max = depth_interval
    w_res, h_res, d_res = resolution

    w = np.int((lat_max - lat_min) * constant_latitude / w_res + 1) - 2
    h = np.int((lon_max - lon_min) * constant_longitude / h_res + 1)
    d_d = np.int((depth_max - depth_min) / d_res + 1) - 1
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

    path_fig = os.getcwd() + "/analysis_result/profile/"
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    path_fig = os.getcwd() + "/analysis_result/profile/" + name_model[23:] + "/"
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    if flag_float:
        path_fig = path_fig + str(epoch_float) + "_" + str(lr_float) + "/"
        if not os.path.exists(path_fig):
            os.mkdir(path_fig)
    path_fig = path_fig + variable
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)
    path_fig = path_fig + "/std_boxplot/"
    if not os.path.exists(path_fig):
        os.mkdir(path_fig)

    months = ["0" + str(month) for month in range(1, 10)] + [str(month) for month in range(10, 52)]
    stds_phys = [[] for _ in range(0, d)]
    stds_mod = [[] for _ in range(0, d)]
    stds_float = [[] for _ in range(0, d)]

    for month in months[2:4]:  # iteration among months
        if month[-1] == "0":
            month = month[:-1]
        datetime = "2015." + month
        data_tensor = os.getcwd() + "/tensor/model2015_n/datetime_" + str(
            datetime) + ".pt"  # get the data_tensor correspondent to the datetime of emodnet sample to feed the nn (
        # NORMALIZED!!)
        data_tensor = torch.load(data_tensor)

        # TEST ON THE MODEL"S MODEL AND THE FLOAT MODEL WITH SAME HOLE
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
            unkn_model = unkn_model * std_unkn + mean_unkn
            if flag_float:
                unkn_float = unkn_float * std_unkn + mean_unkn

            means_phys.append(torch.mean(unkn_phys))
            means_mod.append(torch.mean(unkn_model))
            if flag_float:
                means_flo.append(torch.mean(unkn_float))

            std_phys.append(torch.std(unkn_phys))
            std_mod.append(torch.std(unkn_model))
            if flag_float:
                std_flo.append(torch.std(unkn_float))

            if torch.mean(unkn_phys) > dict_threshold[variable]:
                stds_phys[depth_index].append(float(torch.std(unkn_phys)))
                stds_mod[depth_index].append(float(torch.std(unkn_model)))

    stds_phys = np.array(stds_phys)
    stds_mod = np.array(stds_mod)
    """
    merging = []
    for j in range(np.shape(stds_phys)[1]):
        phys_line = pd.DataFrame(stds_phys[:, j], columns=["0"]).assign(Trial=str(j))
        mod_line = pd.DataFrame(stds_mod[:, j], columns=["1"]).assign(Trial=str(j))
        line_merge = pd.concat([phys_line, mod_line])
        merging.append(pd.melt(line_merge, id_vars="Trial", var_name="model"))
    """
    sns.set_theme(style="whitegrid")

    for index_week in range(np.shape(stds_phys)[1]):
        fig, ax1 = plt.subplots()
        mdf = [stds_phys[:, index_week], stds_mod[:, index_week]]

        colours = ["w", "w"]
        sns.set_palette(sns.color_palette(colours))

        flierprops = dict(markerfacecolor="black", markersize=3, markeredgecolor="black")

        bplot = sns.boxplot(data=mdf,
                            orient="h",
                            flierprops=flierprops,
                            linewidth=2,
                            # showfliers=False
                            )

        for i, box in enumerate(bplot.artists):
            box.set_edgecolor("black")
            box.set_facecolor("white")

        # iterate over whiskers and median lines
            for j in range(6 * i, 6 * (i + 1)):
                bplot.lines[j].set_color("black")

        for line in ax1.get_lines()[4::12]:
            line.set_color("darkorange")
        for line in ax1.get_lines()[10::12]:
            line.set_color("red")

        # plt.legend(title="Model", loc="upper left", labels=["MedBFM", "Emulator"])

        bplot.set_xlabel("")
        bplot.set_ylabel("week " + str(index_week + 1))
        plt.legend([], [], frameon=False)

        bplot.set_yticklabels([])
        plt.savefig(path_fig + "/" + variable + "_" + str(index_week + 1) + "_std_bp.png")
        if index_week in [0, 9, 19, 29, 39]:
            plt.savefig(paper_path + "/std_bp/" + variable + "_" + str(index_week + 1) + "_std_bp.png")
        plt.close()
