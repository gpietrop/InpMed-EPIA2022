import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib import pyplot, transforms

from architecture.completion import CompletionN
from utils.utils import MV_pixel
from utils.utils import generate_input_mask
from utils.utils import Normalization
from dataset.get_dataset import *

sns.set(context='notebook', style='white')
matplotlib.rc('font', **{'size': 3, 'weight': 'bold'})

num_channel = number_channel  # 0,1,2,3
name_model = "model_step3_ep_800"

epoch_float, lr_float = 25, 0.0007

path_model = os.getcwd() + '/model/model2015/' + name_model + '.pt'
path_model_float = os.getcwd() + '/result2/' + name_model + '/' + str(epoch_float) + '/' + str(lr_float) + '/model.pt'

dict_channel = {'temperature': 0, 'salinity': 1, 'oxygen': 2, 'chla': 3, "ppn": 4}
dict_threshold = {"temperature": 5, "salinity": 10, "oxygen": 50, "chla": 0, "ppn": -10}
dict_unit = {"temperature": " (degrees °C)", "salinity": " mg/Kg", "oxygen": " mol", "chla": " mg/Kg",
             "ppn": " gC/m^2/yr"}

vmin = {"temperature": 0, "salinity": 0, "oxygen": 0, "chla": 0, "ppn": -2}
vmax = {"temperature": 20, "salinity": 40, "oxygen": 280, "chla": 1, "ppn": 30}

if not os.path.exists(path_model_float):
    flag_float = False
else:
    flag_float = True

path = "paper_fig/" + name_model  # result directory
if not os.path.exists(path):
    os.mkdir(path)
path = "paper_fig/" + name_model + "/map/"  # result directory
if not os.path.exists(path):
    os.mkdir(path)

train_dataset = get_list_model_tensor()

index_testing = -1

train_dataset, _, _ = Normalization(train_dataset)
testing_x = train_dataset[index_testing]  # test on the last element of the list
train_dataset.pop(index_testing)

mean_value_pixel = MV_pixel(train_dataset)  # compute the mean of the channel of the training set
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, num_channel, 1, 1, 1))

hole_min_d, hole_max_d = 10, 20
hole_min_h, hole_max_h = 30, 50
hole_min_w, hole_max_w = 30, 50

for variable in list(dict_channel.keys()):

    path_variable = path + "/" + variable
    if not os.path.exists(path_variable):
        os.mkdir(path_variable)

    snaperiod = 25
    print("plotting " + variable + " map")

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

    months = ["0" + str(month) for month in range(1, 10)] + [str(month) for month in range(10, 52)]

    for month in months:  # iteration among months

        path_month = path_variable + "/" + month
        if not os.path.exists(path_month):
            os.mkdir(path_month)
        if not os.path.exists(path_month + "/deterministic/"):
            os.mkdir(path_month + "/deterministic/")
        if not os.path.exists(path_month + "/EMUMed/"):
            os.mkdir(path_month + "/EMUMed/")
        if flag_float:
            if not os.path.exists(path_month + "/INPMed/"):
                os.mkdir(path_month + "/INPMed/")

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

        mean_tensor = mean_model[0, dict_channel[variable], 0, 0, 0]
        std_tensor = std_model[0, dict_channel[variable], 0, 0, 0]

        for depth_index in range(0, d):  # iteration among depth
            tensor_phys = data_tensor[0, dict_channel[variable], depth_index, :, :]
            tensor_phys = tensor_phys * std_tensor + mean_tensor
            tensor_model = model_result[0, dict_channel[variable], depth_index, :, :]
            tensor_model = tensor_model * std_tensor + mean_tensor
            if flag_float:
                tensor_float = float_result[0, dict_channel[variable], depth_index, :, :]
                tensor_float = tensor_float * std_tensor + mean_tensor

            cmap = plt.get_cmap("Greys")

            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
            cmap = ListedColormap(my_cmap)

            base = pyplot.gca().transData
            rot = transforms.Affine2D().rotate_deg(90)

            plt.imshow(tensor_phys.transpose(0, 1), vmin=vmin[variable], vmax=vmax[variable], cmap=cmap,
                       interpolation='spline16')

            default_x_ticks = np.linspace(0, tensor_phys.size(dim=0) - 3, 9)
            plt.xticks(default_x_ticks, [str(int(el)) + "°" for el in np.linspace(36, 44, 9)])
            default_y_ticks = np.linspace(0, tensor_phys.size(dim=1) - 1, 8)
            plt.yticks(default_y_ticks, [str(int(el)) + "°" for el in np.linspace(9, 2, 8)])

            plt.colorbar()
            plt.savefig(path_month + "/deterministic/deterministic_" + str(depth_index) + ".png")
            plt.close()

            cmap = plt.get_cmap("BuGn")

            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
            cmap = ListedColormap(my_cmap)

            plt.imshow(tensor_model.transpose(0, 1), vmin=vmin[variable], vmax=vmax[variable], cmap=cmap,
                       interpolation='spline16')
            default_x_ticks = np.linspace(0, tensor_model.size(dim=0) - 3, 9)
            plt.xticks(default_x_ticks, [str(int(el)) + "°" for el in np.linspace(36, 44, 9)])
            default_y_ticks = np.linspace(0, tensor_model.size(dim=1) - 1, 8)
            plt.yticks(default_y_ticks, [str(int(el)) + "°" for el in np.linspace(9, 2, 8)])
            plt.colorbar()
            plt.savefig(path_month + "/EMUMed/emumed" + str(depth_index) + ".png")
            plt.close()
