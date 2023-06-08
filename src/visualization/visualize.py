# Code for visulization of figures

# Basics
import tensorflow as tf
import xarray as xr
import random

# Helpful
import tqdm

# Visualization
import matplotlib.pyplot as plt

# My Methods
from src.utils.CRPS import *  # CRPS metrics
from src.utils.data_split import *  # Splitting data into X and y
from src.utils.drn_make_X_array import *  # Import make train array functions (make_X_array)
from src.models.EMOS import *  # EMOS implementation
from src.models.DRN.DRN_model import *  # DRN implementation
from src.models.EMOS_global.EMOS_global_load_score import *  # Load EMOS_global_scores
from src.models.EMOS_global.EMOS_global_load_model import *  # Load EMOS_global_models
from src.models.EMOS_local.EMOS_local_load_score import *  # Load EMOS_local_scores
from src.models.EMOS_local.EMOS_local_load_model import *  # Load EMOS_local_models
import data.raw.load_data_raw as ldr  # Load raw data
import data.processed.load_data_processed as ldp  # Load processed data normed
import data.processed.load_data_processed_denormed as ldpd  # Load processed data denormed
from src.models.CRPS_baseline.CRPS_load import *  # Load CRPS scores



# 1. Heatmaps of predictions and ground truths for t2m and ws10
def heatmap_t2m_ws10_lead_score(lead_time):
    lead_time_real = lead_time

    t2m_preds = dat_train_denorm[2].t2m_train.isel(
        lead_time=lead_time_real, mean_std=0, forecast_date=ran_forecast_date
    )
    t2m_truth = dat_train_denorm[2].t2m_truth.isel(
        lead_time=lead_time_real, forecast_date=ran_forecast_date
    )
    ws10_preds = dat_train_denorm[5].ws10_train.isel(
        lead_time=lead_time_real, mean_std=0, forecast_date=ran_forecast_date
    )
    ws10_truth = dat_train_denorm[5].ws10_truth.isel(
        lead_time=lead_time_real, forecast_date=ran_forecast_date
    )


    fig = plt.figure(figsize=(20, 20))

    # t2m_preds
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(t2m_preds.values, cmap="inferno")
    ax1.set_title(f"Colormap temperature predictions, lead_time: {lead_time_real}")
    fig.colorbar(im1, ax=ax1, shrink=0.5).set_label("Temperature in Kelvin")

    # t2m_truth
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(t2m_truth.values, cmap="inferno")
    ax2.set_title("Colormap temperature truth")
    fig.colorbar(im2, ax=ax2, shrink=0.5).set_label("Temperature in Kelvin")

    # ws10_preds
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(ws10_preds.values, cmap="viridis")
    ax3.set_title(f"Colormap ws10 predictions, lead_time: {lead_time_real}")
    fig.colorbar(im3, ax=ax3, shrink=0.5).set_label("Wind Speed in m/s")

    # ws10_truth
    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(ws10_truth.values, cmap="viridis")
    ax4.set_title("Colormap ws10 truth")
    fig.colorbar(im4, ax=ax4, shrink=0.5).set_label("Wind Speed in m/s")

    plt.tight_layout()
    plt.savefig(
        f"/home/dchen/BA_CH_EN/reports/figures/heatmap_t2m_ws10_lead_{lead_time_real}_values.pdf"
    )
    plt.show()
# Set up values for all functions below
var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
random.seed(7)
# setup:
ran_lat = random.randint(0, 119)
ran_lon = random.randint(0, 129)
ran_forecast_date = random.randint(0, 356)
ran_lead_time = random.randint(0, 30)
lead_times = [1, 15, 30]  # Lead times to plot boxplot with

    # Raw Data:
dat_raw = ldr.load_data_raw()

# Processese Normed Data
dat_train_norm = ldp.load_data_all_train_proc_norm()
dat_test_norm = ldp.load_data_all_test_proc_norm()

# Processese Denormed Data
dat_train_denorm = ldpd.load_data_all_train_proc_denorm()
dat_test_denorm = ldpd.load_data_all_test_proc_denorm()

dat_arr_X = []
dat_arr_y = []

for i in range(5):
    dat_arr_X.append(dat_raw[0].predictions.isel(var=i))
    dat_arr_y.append(dat_raw[0].ground_truth.isel(var=i))
    
def line_plots_lead_value_variables():

    # Load means and stds
    means = np.load("/mnt/sda/Data2/fourcastnet/data/stats_v0/global_means.npy").flatten()[
        [0, 1, 2, 5, 14]
    ]
    stds = np.load("/mnt/sda/Data2/fourcastnet/data/stats_v0/global_stds.npy").flatten()[
        [0, 1, 2, 5, 14]
    ]

    # Calculate ws10
    u10_X = (
        dat_arr_X[0].isel(forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon)
        * stds[0]
        + means[0]
    )
    v10_X = (
        dat_arr_X[1].isel(forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon)
        * stds[1]
        + means[1]
    )
    u10_y = (
        dat_arr_y[0].isel(forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon)
        * stds[0]
        + means[0]
    )
    v10_y = (
        dat_arr_y[1].isel(forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon)
        * stds[1]
        + means[1]
    )

    ws10_X = np.sqrt(u10_X ** 2 + v10_X ** 2)
    ws10_y = np.sqrt(u10_y ** 2 + v10_y ** 2)
    
    
    labels = ['Wind Speed m/s','Wind Speed m/s', 'Temperature Kelvin', 'Temperature Kelvin', 'Geopotential m', 'Wind Speed m/s']
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 8))

    # Plot for the variable in position 2 and ws10
    for i, index in enumerate([2]):
        (dat_arr_X[index].isel(forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon) * stds[index] + means[index]).plot(
            x="lead_time", hue="ens", add_legend=False, ax=axs[i], color="black", alpha=0.5
        )
        (dat_arr_y[index].isel(forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon) * stds[index] + means[index]).plot(
            x="lead_time", ax=axs[i], color="red"
        )
        axs[i].set_title(var_names[index] + " - lead time - ensemble values")
        axs[i].set_ylabel(labels[index])

    # Plot for ws10
    ws10_X.plot(x="lead_time", hue="ens", add_legend=False, ax=axs[1], color="black", alpha=0.5)
    ws10_y.plot(x="lead_time", ax=axs[1], color="red")
    axs[1].set_title("ws10 - lead time - ensemble values")
    axs[1].set_ylabel(labels[5]) 

    plt.tight_layout()
    plt.savefig('/home/dchen/BA_CH_EN/reports/figures/line_plot_lead_value_t2m_ws10.pdf')  # save plot for variables in position 2 and ws10
    plt.show()


    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10, 12))

    # Plot for the remaining variables, excluding u10 and v10 as they were used to compute ws10
    for i, index in enumerate([0, 3, 4]):
        (dat_arr_X[index].isel(forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon) * stds[index] + means[index]).plot(
            x="lead_time", hue="ens", add_legend=False, ax=axs[i], color="black", alpha=0.5
        )
        (dat_arr_y[index].isel(forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon) * stds[index] + means[index]).plot(
            x="lead_time", ax=axs[i], color="red"
        )
        axs[i].set_title(var_names[index] + " - lead time - ensemble values")
        axs[i].set_ylabel(labels[index])
    plt.tight_layout()
    plt.savefig('/home/dchen/BA_CH_EN/reports/figures/line_plot_lead_value_rest.pdf')  # save plot for remaining variables
    plt.show()

    
def line_plot_lead_ensstd_all():
    labels = [
    "Wind Speed m/s",
    "Wind Speed m/s",
    "Temperature Kelvin",
    "Temperature Kelvin",
    "Geopotential m",
    "Wind Speed m/s",
    ]
    fig, axs = plt.subplots(ncols=1, nrows=6, figsize=(10, 20))
    for i in range(6):
        dat_train_denorm[i][var_names[i] + "_train"].isel(
            forecast_date=ran_forecast_date, lat=ran_lat, lon=ran_lon, mean_std=1
        ).plot(x="lead_time", ax=axs[i])
        axs[i].set_title(var_names[i] + " - lead time - ensemble std")
        axs[i].set_ylabel(labels[i])
    plt.tight_layout()
    plt.savefig(
        "/home/dchen/BA_CH_EN/reports/figures/line_plot_lead_ens-std_all.pdf"
    )  # save plot for remaining variables
    plt.show()