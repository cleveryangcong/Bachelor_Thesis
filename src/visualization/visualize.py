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
    # plt.savefig(
    #     f"/home/dchen/BA_CH_EN/src/visualization/heatmap_t2m_ws10_lead_{lead_time_real}_values.pdf"
    # )
    plt.show()