# Data
import pickle

# Helpful
from tqdm import tqdm

# My Methods
from src.utils.CRPS import *
from src.utils.data_split import *
from src.models.EMOS import *
import data.raw.load_data_raw as ldr
import data.processed.load_data_processed as ldp

# 1. Load Dataset
dat_train_proc = ldp.load_data_all_train_proc_norm()
dat_test_proc = ldp.load_data_all_test_proc_norm()

# 2. Data Split
X_train_var_lead_all, y_train_var_lead_all = split_var_lead(dat_train_proc)
X_test_lead_all, y_test_var_lead_all = split_var_lead(dat_test_proc)

# 3. Calculate Baseline Scores
crps_var_lead_test = crps_var_lead(X_test_lead_all, y_test_var_lead_all)

# 4. Pickle Baseline all Data
with open(
    "/Data/Delong_BA_Data/scores/crps_benchmark_scores/crps_var_lead_test.pkl", "wb"
) as f:  # open a text file
    pickle.dump(crps_var_lead_test, f)  # serialize the list
f.close()

# 5. Take Mean of crps_var_lead_test
crps_var_lead_mean_test = [[], [], [], [], []]
for var in range(5):
    for lead_time in range(31):
        crps_var_lead_mean_test[var].append(crps_var_lead_test[var][lead_time].mean())
        
# 6. Pickle Baseline Mean all Data
with open(
    "/Data/Delong_BA_Data/scores/crps_benchmark_scores/crps_var_lead_mean_test.pkl",
    "wb",
) as f:  # open a text file
    pickle.dump(crps_var_lead_mean_test, f)  # serialize the list
f.close()



