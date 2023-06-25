from numpy import mean
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

# My Methods
from src.utils.CRPS import *  # CRPS metrics
from src.utils.data_split import *  # Splitting data into X and y
from src.utils.drn_make_X_array import *  # Import make train array functions (make_X_array)
from src.models.EMOS import *  # EMOS implementation
from src.models.DRN.DRN_model import *  # DRN implementation
from src.models.EMOS_global.EMOS_global_load_score import *  # Load EMOS_global_scores
from src.models.EMOS_global.EMOS_global_load_model import *  # Load EMOS_global_models
import data.raw.load_data_raw as ldr  # Load raw data
import data.processed.load_data_processed as ldp  # Load processed data normed
import data.processed.load_data_processed_denormed as ldpd  # Load processed data denormed
from src.models.CRPS_baseline.CRPS_load import *  # Load CRPS scores

# Your function definitions remain the same
# ...

# Function to build the model with random initial weights
def build_emb_model_random(
    n_features,
    n_outputs,
    hidden_nodes,
    emb_size,
    max_id,
    compile=False,
    optimizer="Adam",
    lr=0.01,
    loss=crps_cost_function,
    activation="relu",
    reg=None,
):
    keras.backend.clear_session()
    init = RandomNormal(mean=0., stddev=1.) # New line: random initializer with mean=0 and stddev=1
    if type(hidden_nodes) is not list:
        hidden_nodes = [hidden_nodes]

    features_in = Input(shape=(n_features,))
    id_in = Input(shape=(1,))
    emb = Embedding(max_id + 1, emb_size)(id_in)
    emb = Flatten()(emb)
    x = Concatenate()([features_in, emb])
    for h in hidden_nodes:
        x = Dense(h, activation=activation, kernel_regularizer=reg, kernel_initializer=init)(x)
    x = Dense(n_outputs, activation="linear", kernel_regularizer=reg, kernel_initializer=init)(x)
    model = Model(inputs=[features_in, id_in], outputs=x)

    if compile:
        opt = keras.optimizers.__dict__[optimizer](learning_rate=lr)
        model.compile(optimizer=opt, loss=loss)
    return model