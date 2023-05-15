from CRPS import crps_cost_function

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping


def build_EMOS_network_keras(compile=False, optimizer='SGD', lr=0.1):
    """Build (and maybe compile) EMOS network in keras.

    Args:
        compile: If true, compile model
        optimizer: String of keras optimizer
        lr: learning rate

    Returns:
        model: Keras model
    """
    mean_in = Input(shape=(1,))
    std_in = Input(shape=(1,))
    mean_out = Dense(1, activation='linear')(mean_in)
    std_out = Dense(1, activation='linear')(std_in)
    x = tf.keras.layers.concatenate([mean_out, std_out], axis=1)
    model = Model(inputs=[mean_in, std_in], outputs=x)

    if compile:
        opt = tf.keras.optimizers.__dict__[optimizer](learning_rate = lr)
        model.compile(optimizer=opt, loss=crps_cost_function)
    return model