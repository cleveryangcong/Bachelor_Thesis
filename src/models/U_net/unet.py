# Import necessary modules from Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, BatchNormalization, Dropout, Cropping2D
from tensorflow.keras.regularizers import l2

# Import xarray for handling N-D labeled arrays
import xarray as xr
xr.set_options(display_style='text')  # Set the display style of xarray datasets

# Import Python's warnings module to control warning messages
import warnings
warnings.simplefilter("ignore")  # Suppress warning messages


class Unet:
    # Initialize the class
    def __init__(self, v, train_patches, weighted_loss=False):
        """
        Constructor for the Unet model class.

        Args:
        v (str): String parameter controlling learning rate, decay rate, and use of delayed early stopping. 
                 If 'v' is 'tp', learning rate = 0.001, decay rate = 0.005, and delayed early stopping is used.
                 Otherwise, learning rate = 1e-4, decay rate = 0, and delayed early stopping is not used.
        train_patches (bool): Boolean indicating whether the model is trained with patches. If True, input_dims = 32,
                              output_dims = 24. Otherwise, input_dims = output_dims = 0 (uses whole image for training)
        weighted_loss (bool): Boolean indicating whether the model's loss function should be weighted.
        """
        
        # Set model architecture to U-Net and set loss function type
        self.train_patches = train_patches
        self.model_architecture = 'unet'
        self.weighted_loss = weighted_loss
        
        # Initialize parameters related to input/preprocessing
        # Dimensions of input and output depend on whether model is trained with patches
        if self.train_patches == False:
            self.input_dims = 0
            self.output_dims = 0
        else:
            self.input_dims = 32
            self.output_dims = self.input_dims - 8
            self.patch_stride = 12
            self.patch_na = 4 / 8
        self.n_bins = 3
        self.region = 'europe'  # 'europe'

        # params for model architecture
        self.filters = 2
        self.apool = True  # choose between average and max pooling, True = average
        self.n_blocks = 3  # 4  # 5
        self.bn = True  # batch normalization
        self.ct_kernel = (3, 3)  # (2, 2)
        self.ct_stride = (2, 2)  # (2, 2)

        # params related to model training
        self.optimizer_str = 'adam'  # Optimizer for model training
        self.call_back = True  # should early stopping be used?

        # Learning rate, decay rate, and early stopping depend on the value of 'v'
        if v == 'tp':
            self.learn_rate = 0.001
            self.decay_rate = 0.005
            self.delayed_early_stop = True
        else:
            self.learn_rate = 1e-4
            self.decay_rate = 0
            self.delayed_early_stop = False

        # Batch size, number of epochs, patience, and start epoch for early stopping 
        # depend on whether the model is trained with patches
        if self.train_patches == True:
            self.bs = 32
            self.ep = 20
            self.patience = 3  # for callback
            self.start_epoch = 2  # epoch to start with early stopping
        else:  # global unet
            self.bs = 16
            self.ep = 50  # 20
            self.patience = 10  # for callback
            self.start_epoch = 5
            if self.call_back == False:
                self.ep = 30

    def build_model(self, dg_train_shape, dg_train_weight_target=None):
        """
        This function builds a U-Net model given the shape of training data and optional weights for the loss function.

        Args:
            dg_train_shape (tuple): A tuple that specifies the shape of the training data.
            dg_train_weight_target (tuple): A tuple containing arrays of the same shape as the training target 
                                             data and the weight for each pixel in the target. 
                                             Default is None, in which case no weighting is used for the loss.

        Returns:
            cnn (Model): A Keras model object representing the U-Net.
        """

        # Define the shape of the input tensor
        inp_imgs = Input(shape=(dg_train_shape[1], dg_train_shape[2], dg_train_shape[3],))

        # Save input tensor as c0
        c0 = inp_imgs

        # Encoder / contracting path: series of convolutional and pooling layers
        p1, c1 = down(c0, self.filters*4, activation='elu', padding='same', bn=self.bn, apool=self.apool)  # 16
        p2, c2 = down(p1, self.filters*8, activation='elu', padding='same', bn=self.bn, apool=self.apool)  # 8
        p3, c3 = down(p2, self.filters*16, activation='elu', padding='same', bn=self.bn, apool=self.apool)  # 4
        # If n_blocks is 4 or greater, add another layer
        p4, c4 = down(p3, self.filters*32, activation='elu', padding='same', bn=self.bn, apool=self.apool) if (self.n_blocks >= 4) else [p3, c3]
        # If n_blocks is 5 or greater, add another layer
        p5, c5 = down(p4, self.filters*64, activation='elu', padding='same', bn=self.bn, apool=self.apool) if (self.n_blocks >= 5) else [p4, c4]

        # Bottleneck: two convolution layers
        cb = Conv2D(self.filters*4*2**self.n_blocks, (3, 3), activation='elu', padding='same')(p5)
        cb = Conv2D(self.filters*4*2**self.n_blocks, (3, 3), activation='elu', padding='same')(cb)
        # Add batch normalization if specified
        cb = BatchNormalization()(cb) if self.bn else cb

        # Decoder / expanding path: series of convolutional and transpose convolutional layers to upsample back to the original image size
        u5 = up(cb, c5, self.filters*64, self.ct_kernel, self.ct_stride, activation='elu', padding='same', bn=self.bn) if (self.n_blocks >=5 ) else cb
        u4 = up(u5, c4, self.filters*32, self.ct_kernel, self.ct_stride, activation='elu', padding='same', bn=self.bn) if (self.n_blocks >=4 ) else cb
        u3 = up(u4, c3, self.filters*16, self.ct_kernel, self.ct_stride, activation='elu', padding='same', bn=self.bn)
        u2 = up(u3, c2, self.filters*8, self.ct_kernel, self.ct_stride, activation='elu', padding='same', bn=self.bn)
        u1 = up(u2, c1, self.filters*4, self.ct_kernel, self.ct_stride, activation='elu', padding='same', bn=False)  # no normalization directly before softmax

        # Apply a softmax activation function to the last layer
        out = Conv2D(3, (1, 1), activation='softmax')(u1)

        # Crop the output to remove patch edges, depending on the region and whether training is done on patches
        if self.train_patches == True:
            out = Cropping2D(cropping=((4, 4), (4, 4)))(out)
        else:
            if self.region == 'europe':
                out = Cropping2D(cropping=((8, 8), (8, 8)))(out)
            if self.region == 'global':
                out = Cropping2D(cropping=((8, 8), (4, 3)))(out)

        # If training on patches and using a weighted loss, build the model with additional inputs for the weights and targets
        if (self.train_patches == True) & (self.weighted_loss == True):
            weight_shape = dg_train_weight_target[0]
            weights = Input(shape=(weight_shape[1], weight_shape[2],))
            target_shape = dg_train_weight_target[1]
            target = Input(shape=(target_shape[1], target_shape[2], target_shape[3],))
            inputs = [inp_imgs]

            cnn = Model(inputs=[inputs] + [weights, target], outputs=out)

            cnn.target = target
            cnn.weight_mask = weights
            cnn.out = out
        else:
            cnn = Model(inputs=[inp_imgs], outputs=out)

        # Return the final model
        return cnn



def down(c, filters, activation='elu', padding='same', lamda=0,
         dropout_rate=0, bn=True, apool=True):
    """
    Constructs a level of the contracting path in the U-Net.

    Args:
        c (tensor): Input tensor.
        filters (int): Number of filters for the convolutional layers.
        activation (str): Activation function to use. Defaults to 'elu'.
        padding (str): Padding to use in the convolutional layers. Defaults to 'same'.
        lamda (float): Regularization factor. Defaults to 0.
        dropout_rate (float): Dropout rate. Defaults to 0.
        bn (bool): If True, applies batch normalization. Defaults to True.
        apool (bool): If True, applies average pooling, else applies max pooling. Defaults to True.

    Returns:
        p (tensor): Output tensor after applying pooling.
        c (tensor): Output tensor after applying convolutions and possibly batch normalization.
    """
    # Apply two convolutions with dropout in between
    c = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(c)
    c = Dropout(dropout_rate)(c)
    c = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(c)

    # Apply batch normalization if specified
    c = BatchNormalization()(c) if bn else c

    # Apply either average or max pooling
    p = AveragePooling2D((2, 2))(c) if apool else MaxPooling2D((2, 2))(c)

    return p, c



def up(u, c, filters, ct_kernel, ct_stride, activation='elu',
       padding='same', lamda=0, dropout_rate=0, bn=True):
    """
    Constructs a level of the expansive path in the U-Net.

    Args:
        u (tensor): Input tensor from lower level.
        c (tensor): Input tensor from corresponding level in contracting path.
        filters (int): Number of filters for the convolutional layers.
        ct_kernel (int): Kernel size for the transposed convolution.
        ct_stride (int): Stride for the transposed convolution.
        activation (str): Activation function to use. Defaults to 'elu'.
        padding (str): Padding to use in the convolutional layers. Defaults to 'same'.
        lamda (float): Regularization factor. Defaults to 0.
        dropout_rate (float): Dropout rate. Defaults to 0.
        bn (bool): If True, applies batch normalization. Defaults to True.

    Returns:
        u (tensor): Output tensor after applying convolutions and possibly batch normalization.
    """
    # Apply a transposed convolution to upsample
    u = Conv2DTranspose(filters, ct_kernel, strides=ct_stride, padding=padding,
                        kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)

    # Concatenate with the corresponding tensor from the contracting path
    u = Concatenate()([c, u])

    # Apply two convolutions with dropout in between
    u = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)
    u = Dropout(dropout_rate)(u)
    u = Conv2D(filters, (3, 3), activation=activation, padding=padding,
               kernel_regularizer=l2(lamda), bias_regularizer=l2(lamda))(u)

    # Apply batch normalization if specified
    u = BatchNormalization()(u) if bn else u

    return u