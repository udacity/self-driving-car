# -----------------------------------------------------------------------------------------
#  Challenge #2 -  epoch_model.py - Model Structure
# -----------------------------------------------------------------------------------------

'''
build_cnn contains the final model structure for this competition
I also experimented with transfer learning with Inception V3
Original By: dolaameng Revd: cgundling
'''

from keras.models import Model, Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras import backend as K

def build_cnn(image_size=None,weights_path=None):
    image_size = image_size or (128, 128)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )

    img_input = Input(input_shape)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    y = Flatten()(x)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(1)(y)

    model = Model(input=img_input, output=y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')

    if weights_path:
        model.load_weights(weights_path)

    return model


def build_InceptionV3(image_size=None,weights_path=None):
    image_size = image_size or (299, 299)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )
    bottleneck_model = InceptionV3(weights='imagenet',include_top=False, 
                                   input_tensor=Input(input_shape))
    for layer in bottleneck_model.layers:
        layer.trainable = False

    x = bottleneck_model.input
    y = bottleneck_model.output
    # There are different ways to handle the bottleneck output
    y = GlobalAveragePooling2D()(x)
    #y = AveragePooling2D((8, 8), strides=(8, 8))(x)
    #y = Flatten()(y)
    #y = BatchNormalization()(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(1)(y)

    model = Model(input=x, output=y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse')
    return model
