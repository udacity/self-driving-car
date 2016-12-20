# adapted from https://github.com/fchollet/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa

from keras import backend as K
from keras.utils.np_utils import convert_kernel
import tensorflow as tf

from config import TestConfig

ops = []
for layer in model.layers:
   if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
      original_w = K.get_value(layer.W)
      converted_w = convert_kernel(original_w)
      ops.append(tf.assign(layer.W, converted_w).op)
      
config = TestConfig()

model.load_weights(config.model_path)
K.get_session().run(ops)
model.save_weights(config.model_path.replace(".hdf5", "_tensorflow.hdf5"))
