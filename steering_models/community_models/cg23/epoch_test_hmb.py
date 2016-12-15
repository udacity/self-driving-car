# -----------------------------------------------------------------------------------------
# Challenge #2 - epoch_test_hmb.py - run script
# -----------------------------------------------------------------------------------------

'''
Tests the model on the final test set

Original By: dolaameng Revd: cgundling
'''

from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np
import os
from sklearn.cross_validation import KFold
from os import path
from collections import defaultdict
import time
import random

from epoch_data_hmb import *
from epoch_model import *

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Main Program
# ----------------------------------------------------------------------------------------
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Testing Udacity SDC data")
    parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')
    parser.add_argument('--model', type=str, help='model to evaluate, current list: {cnn}')
    parser.add_argument('--resized-image-height', type=int, help='image resizing')
    parser.add_argument('--resized-image-width', type=int, help='image resizing')
    parser.add_argument('--nb-epoch', type=int, help='# of training epoch')
    parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    args = parser.parse_args()

    # Model and data gen args
    # ---------------------------------------------------------------------------------
    dataset_path = args.dataset
    model_name = args.model
    image_size = (args.resized_image_height, args.resized_image_width)
    camera = args.camera
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    weights_path = 'weights_HMB_1.hdf5' # Change to your model weights

    # Data paths
    # ---------------------------------------------------------------------------------
    steering_log = path.join(dataset_path, 'final_example.csv')
    image_log = path.join(dataset_path, 'camera.csv')
    camera_images = dataset_path

    # Model build
    # ---------------------------------------------------------------------------------
    model_builders = {
                     'V3': (build_InceptionV3, preprocess_input_InceptionV3, exact_output) 
                     , 'cnn': (build_cnn, normalize_input, exact_output)}

    if model_name not in model_builders:
        raise ValueError("unsupported model %s" % model_name)
    model_builder, input_processor, output_processor = model_builders[model_name]
    model = model_builder(image_size,weights_path)
    print('model %s built...' % model_name)

    # Testing on the Test Images
    test_generator = data_generator(steering_log=steering_log,
                         image_log=image_log,
                         image_folder=camera_images,
                         camera=camera,
                         batch_size=5614,
                         image_size=image_size,
                         # HMB Test Set
                         timestamp_start=14794254411,
                         timestamp_end=14794257218,
                         shuffle=False,
                         preprocess_input=input_processor,
                         preprocess_output=output_processor)

    print('Made it to Testing')

    test_x, test_y = test_generator.next()
    print('test data shape:', test_x.shape)
    yhat = model.predict(test_x)

    # Use dataframe to write results, calculate RMSE and plot actual vs. predicted
    # ----------------------------------------------------------------------------
    df_test = pd.read_csv('testset/final_example.csv',usecols=['frame_id','steering_angle'],index_col = None)
    df_test['steering_angle'] = yhat
    df_test.to_csv('epoch.csv',index = False)

    plt.figure(figsize = (32, 8))
    plt.plot(yhat, 'b.-', label='predict')
    plt.legend(loc='best')
    plt.title("Test Set Predictions")
    plt.show()
    model_fullname = "%s_%d.png" % (model_name, int(time.time()))
    plt.savefig(model_fullname)

if __name__ == '__main__':
        main()
