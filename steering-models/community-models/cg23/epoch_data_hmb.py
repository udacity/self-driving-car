# ----------------------------------------------------------------------------------
# Challenge #2 - epoch_data_hmb.py - Data Generator
# ----------------------------------------------------------------------------------

'''
Generates images/steerings for final testing
Original By: dolaameng Revd By: cgundling
'''

from __future__ import print_function

import numpy as np
import pandas as pd
import csv
import random

from collections import defaultdict
from os import path
from scipy.misc import imread, imresize, imsave
from scipy import ndimage
from scipy import misc
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

'''
Generator assumes the testset folder has the following structure:

camera.csv  center/  final_example.csv

Bag files processed by:
[rwightman/udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader)
'''

def read_steerings(steering_log, time_scale):
    steerings = defaultdict(list)
    with open(steering_log) as f:
        for line in f.readlines()[1:]:
            fields = line.split(",")
            nanosecond, angle = int(fields[0]), float(fields[1])
            timestamp = nanosecond / time_scale
            steerings[timestamp].append(angle)
    return steerings


def read_image_stamps(image_log, camera, time_scale):
    timestamps = defaultdict(list)
    with open(image_log) as f:
        for line in f.readlines()[1:]:
            if camera not in line:
                continue
            fields = line.split(",")
            nanosecond = int(fields[1])
            timestamp = nanosecond / time_scale
            timestamps[timestamp].append(nanosecond)
    return timestamps


def read_images(image_folder, camera, ids, image_size):
    prefix = path.join(image_folder, camera)
    imgs = []
    for id in ids:
        img = imread(path.join(prefix, '%d.jpg' % id))
        # Cropping
        crop_img = img[200:,:]
        # Resizing
        img = imresize(crop_img, size=image_size)
        imgs.append(img)
    if len(imgs) < 1:
        print('Error no image at timestamp')
        print(ids)
    img_block = np.stack(imgs, axis=0)
    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))
    return img_block


def normalize_input(x):
    return x / 255.


def exact_output(y):
    return y


def preprocess_input_InceptionV3(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# Data generator (output mean steering angles and image arrays)
# ---------------------------------------------------------------------------------
def data_generator(steering_log, image_log, image_folder,
                   batch_size=32, camera='center', time_factor=10, image_size=0.5,
                   timestamp_start=None, timestamp_end=None, shuffle=False,
                   preprocess_input=normalize_input,
                   preprocess_output=exact_output):
    
    # Constants
    # -----------------------------------------------------------------------------
    minmax = lambda xs: (min(xs), max(xs))
    time_scale = int(1e9) / time_factor

    # Read the image stamps for camera
    # -----------------------------------------------------------------------------
    image_stamps = read_image_stamps(image_log, camera, time_scale)
    
    # Read all steering angles
    # -----------------------------------------------------------------------------
    steerings = read_steerings(steering_log, time_scale)

    # More data exploration stats
    # -----------------------------------------------------------------------------
    print('timestamp range for all steerings: %d, %d' % minmax(steerings.keys()))
    print('timestamp range for all images: %d, %d' % minmax(image_stamps.keys()))
    print('min and max # of steerings per time unit: %d, %d' % minmax(map(len, steerings.values())))
    print('min and max # of images per time unit: %d, %d' % minmax(map(len, image_stamps.values())))
    
    # Generate images and steerings within one time unit
    # -----------------------------------------------------------------------------
    start = min(image_stamps.keys())
    if timestamp_start:
        start = max(start, timestamp_start)
    end = max(image_stamps.keys())
    if timestamp_end:
        end = min(end, timestamp_end)
    print("sampling data from timestamp %d to %d" % (start, end))
 
    # While loop for data generator
    # -----------------------------------------------------------------------------
    i = start
    x_buffer, y_buffer, buffer_size = [], [], 0
    while True:
        if i > end:
            i = start
        # Get images
        images = read_images(image_folder, camera, image_stamps[i], image_size)

        # Mean angle with a time unit
        if steerings[i]:
            angle = np.repeat([np.mean(steerings[i])], images.shape[0])
            x_buffer.append(images)
            y_buffer.append(angle)
            buffer_size += images.shape[0]
            if buffer_size >= batch_size:
                indx = range(buffer_size)
                x = np.concatenate(x_buffer, axis=0)[indx[:batch_size], ...]
                y = np.concatenate(y_buffer, axis=0)[indx[:batch_size], ...]
                x_buffer, y_buffer, buffer_size = [], [], 0
                yield preprocess_input(x.astype(np.float32)), preprocess_output(y)
        if shuffle:
            i = np.random.randint(start, end)
            while i not in image_stamps:
                i = np.random.randint(start, end)
        else:
            i += 1
            while i not in image_stamps:
                i += 1

