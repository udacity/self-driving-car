# ----------------------------------------------------------------------------------
# Challenge #2 - epoch_data.py - Data Generator
# ----------------------------------------------------------------------------------

'''
Calculates stats on dataset, performs image augmentation,
and generates images/steerings to train, validate and test model
Original By: dolaameng Revd By: cgundling
*Note that steering shifts for left/right cameras is currently 
commented out. Change line 323 to add back in.
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
Generator assumes the dataset folder has the following structure:

camera.csv  center/  left/  right/  steering.csv

Bag files processed by:
[rwightman/udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader)
'''

# The following are all the functions used for data processing/augmentation
# ----------------------------------------------------------------------------------

def data_kfold(steering_log):
    df_steer = pd.read_csv(steering_log,usecols=['timestamp','angle'],index_col = False)

    # Mod the timestamp data for k-fold val splitting
    # ----------------------------------------------------------------------------------
    time_factor = 10
    time_scale = int(1e9) / time_factor
    df_steer['time_mod'] = df_steer['timestamp'].astype(int) / time_scale
    
    # Setup data for k-fold the steering data
    angle = np.zeros((df_steer.shape[0],1))
    time = np.zeros((df_steer.shape[0],1))

    angle[:,0] = df_steer['angle'].values
    time[:,0] = df_steer['time_mod'].values.astype(int)
    data = np.append(time,angle,axis=1)
    
    return data


def kfold_cross(data):
    # Find the timebreaks for k-fold
    timebreak_end = []  # Store ending timestamps of timebreaks
    timebreak_start = [14774295160]

    for i in range(1,data.shape[0]):
        if data[i,0] != data[i-1,0] and data[i,0] != (data[i-1,0] + 1):
            timebreak_end.append(int(data[i-1,0]))
            timebreak_start.append(int(data[i,0]))
    timebreak_end.append(int(data[-1:,0]))
    unique_times = range(1,76)  # 75 is # of timesections in Dataset1 and Dataset 2 combined
    list_r = [6,9,15,29,38,42,47,52,58,68,69,70] # Remove these timesections from training based on analysis
    for i in list_r:
        unique_times.remove(i)

    return timebreak_start, timebreak_end, unique_times


def combine_folds(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= (nfolds)
    return a.tolist()


def read_steerings(steering_log, time_scale):
    steerings = defaultdict(list)
    speeds = defaultdict(list)
    with open(steering_log) as f:
        for line in f.readlines()[1:]:
            fields = line.split(",")
            nanosecond, angle, speed = int(fields[1]), float(fields[2]), float(fields[4])
            timestamp = nanosecond / time_scale
            steerings[timestamp].append(angle)
            speeds[timestamp].append(speed)
    return steerings, speeds


def camera_adjust(angle,speed,camera):

    # Left camera -20 inches, right camera +20 inches (x-direction)
    # Steering should be correction + current steering for center camera

    # Chose a constant speed
    speed = 10.0  # Speed

    # Reaction time - Time to return to center
    # The literature seems to prefer 2.0s (probably really depends on speed)
    if speed < 1.0:
        reaction_time = 0
        angle = angle
    else:
        reaction_time = 2.0 # Seconds

        # Trig to find angle to steer to get to center of lane in 2s
        opposite = 20.0 # inches
        adjacent = speed*reaction_time*12.0 # inches (ft/s)*s*(12 in/ft) = inches (y-direction)
        angle_adj = np.arctan(float(opposite)/adjacent) # radians
    	
        # Adjust based on camera being used and steering angle for center camera
        if camera == 'left':
            angle_adj = -angle_adj
        angle = angle_adj + angle

    return angle


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
        # Uncomment to view cropped images and sizes
        img = imread(path.join(prefix, '%d.jpg' % id))

        #imsave('original.jpg', img)
        #lx, ly = img.shape[0],img.shape[1]
        #print('Original image shape: %d by %d' % (lx,ly))

        # Cropping
        crop_img = img[200:,:]

        # Resizing
        img = imresize(crop_img, size=image_size)

        #lx, ly = crop_img.shape[0],crop_img.shape[1]
        #print('Cropped image shape: %d by %d' % (lx,ly))
        
        #lx, ly = img.shape[0],img.shape[1]
        #print('Final resized image shape: %d by %d' % (lx,ly))
        #imsave('resized.jpg', img)  
        
        imgs.append(img)
    if len(imgs) < 1:
        print('Error no image at timestamp')
        print(ids)
    img_block = np.stack(imgs, axis=0)
    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))
    return img_block


def read_images_augment(image_folder, camera, ids, image_size):
    prefix = path.join(image_folder, camera)
    imgs = []
    j = 0
    for id in ids:
        # Uncomment to view cropped images and sizes
        img = imread(path.join(prefix, '%d.jpg' % id))

        # Flip image
        img = np.fliplr(img)

        # Cropping
        crop_img = img[200:,:]

        # Resizing
        img = imresize(crop_img, size=image_size)

        # Rotate randomly by small amount (not a viewpoint transform)
        rotate = random.uniform(-1, 1)
        img = ndimage.rotate(img, rotate, reshape=False)

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
def data_generator(steering_log, image_log, image_folder, unique_list, gen_type='train',
                   camera='center', batch_size=32, time_factor=10, image_size=0.5,
                   timestamp_start=None, timestamp_end=None, shuffle=True,
                   preprocess_input=normalize_input,
                   preprocess_output=exact_output):
    
    # Constants
    # -----------------------------------------------------------------------------
    minmax = lambda xs: (min(xs), max(xs))
    time_scale = int(1e9) / time_factor

    # Read the image stamps for each camera
    # -----------------------------------------------------------------------------
    if gen_type == 'train':
        image_stamps = read_image_stamps(image_log, camera[0], time_scale)
        image_stamps_r = read_image_stamps(image_log, camera[1], time_scale)
        image_stamps_l = read_image_stamps(image_log, camera[2], time_scale)
    else:
        image_stamps = read_image_stamps(image_log, camera, time_scale)
    
    # Read all steering angles and speeds
    # -----------------------------------------------------------------------------
    steerings, speeds = read_steerings(steering_log, time_scale)

    # More data exploration stats
    # -----------------------------------------------------------------------------
    print('timestamp range for all steerings: %d, %d' % minmax(steerings.keys()))
    print('timestamp range for all images: %d, %d' % minmax(image_stamps.keys()))
    print('min and max # of steerings per time unit: %d, %d' % minmax(map(len, steerings.values())))
    print('min and max # of images per time unit: %d, %d' % minmax(map(len, image_stamps.values())))
    
    # Generate images and steerings within one time unit
    # (Mean steering angles used when mulitple steering angels within a single time unit)
    # -----------------------------------------------------------------------------
    start = max(min(steerings.keys()), min(image_stamps.keys()))
    if timestamp_start:
        start = max(start, timestamp_start)
    end = min(max(steerings.keys()), max(image_stamps.keys()))
    if timestamp_end:
        end = min(end, timestamp_end)
    print("sampling data from timestamp %d to %d" % (start, end))
    
    # While loop for data generator
    # -----------------------------------------------------------------------------
    i = start
    num_aug = 0
    x_buffer, y_buffer, buffer_size = [], [], 0
    while True:
        if i > end:
            i = start

        if gen_type =='train':
            if i == start:
                camera_select = camera[0]
        else:
            camera_select = camera

        coin = random.randint(1, 2)
        if steerings[i] and image_stamps[i]:
            if camera_select == 'right':
                images = read_images(image_folder, camera_select, image_stamps_r[i], image_size)
            elif camera_select == 'left':
                images = read_images(image_folder, camera_select, image_stamps_l[i], image_size)
            elif camera_select == 'center':
                if gen_type == 'train':
                    if coin == 1:
                        images = read_images(image_folder, camera_select, image_stamps[i], image_size)
                    else:
                        images = read_images_augment(image_folder, camera_select, image_stamps[i], image_size)         
                else:
                    images = read_images(image_folder, camera_select, image_stamps[i], image_size)

            # Mean angle with a timestamp
            angle = np.repeat([np.mean(steerings[i])], images.shape[0])
            # Adjust steering angle for horizontal flipping
            if gen_type == 'train':
                if coin == 2:
                    angle = -angle
            speed = np.repeat([np.mean(speeds[i])], images.shape[0])
            # Adjust the steerings of the offcenter cameras
            if camera_select != 'center':
                angle = camera_adjust(angle[0],speed[0],camera_select)
                angle = np.repeat([angle], images.shape[0])
            x_buffer.append(images)
            y_buffer.append(angle)
            buffer_size += images.shape[0]
            if buffer_size >= batch_size:
                indx = range(buffer_size)
                if gen_type == 'train':
                    np.random.shuffle(indx)
                x = np.concatenate(x_buffer, axis=0)[indx[:batch_size], ...]
                y = np.concatenate(y_buffer, axis=0)[indx[:batch_size], ...] 
                x_buffer, y_buffer, buffer_size = [], [], 0
                yield preprocess_input(x.astype(np.float32)), preprocess_output(y)
        
        # Using all three cameras
        # ---------------------------------------------------------------------
        # Which camera to use (change line 323 if you want to use all)
        if gen_type == 'train':
            camera_select = 'center' #random.choice(camera)

        if shuffle:
            i = int(random.choice(unique_list))
            while i > end:
                i = int(random.choice(unique_list))
        else:
            i += 1
            while i not in unique_list:
                i += 1
                if i > end:
                    i = start
