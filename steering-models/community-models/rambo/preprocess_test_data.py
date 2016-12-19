from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.colors import rgb_to_hsv
from config import TestConfig
import glob

def make_grayscale_diff_data(num_channels=2):
    num_rows = len(filenames)
    
    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print "Processed " + str(i) + " images..."
        for j in range(num_channels):
            path0 = filenames[i - j - 1]
            path1 = filenames[i - j]
            img0 = load_img(path0, grayscale=True, target_size=(row, col))
            img1 = load_img(path1, grayscale=True, target_size=(row, col))
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j] = img[:, :, 0]
    return X


def make_grayscale_diff_tx_data(num_channels=2):
    num_rows = len(filenames)
    
    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print "Processed " + str(i) + " images..."
        path1 = filenames[i]
        img1 = load_img(path1, grayscale=True, target_size=(row, col))
        img1 = img_to_array(img1)
        for j in range(1, num_channels + 1):
            path0 = filenames[i - j]
            img0 = load_img(path0, grayscale=True, target_size=(row, col))
            img0 = img_to_array(img0)
            
            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j - 1] = img[:, :, 0]
    return X

def make_hsv_grayscale_diff_data(num_channels=2):
    num_rows = len(filenames)
    
    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print "Processed " + str(i) + " images..."
        for j in range(num_channels):
            path0 = filenames[i - j - 1]
            path1 = filenames[i - j]
            img0 = load_img(path0, target_size=(row, col))
            img1 = load_img(path1, target_size=(row, col))
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img0 = rgb_to_hsv(img0)
            img1 = rgb_to_hsv(img1)
            img = img1[:, :, 2] - img0[:, :, 2]
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)
    
            X[i - num_channels, :, :, j] = img
    return X

if __name__ == "__main__":
    config = DataConfig()
    data_path = config.data_path
    row, col = config.height, config.width

    print "Pre-processing phase 1 data..."
    filenames = glob.glob("{}/Challenge 2/Test/center/*.png".format(data_path))
    filenames = sorted(filenames)
    out_name = "{}/X_test_round1_hsv_gray_diff_ch4".format(data_path)
    X_test = make_hsv_grayscale_diff_data(4)
    np.save(out_name, X_test)
    
        
    print "Pre-processing phase 2 data..."
    filenames = glob.glob(DATA_PATH + "{}/Ch2_001/center/*.jpg".format(data_path))
    filenames = sorted(filenames)
    out_name = DATA_PATH + "{}/X_test_round2_hsv_gray_diff_ch4".format(data_path)
    X_test = make_hsv_grayscale_diff_data(4)
    np.save(out_name, X_test)
