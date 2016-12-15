'''
Results video generator Udacity Challenge 2
Original By: Comma.ai Revd: Chris Gundling
'''

from __future__ import print_function

import argparse
import sys
import numpy as np
import h5py
import pygame
import json
import pandas as pd
from os import path
#from keras.models import model_from_json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab
from pygame.locals import *

from data_TS import *

pygame.init()
size = (320*2, 160*3)
#size2 = (640,160)
pygame.display.set_caption("epoch data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
screen.set_alpha(None)

#camera_surface = pygame.surface.Surface((320,160),0,24).convert()
camera_surface = pygame.surface.Surface((320,160),0,24).convert()
clock = pygame.time.Clock()

# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
  row, col = perspective_tform(x, y)
  if row >= 0 and row < img.shape[0] and\
     col >= 0 and col < img.shape[1]:
    img[row-sz:row+sz, col-sz:col+sz] = color

def draw_path(img, path_x, path_y, color):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) #* deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255)):
  path_x = np.arange(0., 50.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
  draw_path(img, path_x, path_y, color)

# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')
    parser.add_argument('--camera', type=str, default='center', help='camera to use, default is center')
    parser.add_argument('--resized-image-width', type=int, help='image resizing')
    parser.add_argument('--resized-image-height', type=int, help='image resizing')
    args = parser.parse_args()

    dataset_path = args.dataset
    image_size = (args.resized_image_width, args.resized_image_height)
    camera = args.camera

    # steerings and images
    steering_log = path.join(dataset_path, 'final_example.csv')
    image_log = path.join(dataset_path, 'camera.csv')
    camera_images = dataset_path

    df_test = pd.read_csv('epoch.csv',usecols=['frame_id','steering_angle'],index_col = None)
    df_truth = pd.read_csv('ch2_final_eval.csv',usecols=['frame_id','steering_angle'],index_col = None)
    
    # Testing on the Test Images
    test_generator = data_generator(steering_log=steering_log,
                         image_log=image_log,
                         image_folder=camera_images,
                         camera=camera,
                         batch_size=5614,
                         image_size=image_size,
                         timestamp_start=14794254411,
                         timestamp_end=14794257218,
                         shuffle=False,
                         preprocess_input=normalize_input,
                         preprocess_output=exact_output)

    print('Made it to Testing')

    test_x, test_y = test_generator.next()
    print('test data shape:', test_x.shape)
    
    # Create second screen with matplotlib
    fig = pylab.figure(figsize=[6.4, 1.6], dpi=100)
    ax = fig.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    #ax.legend(loc='upper left',fontsize=8)
    line1, = ax.plot([], [],'b.-',label='Human')
    line2, = ax.plot([], [],'r.-',label='Model')
    A = []
    B = []
    ax.legend(loc='upper left',fontsize=8)
    
    red=(255,0,0)
    blue=(0,0,255)
    myFont = pygame.font.SysFont("monospace", 18)
    randNumLabel = myFont.render('Human Steer Angle:', 1, blue)
    randNumLabel2 = myFont.render('Model Steer Angle:', 1, red)
    speed_ms = 5 #log['speed'][i]

    # Run through all images
    for i in range(5614):
        #if i%100 == 0:
        #    print('%.2f seconds elapsed' % (i/20))
        img = test_x[i,:,:,:].swapaxes(0,2).swapaxes(0,1)

        predicted_steers = df_test['steering_angle'].loc[i]
        actual_steers = df_truth['steering_angle'].loc[i]

        draw_path_on(img, speed_ms, actual_steers/5.0)
        draw_path_on(img, speed_ms, predicted_steers/5.0, (255, 0, 0))

        A.append(df_test['steering_angle'].loc[i])
        B.append(df_truth['steering_angle'].loc[i])
        line1.set_ydata(A)
        line1.set_xdata(range(len(A)))
        line2.set_ydata(B)
        line2.set_xdata(range(len(B)))
        ax.relim()
        ax.autoscale_view()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        screen.blit(surf, (0,320))

        # draw on
        pygame.surfarray.blit_array(camera_surface, img.swapaxes(0,1))
        camera_surface_2x = pygame.transform.scale2x(camera_surface)
        screen.blit(camera_surface_2x, (0,0))
	
        diceDisplay = myFont.render(str(actual_steers*(180/np.pi)), 1, blue)
        diceDisplay2 = myFont.render(str(predicted_steers*(180/np.pi)), 1, red)
        screen.blit(randNumLabel, (50, 280))
        screen.blit(randNumLabel2, (400, 280))
        screen.blit(diceDisplay, (50, 300))
        screen.blit(diceDisplay2, (400, 300))
        clock.tick(60)
        pygame.display.flip()
