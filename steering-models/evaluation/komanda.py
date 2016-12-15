#!/usr/bin/env python

"""
Udacity self-driving car challenge 2
Team komanda steering model
Author: Ilya Edrenkin, ilya.edrenkin@gmail.com
"""

import argparse
import tempfile
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from math import pi
from rmse import calc_rmse
from generator import gen
import time


class KomandaModel(object):
    def __init__(self, checkpoint_dir, metagraph_file):
        self.graph =tf.Graph()
        self.LEFT_CONTEXT = 5 # TODO remove hardcode; store it in the graph
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(metagraph_file)
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        self.session = tf.Session(graph=self.graph)
        saver.restore(self.session, ckpt)
        self.input_images = deque() # will be of size self.LEFT_CONTEXT + 1
        self.internal_state = [] # will hold controller_{final -> initial}_state_{0,1,2}

        # TODO controller state names should be stored in the graph
        self.input_tensors = map(self.graph.get_tensor_by_name, ["input_images:0", "controller_initial_state_0:0", "controller_initial_state_1:0", "controller_initial_state_2:0"])
        self.output_tensors = map(self.graph.get_tensor_by_name, ["output_steering:0", "controller_final_state_0:0", "controller_final_state_1:0", "controller_final_state_2:0"])

    def predict(self, img):
        if len(self.input_images) == 0:
            self.input_images += [img] * (self.LEFT_CONTEXT + 1)
        else:
            self.input_images.popleft()
            self.input_images.append(img)
        input_images_tensor = np.stack(self.input_images)
        if not self.internal_state:
            feed_dict = {self.input_tensors[0] : input_images_tensor}
        else:
            feed_dict = dict(zip(self.input_tensors, [input_images_tensor] + self.internal_state))
        steering, c0, c1, c2 = self.session.run(self.output_tensors, feed_dict=feed_dict)
        self.internal_state = [c0, c1, c2]
        return steering[0][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Runner for team komanda')
    parser.add_argument('bagfile', type=str, help='Path to ROS bag')
    parser.add_argument('metagraph_file', type=str, help='Path to the metagraph file')
    parser.add_argument('checkpoint_dir', type=str, help='Path to the checkpoint dir')
    parser.add_argument('--debug_print', dest='debug_print', action='store_true', help='Debug print of predicted steering commands')
    args = parser.parse_args()

    def make_predictor():
        model = KomandaModel(
            checkpoint_dir=args.checkpoint_dir,
            metagraph_file=args.metagraph_file)
        return lambda img: model.predict(img)

    def process(predictor, img):
        steering = predictor(img)
        if args.debug_print: print steering
        return steering
    model = make_predictor()

    print calc_rmse(lambda image_pred: model(image_pred),
                   gen(args.bagfile))
