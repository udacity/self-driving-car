
import cv2
import numpy as np
#import rospy
from keras import backend as K
from keras import metrics
from keras.models import load_model
from keras.models import model_from_json
import argparse
from collections import deque

from scipy import misc
import csv
import os
from math import pi
from rmse import calc_rmse
from generator import gen


class ChauffeurModel(object):
    def __init__(self,
                 cnn_json_path,
                 cnn_weights_path,
                 lstm_json_path,
                 lstm_weights_path):

        self.cnn = self.load_from_json(cnn_json_path, cnn_weights_path)
        self.encoder = self.load_encoder(cnn_json_path, cnn_weights_path)
        self.lstm = self.load_from_json(lstm_json_path, lstm_weights_path)

        # hardcoded from final submission model
        self.scale = 16.
        self.timesteps = 100

    def load_encoder(self, cnn_json_path, cnn_weights_path):
        model = self.load_from_json(cnn_json_path, cnn_weights_path)
        model.load_weights(cnn_weights_path)

        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        return model

    def load_from_json(self, json_path, weights_path):
        model = model_from_json(open(json_path, 'r').read())
        model.load_weights(weights_path)
        return model

    def make_cnn_only_predictor(self):
        def predict_fn(img):
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:,:,0] = cv2.equalizeHist(img[:,:,0])
            img = ((img-(255.0/2))/255.0)

            return self.cnn.predict_on_batch(img.reshape((1, 120, 320, 3)))[0, 0] / self.scale

        return predict_fn

    def make_stateful_predictor(self):
        steps = deque()

        def predict_fn(img):
            # preprocess image to be YUV 320x120 and equalize Y histogram
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:,:,0] = cv2.equalizeHist(img[:,:,0])
            img = ((img-(255.0/2))/255.0)

            # apply feature extractor
            img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))

            # initial fill of timesteps
            if not len(steps):
                for _ in xrange(self.timesteps):
                    steps.append(img)

            # put most recent features at end
            steps.popleft()
            steps.append(img)

            timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
            for i, img in enumerate(steps):
                timestepped_x[0, i] = img

            return self.lstm.predict_on_batch(timestepped_x)[0, 0] / self.scale

        return predict_fn

def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

metrics.rmse = rmse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Runner for team chauffeur')
    parser.add_argument('bagfile', type=str, help='Path to ROS bag')
    parser.add_argument('cnn_json_path', type=str, help='Path to cnn encoding json')
    parser.add_argument('cnn_weights_path', type=str, help='Path to cnn encoding weights')
    parser.add_argument('lstm_json_path', type=str, help='Path to lstm encoding json')
    parser.add_argument('lstm_weights_path', type=str, help='Path to lstm encoding weights')
    args = parser.parse_args()

    def make_predictor():
        K.set_learning_phase(0)
        model = ChauffeurModel(
            args.cnn_json_path,
            args.cnn_weights_path,
            args.lstm_json_path,
            args.lstm_weights_path)
        return model.make_stateful_predictor()

    def process(predictor, img):
        return predictor(img)

    model = make_predictor()

    print calc_rmse(lambda image_pred: model(image_pred),
                   gen(args.bagfile))
