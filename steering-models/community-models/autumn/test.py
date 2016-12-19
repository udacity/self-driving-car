# import rospy
# from steering_node import SteeringNode

from collections import deque
import argparse
import csv
import scipy.misc
import cv2

import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.layers.recurrent import LSTM
from autumn import ConvModel


class AutumnModel(object):
    def __init__(self, cnn_graph, lstm_json, cnn_weights, lstm_weights):
        sess = tf.Session()
        self.cnn = ConvModel(batch_norm=False, is_training=True)
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.cnn.y_, self.cnn.y))))
        saver = tf.train.Saver()
        saver.restore(sess, args.cnn_weights)

        self.fc3 = self.cnn.fc3
        self.y = self.cnn.y
        self.x = self.cnn.x
        self.keep_prob = self.cnn.keep_prob

        with open(lstm_json, 'r') as f:
            json_string = f.read()
        self.model = model_from_json(json_string)
        self.model.load_weights(lstm_weights)

        self.prev_image = None
        self.last = []
        self.steps = []

    def process(self, img):
        prev_image = self.prev_image if self.prev_image is not None else img
        self.prev_image = img
        prev = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.last.append(flow)

        if len(self.last) > 4:
            self.last.pop(0)

        weights = [1, 1, 2, 2]
        last = list(self.last)
        for x in range(len(last)):
            last[x] = last[x] * weights[x]

        avg_flow = sum(self.last) / len(self.last)
        mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])

        hsv = np.zeros_like(prev_image)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def predict(self, img):
        img = self.process(img)
        cv2.imshow("Flow", img)
        cv2.waitKey(1)
        image = scipy.misc.imresize(img[-400:], [66, 200]) / 255.0
        cnn_output = self.fc3.eval(feed_dict={self.x: [image], self.keep_prob: 1.0})
        self.steps.append(cnn_output)
        if len(self.steps) > 100:
            self.steps.pop(0)
        output = self.y.eval(feed_dict={self.x: [image], self.keep_prob: 1.0})
        angle = output[0][0]
        return angle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSTM on test data and write to file')
    parser.add_argument('--input', '-i', action='store', dest='input_file',
                        default='example.csv', help='Input model csv file name')
    parser.add_argument('--output', '-o', action='store', dest='output_file',
                        default='output-cnn.csv', help='Output csv file name')
    parser.add_argument('--data-dir', '--data', action='store', dest='data_dir',
                        default='/vol/data/Ch2_Test/center')
    parser.add_argument('--cnn-graph', '--cnn-meta', action='store', dest='cnn_graph',
                        default='autumn-cnn-model-tf.meta')
    parser.add_argument('--lstm-json', '--lstm-meta', action='store', dest='lstm_json',
                        default='autumn-lstm-model-keras.json')
    parser.add_argument('--cnn-weights', action='store', dest='cnn_weights',
                        default='autumn-cnn-weights.ckpt')
    parser.add_argument('--lstm-weights', action='store', dest='lstm_weights',
                        default='autumn-lstm-weights.hdf5')
    args = parser.parse_args()

    def make_predictor():
        model = AutumnModel(args.cnn_graph, args.lstm_json, args.cnn_weights, args.lstm_weights)
        return lambda img: model.predict(img)

    def process(predictor, img):
        return predictor(img)

    def test():
        output_file = args.output_file
        limit = 100000
        predictor = make_predictor()
        with open(args.input_file) as f:
            with open(output_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['frame_id', 'steering_angle'])
                writer.writeheader()
                reader = csv.DictReader(f)
                for row in reader:
                    if limit < 0:
                        break

                    filename = row['frame_id'] + '.jpg'
                    full_image = scipy.misc.imread(args.data_dir + "/" + filename, mode="RGB")
                    result = process(predictor, full_image)

                    img = cv2.imread('wheel.png', -1)
                    img = scipy.misc.imresize(img, 0.2)
                    height, width, _ = img.shape
                    M = cv2.getRotationMatrix2D((width/2, height/2), result * 360.0 / scipy.pi, 1)
                    dst = cv2.warpAffine(img, M, (width, height))

                    x_offset = (full_image.shape[1] - width) / 2
                    y_offset = 300
                    new_height = min(height, full_image.shape[0] - y_offset)
                    for c in range(0, 3):
                        alpha = dst[0:new_height, :, 3] / 255.0
                        color = dst[0:new_height, :, c] * (alpha)
                        beta = full_image[y_offset:y_offset+new_height, x_offset:x_offset+width, c] * (1.0 - alpha)
                        full_image[y_offset:y_offset+new_height, x_offset:x_offset+width, c] = color + beta

                    cv2.imshow("Output", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                    print((result, error))
                    limit -= 1
        print('Written to ' + output_file)
    test()

    # mode = SteeringNode(make_predictor, process)
    # rospy.spin()
