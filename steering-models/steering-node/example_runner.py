import rospy
from steering_node import SteeringNode

import argparse
import json

from scipy import misc

from keras.optimizers import SGD
from keras.models import model_from_json


def process(model, img):
    img = misc.imresize(img[320:, :, :], (50, 200, 3))
    steering = model.predict(img[None, :, :, :])[0][0]
    print steering
    return steering


def get_model(model_file):
    with open(model_file, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(sgd, "mse")
    weights_file = model_file.replace('json', 'keras')
    model.load_weights(weights_file)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Runner')
    parser.add_argument('model', type=str, help='Path to model definition json. \
                        Model weights should be on the same path.')

    args = parser.parse_args()
    node = SteeringNode(lambda: get_model(args.model), process)
    rospy.spin()
