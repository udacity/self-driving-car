import tensorflow as tf
import os
import argparse
import numpy as np

from rmse import calc_rmse
from generator import gen


class RwightmanModel(object):
    """Steering angle prediction model for Udacity challenge 2.
    """

    def __init__(self, alpha=0.5, graph_path='',  metagraph_path='', checkpoint_path=''):
        """Model constructor.

        The model requires either a graph_path or a metagraph path and checkpoint path to initialize.

        :param alpha: Exponential moving average alpha value, set to 0 to disable smoothing.
        :param graph_path: Tensorflow self-contained graph file path with model variables frozen
        as constants. If this is set, metagraph_path and checkpoint_path are ignored.
        :param metagraph_path: Tensorflow meta-graph file for model if graph_path is not specified..
        :param checkpoint_path:  Checkpoint file containing variable values for specified meta-graph.
        """
        if graph_path:
            assert os.path.isfile(graph_path)
        else:
            assert checkpoint_path and metagraph_path
            assert os.path.isfile(checkpoint_path) and os.path.isfile(metagraph_path)
        self.graph = tf.Graph()
        with self.graph.as_default():
            if graph_path:
                # load a graph with weights frozen as constants
                graph_def = tf.GraphDef()
                with open(graph_path, "rb") as f:
                    graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(graph_def, name="")
                self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            else:
                # load a meta-graph and initialize variables from checkpoint
                saver = tf.train.import_meta_graph(metagraph_path)
                self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                saver.restore(self.session, checkpoint_path)
        self.model_input = self.session.graph.get_tensor_by_name("input_placeholder:0")
        self.model_output = self.session.graph.get_tensor_by_name("output_steer:0")
        self.last_steering_angle = 0
        self.alpha = alpha

    def predict(self, image):
        """ Predict steering angle from image.

        :param image: Input image, expected to be an RGB (H, W, C) numpy array with 0-255 pixel values.
        Normalization of image is perfomed in TF model.
        :return: steering angle prediction, radians as float
        """
        feed_dict = {self.model_input: image}
        steering_angle = self.session.run(self.model_output, feed_dict=feed_dict)
        if self.alpha:
            if self.last_steering_angle is None:
                self.last_steering_angle = steering_angle
            steering_angle = self.alpha * steering_angle + (1 - self.alpha) * self.last_steering_angle
            self.last_steering_angle = steering_angle
        return steering_angle


def main():
    parser = argparse.ArgumentParser(description='Model Runner for team rwightman')
    parser.add_argument('bagfile', type=str, help='Path to ROS bag')
    parser.add_argument('--alpha', type=float, default=0.1, help='Path to the metagraph path')
    parser.add_argument('--graph_path', type=str, help='Path to the self contained graph def')
    parser.add_argument('--metagraph_path', type=str, help='Path to the metagraph path')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint path')
    parser.add_argument('--debug_print', dest='debug_print', action='store_true',
                        help='Debug print of predicted steering commands')
    args = parser.parse_args()

    def get_model():
        model = RwightmanModel(
            alpha=args.alpha,
            graph_path=args.graph_path,
            metagraph_path=args.metagraph_path,
            checkpoint_path=args.checkpoint_path)
        # Push one empty image through to ensure Tensorflow is ready.
        # There is typically a large wait on the first frame through.
        model.predict(np.zeros(shape=[480, 640, 3]))
        return model

    def process(model, img):
        steering_angle = model.predict(img)
        if args.debug_print:
            print(steering_angle)
        return steering_angle

    model = get_model()

    print calc_rmse(lambda image_pred: model.predict(image_pred),
                   gen(args.bagfile))

if __name__ == '__main__':
    main()
