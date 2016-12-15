import glob
import argparse
import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity

from rmse import calc_rmse
from generator import gen
from scipy import misc


class Model(object):
    def __init__(self,
                 model_path,
                 X_train_mean_path):

        self.model = load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        print self.mean_angle
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img):
        img_path = 'test.jpg'
        misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle[0]

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return self.mean_angle[0]

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)
            X = X[:,:,::-1]
            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0
            return self.model.predict(X)[0][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Runner for team rambo')
    parser.add_argument('bagfile', type=str, help='Path to ROS bag')

    args = parser.parse_args()


    model = Model("tanel/final_model.hdf5", "tanel/X_train_mean.npy")
    print calc_rmse(lambda image_pred: model.predict(image_pred),
                   gen(args.bagfile))
