import glob
import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity

class Model(object):
    def __init__(self, 
                 model_path,
                 X_train_mean_path):
        
        self.model = load_model(model_path)
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        self.img0 = None
        self.state = deque(maxlen=2)
        
    def predict(self, img_path):
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)
        
        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle
            
        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1
            
            return self.mean_angle
            
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
            
            return self.model.predict(X)[0]
        

if __name__ == "__main__":        
    filenames = glob.glob("imgs/*.jpg")
    model = Model("checkpoints/final_model.hdf5", "data/X_train_mean.npy")
    
    for f in filenames:
        print model.predict(f)