import numpy as np
from keras.models import load_model
import pandas as pd
import glob
from config import TestConfig

config = TestConfig()

data_path = config.data_path
data_name = config.data_name
val_part = config.val_part
model_path = config.model_path

test_path = "{}/X_test_round2_{}.npy".format(data_path, data_name)
test_path_phase1 = "{}/X_test_round1_{}.npy".format(data_path, data_name)
out_name = model_path.replace("checkpoints", "submissions").replace("hdf5", "txt")
out_name_phase1 = model_path.replace("checkpoints", "submissions").replace(".hdf5", "_phase1.txt")

print "Loading model..."
model = load_model(config.model_path)

print "Loading training data mean..."
X_train_mean = np.load(config.X_train_mean_path)

print "PHASE 2"
print "Reading test data..."
X_test = np.load(test_path)
X_test = X_test.astype('float32')
X_test -= X_train_mean
X_test /= 255.0

print "Predicting..."
preds = model.predict(X_test)
preds = preds[:, 0]

dummy_preds = np.repeat(config.angle_train_mean, config.num_channels)
preds = np.concatenate((dummy_preds, preds))

# join predictions with frame_ids
filenames = glob.glob("{}/round2/test/center/*.jpg".format(data_path))
filenames = sorted(filenames)
frame_ids = [f.replace(".jpg", "").replace("{}/round2/test/center/".format(data_path), "") for f in filenames]

print "Writing predictions..."
pd.DataFrame({"frame_id": frame_ids, "steering_angle": preds}).to_csv(out_name, index=False, header=True)

print "PHASE 1"
print "Reading phase1 test data..."
X_test = np.load(test_path_phase1)
X_test = X_test.astype('float32')
X_test -= X_train_mean
X_test /= 255.0

print "Predicting..."
preds = model.predict(X_test)
preds = preds[:, 0]

dummy_preds = np.repeat(config.angle_train_mean, config.num_channels)
preds = np.concatenate((dummy_preds, preds))

# join predictions with frame_ids
filenames = glob.glob("{}/round1/test/center/*.png".format(data_path))
filenames = sorted(filenames)
frame_ids = [f.replace(".png", "").replace("{}/round1/test/center/".format(data_path), "") for f in filenames]

print "Writing predictions..."
pd.DataFrame({"frame_id": frame_ids, "steering_angle": preds}).to_csv(out_name_phase1, index=False, header=True)

print "Done!"