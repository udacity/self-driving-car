# Udacity Self Driving Car Challenge 2 - Using Deep Learning to Predict Steering Angles

XXX place solution for the [Udacity Self Driving Car Challenge 2](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3).

The goal of the Udacity Self-Driving Car Challenge is to build an open-source self-driving vehicle.
This goal was broken into challenges. In Challenge 2 your program had to predict steering commands from camera imagery.

# Data

The first phase of the challenge was based on driving data from El Camino Real (small curves, mostly straight driving), 
and the second phase was based on driving data from San Mateo to Half Moon Bay (curvy and highway driving).

* Phase 1 data: ADD LINK
* Phase 2 data: ADD LINK

The phase 1 data set has PNG images. 
The phase 2 data set is in the ROSBAG format.
We used rwightman's [Udacity Reader docker tool](https://github.com/rwightman/udacity-driving-reader) 
to convert the images into PNGs.

Even though the imagery is from three different cameras (left, center, right), we only used the center images.

The code assumes the following directory structure for data:

```
- models

- round1
-- train
--- 1 
---- center
----- 1477431483393873340.png
----- ...
----- 1477431802438024821.png
--- 2
---- center
----- 1477429515920298589.png
----- ...
----- 1477429548524716563.png
...
--- 21
---- center
----- 1477436619607286002.png
----- ...
----- 1477436971856447647.png
-- test
--- center
---- 1477439402646429224.png
---- ....
---- 1477439732692922522.png

- round2
-- train
--- center
---- 1479424215880976321.jpg
---- ...
---- 1479426572343447996.jpg
-- test
--- center
---- 1479425441182877835.jpg
---- ...
---- 1479425721881751009.jpg
```

Change `data_path` in `config.py` value to point to this data directory.

# Pre-processing

The raw images are of size 640 x 480. We resized the images to 256 x 192, converted from RGB color format to HSV, 
used only the V channel, computed lag 1 differences between frames and used 4 consecutive differenced images.
For example, at time t we used [x_{t} - x_{t-1}, x_{t-1} - x_{t-2}, x_{t-2} - x_{t-3}, x_{t-3} - x_{t-4}] as input where x corresponds to the V-channel of the original image. 
No future frames were used to predict the current steering angle.

To pre-process phase 2 training data, run:

```
python preprocess_train_data.py
```

To pre-process phase 2 test data, run:

```
python preprocess_train_data.py
```

These pre-processing scripts convert image sets to numpy arrays.

# Model

Our final model is a 4-layer convolutional neural network with PReLU activation function. We used Adam optimizer.
We were surprised that tricks that work well with classification models (batchnorm, small 3x3 filters, deeper models, dropout) didn't help us much.
We mostly tried models similar to comma.ai [steering model](https://github.com/commaai/research/blob/master/train_steering_model.py), Nvidia's self-driving car model and VGG-like models.
Our final model was the following:

```
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(PReLU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(PReLU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(PReLU())
model.add(Dense(512))
model.add(PReLU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
```

To train the model, run:

```
python train.py
```

Available parameters:

* `--data` - alias for pre-processed data. There are multiple ways to pre-process the data (how many consecutive frame to use, image size, etc).
This parameter value gives us information what data set to use.
* `--num_channels` - number of channels the data has. For example, if you use 4 consecutive frames, then `num_channels` must be 4.
* `--img_height` - image height in pixels, default is 256.
* `--img_width` - image width in pixels, default is 256.
* `--model` - model definition file, see `models.py` for different models.
* `--test_part` - which part of the data to use as validation set.
* `--batch_size` - minibatch size, default is 32.
* `--num_epoch` - number epochs to train, default is 10.
* `--data_path` - folder path to pre-processed numpy arrays.

To predict steering angles from test data, run:

```
python predict.py
```

# Inspecting the model

To get an automatic report with predicted steering angle distributions and error visualizations, run the following R script:

```
Rscript render_reports.R
```

To visualize model predictions on test data, run:

```
python visualize.py
```

White circle shows the true angle, black circle shows the predicted angle.
In both scripts, you might need to change the variable `IMG_PATH` to point to the location of phase 2 images.

These visualizations can help us understand the weaknesses of the model.
For example, human steering movements are smoother on straight road while the model zig-zags.

VIDEO

# Pointers and Acknowledgements

Some of the code is based on comma.ai steering angle prediction scripts.
Some of the model architectures are based on Nvidia's end-to-end self-driving car paper.
Rwightman docker tool was used to convert the round 2 data from ROSBAG to JPG.
Keras.