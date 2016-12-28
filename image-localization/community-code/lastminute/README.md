# Overview

This is a ROS package submission to the Udacity Image Localization challenge, described in detailed [here](https://medium.com/udacity/challenge-3-image-based-localization-5d9cadcff9e7#.i6fir8d5v).

The goal of the this submission is to develop a system that train against a set of geotagged images, and attempts to generate match candidate from the training dataset, upon receiving an image that is geographically overlapped with the train images.

The approach we took is to adapt a [DBoW2](https://github.com/dorian3d/DBoW2) and [DLoopDetector](https://github.com/dorian3d/DLoopDetector) to train against a set of images along with its GPS location, and generate a csv of image name, lat, and lon as the output by matching a set of test images against the training dataset.

## Bag of Words

Bag of words (BoW) is at the center of our challenge submission. It is an algorithm for simplyfing representation of a N-dimensional signal (audio, image) to a 1-D vector. It follows a few simple steps to achieve this:

1. Extract features from the signal instances from a training dataset.
2. Perform k-mean clusterings on the training features to construct K clusters, forming first layer of the vocabulary tree
3. For each clusters, repeat clustering to generate a new layer of layer of vocabulary tree, and repeat for L layers.
4. The leaves of the vocabulary tree is a word!
5. Upon receiving a new image, we represent it as BoW vector by extracting features from this image, and bin the features to its closest word (L1 norm is typically used in literature, but it depends on the image feature type). This generates a K^L size vector, with each entry keeping track Aof the count of features binned into a particular word bucket.
6. A query image will be answered with an image with the closest BoW vector given a distance function.
7. DBoW2 then perform various checks (RANSAC for geometry consistency of features between the images, neighbourhood consistency, etc). If all tests pass, the image is considered a successful match.

# Installation

This package has been tested on Ubuntu 16.04. To install this package, follow the steps below:

1. Install ROS, instructions provided [here](http://wiki.ros.org/kinetic/Installation/Ubuntu)
2. Install [DLib](https://github.com/dorian3d/DLib)
3. Install [DBoW2](https://github.com/dorian3d/DBoW2)

If you don't already have a ROS workspace, please create via:
```
mkdir ~/ros_ws
mkdir ~/ros_ws/src
cd ~/ros_ws/src
catkin_init_workspace
cd ~/ros_ws & catkin_make
```
Install this package via:
```
git clone https://github.com/y22ma/udacity_place_recognition
cd ~/ros_ws
catkin_make
```

## Python node dependencies

Script depends on following libraries:
- ros (obviously)
- cv2
- geo.py from OpenSFM
- numpy

# Running

## Convert ros bags to images and coordinates

Please run the conversion script via:
```
roscore
rosrun udacity_place_recognition bag2images.py
```

Please play the rosbags that contains /vehicle/gps/fix and /center_camera/image_color/compressed:
```
rosbag play udacity-dataset_sensor_camera_center_*.bag udacity-dataset_io_vehicle_*.bag  
```

or just play all bag files:

```
rosbag play *.bag
```

You can also speedup export with rate parameter:

```
rosbag play *.bag -r 10
```

## Running against Challenge 3 test dataset

An example launch file is provided. Please change image_dir parameter to point to the training image dataset, and test_image_dir to the test set.

Our submission used the [ElCaminoBack](http://academictorrents.com/details/c9dae89d2e3897e6aa98c0c8196348c444998a2a) dataset in the legacy dataset section of the [Udacity SDC github page](https://github.com/udacity/self-driving-car/tree/master/datasets).

The output csv is saved under /tmp/result.csv!

The launch file should be modified with paths according to your data paths:
- image_dir - directory with images exported from bag files; default: /tmp
- pose_file - lla.txt file generated with exporter; default: /tmp/lla.txt
- test_image_dir - directory with test dataset

```
roslaunch udacity_place_recognition place_recognition.launch
```

The program first spend a while loading all the training images to construct the vocabulary. Definitely should improve this by saving the descriptors and database constructed from the training dataset, and load in the feature.
