#Image-Based Localization
The solution for Udacity Self-Driving Car Challenge #3 by Robocar team (by Nikolay Falaleev).
Details on this Challenge can be found [here](https://medium.com/udacity/challenge-3-image-based-localization-5d9cadcff9e7#.cv1xx261f)

The solution is based on Deep Learning and realized the "Localization as Classification approach" with Googlenet artificial neural networks.

##Datasets
3 datasets with images of trips between Mountain View (MV) and San Francisco (SF) (provided by Udacity) were used for training of the neural networks:
* Ch2-Train.tar.gz
* UdacitySDC_ElCamino.tar.gz
* CH3_001.tar.gz

All archives were unpacked. Rosbags were processed by the Ross Wightman's (rwightman) [script](https://github.com/rwightman/udacity-driving-reader). Only center camera frames were saved.
To do it unused CAMERA_TOPICS from bagutils.py were removed.

##About the algorithm
###Approach
The problem was divided into two parts:
* Determination of the direction of ride (SF->MV or MV->SF)
* Localization on the El Camino road
The heart of the approach is a set of three [Googlenets](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt). One is used to determine the direction (we assume that all images in the dataset were obtained in the same direction, there were no U-turns), two others dedicated to localization (classification) of images within SF->MV or MV->SF path. The return trip was treated as a completely different road. Each of the two "roads" were divided into small "pointers" - areas, created by 50 pictures in a row (with consequent merging of too close pointers occured because of slow ride, for details see ./python/class_creator.py). The two CNNs were used to classify images by assigning pointers to them.
Resulting predictions were interpolated inside predicted pointers. As the algorithm does not have any speed estimation, speed from training dataset was used to calculate average coordinates shift between frames in each pointer.

###Data preparation
Images that were preprocessed by openCV:
* Histogram equalization
* Applying ROI mask. It was found out that masking the region of images which is usually occupied only by the road can increase the accuracy of predictions
* Image cropping to delete unused image area
* Image scaling to 224x224 px to feed into the neural network
To make pointers more balanced, maximal number of frames per pointer from a dataset was limited.
Due to the fact that the first image of a pointer is very similar to the last image of the previous pointer, only pictures with coordinates deviation less than 50% of the pointer size were used. It helps the neural network to predict pointers more confidently.

##Run the code

Start by setting the working directory to the root directory of the project

####Create pointers:

For path from MV to SF:

```
python ./python/class_creator.py -i ./data/MVSF/interpolated.csv -o ./pointers/pointers_MVSF.csv -od ./pointers/pointers_data_MVSF.pkl > ./pointers/pointers_create_MVSF.log
```
For path from SF to MV:

```
python ./python/class_creator.py -i ./data/SFMV/interpolated.csv -o ./pointers/pointers_SFMV.csv -od ./pointers/pointers_data_SFMV.pkl > ./pointers/pointers_create_SFMV.log
```

####Prepare images sets

MV->SF

```
python ./python/class_adder.py -o ./input/MVSF/pointers_MVSF.csv -id ./pointers/pointers_data_MVSF.pkl -i ./data/MVSF/interpolated.csv -sf ./data/MVSF/center/ -df ./input/MVSF/center/ -l MVSF -n 1534 -m 100
python ./python/class_adder.py -o ./input/MVSF/pointers_elcm_n.csv -id ./pointers/pointers_data_MVSF.pkl -i ./data/el_camino_north/interpolated.csv -sf ./data/el_camino_north/center/ -df ./input/MVSF/center/ -l elcn -n 1534 -m 300
python ./python/class_adder.py -o ./input/MVSF/pointers_Ch2.csv -id ./pointers/pointers_data_MVSF.pkl -i ./data/Ch2-Train2/interpolated.csv -sf ./data/Ch2-Train2/center/ -df ./input/MVSF/center/ -l Ch2 -n 1534 -m 300
```

SF->MV

```
python ./python/class_adder.py -o ./input/SFMV/pointers_SFMV.csv -id ./pointers/pointers_data_SFMV.pkl -i ./data/SFMV/interpolated.csv -sf ./data/SFMV/center/ -df ./input/SFMV/center/ -l SFMV -n 1798 -m 100
python ./python/class_adder.py -o ./input/SFMV/pointers_elcm_s.csv -id ./pointers/pointers_data_SFMV.pkl -i ./data/el_camino_south/interpolated.csv -sf ./data/el_camino_south/center/ -df ./input/SFMV/center/ -l elcs -n 1798 -m 300 
python ./python/class_adder.py -o ./input/SFMV/pointers_Ch2.csv -id ./pointers/pointers_data_SFMV.pkl -i ./data/Ch2-Train2/interpolated.csv -sf ./data/Ch2-Train2/center/ -df ./input/SFMV/center/ -l Ch2 -n 1798 -m 300
```
Note that there is no clear markers of direction in the Ch2-Train dataset, so, a part of the code in  _./python/class_adder.py_ should be uncommented with the correct sign ">" of "<" (see comments in the corresponding file). The code uses a special timepoint of the reversal turn which can easily be obtained by analyzing the coordinates of images in the training set.

####Create lmdbs

```
python ./python/create_lmdb.py -i ./input/MVSF/center/ -o ./input/MVSF/
python ./python/create_lmdb.py -i ./input/SFMV/center/ -o ./input/SFMV/
python ./python/create_lmdb_way.py
```

####Compute mean images

```
/path/to/caffe/build/tools/compute_image_mean -backend=lmdb ./input/MVSF/train_lmdb/ ./input/MVSF/mean.binaryproto
/path/to/caffe/build/tools/compute_image_mean -backend=lmdb ./input/SFMV/train_lmdb/ ./input/SFMV/mean.binaryproto
/path/to/caffe/build/tools/compute_image_mean -backend=lmdb ./input/way/train_lmdb ./input/way/mean.binaryproto
```

####Start training of the neural networks:

```
/path/to/caffe/build/tools/caffe train -solver ./caffe/MVSF/solver.prototxt > ./caffe/MVSF/model_train.log
/path/to/caffe/build/tools/caffe train -solver ./caffe/SFMV/solver.prototxt > ./caffe/SFMV/model_train.log
/path/to/caffe/build/tools/caffe train -solver ./caffe/way/solver.prototxt > ./caffe/way/model_train.log
```

It may take hours or days depending on your hardware, but I supplied you with pre-trained networks, so you don't have to wait for a long time.

Because the algorithm uses random sampling for training/validation datasets creating and random neural networks initialization, the results of training may be slightly different. 

####Let magic happen!

Use the trained neural networks to predict on the Test dataset.

```
python ./python/make_predictions.py
```


##Data structure

_./data_ folder with unbagged raw data. Initial .bag data files were processed with bagdump.py

_./data/Ch2-Train2_ directory with all extracted content from Ch2-Train.tar.gz

_./data/SFMV_ extracted content of udacity-datasetElCaminoBack directory from UdacitySDC_ElCamino.tar.gz

_./data/MVSF_ extracted content of udacity-datasetElCamino directory from UdacitySDC_ElCamino.tar.gz

_./data/el_camino_north_ extracted el_camino_north.bag from CH3_001.tar.gz

_./data/el_camino_south_ extracted el_camino_south.bag from CH3_001.tar.gz

_./python_ contains all python scripts

_./input_ contains preoared for training datasets

_./input/Test/centre_ contains test images

_./pointers_ pointers files

_./caffe_ contains files related to Caffe

All actual data is skipped in the repository in order to reduce its volume.

##Hardware and Software
###Hardware
Data processing and training of NNs were performed on a machine with the following components:

* AMD Phenom(tm) II X4 925
* 16 GB of RAM
* ASUS Dual GeForce GTX 1070
* SSD storage device

###Software

* OS openSUSE 13.2 (Harlequin) (x86_64), kernel 3.16.7-45-desktop
* Python 2.7.12 with lmdb, cv2, numpy
* Caffe (version obtained by _git clone_ from offisial [GitHub](https://github.com/BVLC/caffe) on Nov 1, 2016) with all [Prerequisites](http://caffe.berkeleyvision.org/installation.html#prerequisites) and Python interface
* CUDA ver. 8.0.44
* cuDNN 5.1
* Nvidia driver v. 367.57

Average training rate for the setup was 4.23 iter/sec.

Average prediction rate 45.7 fps.

##Acknowledgments
Thank you to [rwightman](https://github.com/rwightman/udacity-driving-reader) for an excellent ros bag processing script.
Thank you to my parents and friends who have supported me throughout the Challenge!
