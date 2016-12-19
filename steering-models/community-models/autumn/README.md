# Using Deep Learning to Predict Steering Angles

## Udacity Open-Source Self-Driving Car Challenge 2 - Intro
Getting to contribute code to a real, (and open-source!) self-driving car was a huge opportunity to learn and get hands-on experience. I’d never seen anything like it, so I had to give the challenge a try. Not only was the task well-defined and documented, but also there were plenty of data and tools provided to get started quickly. The Udacity team did a fantastic job in organizing the challenge and providing support for the community from beginning to end.

## Background

As I got started with the challenge, I had thought about the work from NVIDIA’s recent [paper](https://arxiv.org/pdf/1604.07316v1.pdf) and similar prior works, which perhaps inspired the challenge. The paper’s authors were able to achieve impressive results with a simple end-to-end deep learning model, generating steering angles from raw images, using video from a single view. Since this model (and previous ones) considered only a single frame from the video at a time, I wanted to explore whether using time series data would improve the predictions. For inspiration, I looked at several papers and projects on deep learning for video classification and captioning.

The paper [Deep Learning for Video Classification and Captioning](https://arxiv.org/abs/1609.06782) by Wu et al. is an up-to-date and extensive overview on the topics of Video (Action) Classification, Video Captioning, as well as related benchmarks, datasets, and deep learning models. The paper explores serveral end-to-end convolutional models starting with off-the-shelf features which provide good results. 

Papers such as 3D Convolutional Neural Networks for Human Action Recognition by Ji et al. and [Large-scale Video Classification with Convolutional Neural Networks](http://vision.stanford.edu/pdf/karpathy14.pdf) by Karpathy et al. used 3D CNNs and stacks of frames at mixed resolutions, respectively, to pick up on spatio-temporal features. Interestingly, these models performed similarly to the CNN model with a single frame as input. Wang and Shcmid reported better performance in hand-crafted features (optical flow and trajectory) in [Action Recognition with Improved Trajectories](https://hal.inria.fr/hal-00873267v2/document). This approached used dense optical flow features produced by Farneback's algorithm.

[Two-Frame Motion Estimation Based on Polynomial Expansion](http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf)
Other algorithms: [TV-L1](http://www.ipol.im/pub/art/2013/26/article.pdf), [Dual TV-L1](http://www.icg.tugraz.at/publications/pdf/pockdagm07.pdf) and [DeepFlow](https://hal.inria.fr/hal-00873592/document)

Motivated by the fact that videos can be decomposed into spatial and temporal components, Simonyan and Zisserman proposed [Two-Stream Convolutional Networks for Action Recognition in Videos](https://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf), fusing outputs of a spatial and motion stream CNN. Further approaches extending the two-stream model have since produced better results. In order to incorporate and model long-term temporal dynamics, later approaches added LSTM networks to the two-stream networks. Donahue et al. train two-layer LSTM networks on features from the two-stream network with much success, while Wu et al. fused outputs of LSTM and CNN networks to show that the two are highly complementary. Ng et al. further compared deep LSTM networks and feature pooling following CNN feature computation, showing similar performance of both.

[Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389)
[Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/pdf/1503.08909v2.pdf)

## Requirements & Dependencies
- Python 2.7
- [Numpy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Tensorflow](https://www.tensorflow.org/get_started/os_setup) 0.11.0rc1
- [OpenCV](http://opencv.org/downloads.html) 2 for Python
- [Keras](https://keras.io/) 1.1.0
- [Autopilot-TensorFlow](https://github.com/SullyChen/Autopilot-TensorFlow) is an open-source TensorFlow implementation of the NVIDIA paper, provided much of the model and training code
- [udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader) from rwightman, used to extract images from ROS bag files
- [GoDeeper](https://github.com/Miej/GoDeeper) is an AWS EC2 Community AMI with support for GPU acceleration, CUDA, CuDNN, TensorFlow, Keras, OpenCV

## Development Hardware
I began development locally on my Macbook, but transitioned to using AWS EC2 instances to improve speed of training and iteration. It was fortunate to be able to use the newer P2 instances, powered by NVIDIA K80 GPUs. Being able to launch additional instances helped with experimentation, and using Spot Instances also helped on cost savings. There are now many community AMIs for EC2 that help overcome some of the initial overhead in installing TensorFlow and other tools with GPU acceleration enabled. Though there is quite a lot of configuration to learn initially, cloud services like AWS allow for much quicker experimentation when using deep learning frameworks.

To get started with AWS, create an instance using the GoDeep AMI (IDs provided below), connected to at least 50GB of EBS storage. The compressed data may be downloaded via the [Transmission CLI](https://help.ubuntu.com/community/TransmissionHowTo) tool for Ubuntu, then extracted from ROS bag files to images via rwightman's [tool](https://github.com/rwightman/udacity-driving-reader). 

| Region                  | AMI ID       |
| ----------------------- |:------------:|
| US East (N. Virginia)   | `ami-a195cfb6` |
| US East (Ohio)          | `ami-86277de3` |
| US West (N. California) | `ami-3e22685e` |
| US West (Oregon)        | `ami-da3096ba` |
| EU (Ireland)            | `ami-afa3e9dc` |
| EU (Frankfurt)          | `ami-8f906be0` |
| Asia Pacific (Tokyo)    | `ami-32d37e53` |
| Asia Pacific (Seoul)    | `ami-6a20f404` |
| Asia Pacific (Singapore)| `ami-20e54543` |
| Asia Pacific (Sydney)   | `ami-6f2b170c` |
| Asia Pacific (Mumbai)   | `ami-8da5d1e2` |
| South America (São Paulo)| `ami-b3a438df` |


## Approach
My approach was to start from the basic model, and then try out ideas from other image and video models. There was a list off variations I had in mind: data augmentation, transforming regression into classification, swapping the RGB color space for YUV, edge detectors and Hough transforms, deeper and more complex convolutional networks. Some changes helped a bit, while some were ineffective or were difficult to train well. Luckily, noticeable changes came from adapting ideas from video classification models: using dense optical flow and recurrent networks to incorporate temporal data in addition to spatial data. Transfer learning was helpful in getting good results. Transforming a window of optical flow into three channels made it possible to train the images with the spatial CNN and existing weights. Using the activations as inputs to an LSTM layer provided additional conditioning on a much larger time scale.

## Data Processing
Images were extracted into PNG via rwightman's [udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader). This also provivded a CSV of interpolated steering wheel angles for the provided timestamps/frame IDs. In each phase of the challenge, only the center frame was used in final training, although using left and right images as shifted center images was experiemented with. 

To increase robustness to camera shifts and rotations, slight random translations and rotations to the original image were used as data augmentation, similar to the NVIDIA paper. This did not seem to improve performance significantly, particularly when applied before/after optical flow computation.  

Reducing the amount of cropping of the original image from 200 to 100 pixels boosted the performance, and was receptive to fine-tuning while keeping the size of the resized image the same. 

For optical flow, various approaches were considered for pre-processing training images. One consideration was to use a stack of optical flow outputs, where each layer is the horizontal or vertical displacement for each pair of frames (i + t - 1) and (i + t), where t = 0 to T, and T is the number of input frames. However, to take advantage of transfer learning and fine-tune the model based on the pre-trained weights from the base model, a 3-channel output image was desired. In order to achieve this, the dense optical flow output was converted from cartesian coordinates to polar, then mapped to the HSV coordinate space. The angular component was mapped to the hue, and the magnitude component was mapped to the value, with all pixels given full saturation. This was then mapped back to the BGR color space to be consumed by the spatial CNN model. Visually, the result is that the displacement's direction and magnitude are shown in the color's hue and value, where a black pixel represents no motion. 

From this result, the best variation was chosen based on further training and testing of the model, as well as the possibility of inference from visual inspection. Observing the optical flow images mapped to the RGB/BGR representation, not only car and object motion is detected, but also the movement of lane markers and occassionally displacement of the landscape due to camera motion. However, in providing frame-to-frame mapping where the output only depends on the current frame and previous frame, lots of artifacts are present due to camera motion, while information from long-term displacement is not present. One possible solution is to consider use a larger fixed window k, where the optical flow for frame i would be based on frames i and i - k. Another way would be to compute flow for all frame pairs from (i - k, i) to (i - 1, i), but would incur additional computational cost based on linear factor k. Upon further exploration, averaging a fixed window k for frame pairs (i - k - 1, i - k) to (i - 1, i) provided the desired result of reducing sudden camera motions and showing long-term motion, while preventing the need to compute optical flow more than once per frame pair. Further selection of window size and weighting of the outputs by frame pairs was determined by visual inspection and training on the model.

```
prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

last.append(flow)
if len(last) > window_size:
    last.popleft()

last = list(self.last)
for x in range(len(last)):
    last[x] = last[x] * weights[x]

avg_flow = sum(last) / sum(weights)
mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])

hsv = np.zeros_like(prev_image)
hsv[..., 1] = 255
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

# Model
Though other open-source models were tested (VGGNet, ResNet V1), the open-source implementation of NVIDIA’s paper, given the pre-trained model, was able to provide the most promising results given the time constraint. Perhaps given more data, the margin of benefit from using a larger model would be greater. After cropping the image, it is fed into three 5x5 conv layers with stride 2, followed by two 3x3 conv layers. This is followed by five fully-connected layers with dropout. Batch normalization is added after the activation function, which is recommended versus before the activation function, when using dropout.

```
def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    initial = initializer(shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


class ConvModel(object):
    ''' Implements the ConvNet model from the NVIDIA paper '''
    def __init__(self, dropout_prob=0.2, batch_norm=False, whitening=False, is_training=True):
        x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        W_conv1 = weight_variable([5, 5, 3, 24])
        b_conv1 = bias_variable([24])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1, 2) + b_conv1)
        if batch_norm:
            h_conv1 = tf.contrib.layers.batch_norm(h_conv1, is_training=is_training, trainable=True)

        W_conv2 = weight_variable([5, 5, 24, 36])
        b_conv2 = bias_variable([36])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

        W_conv3 = weight_variable([5, 5, 36, 48])
        b_conv3 = bias_variable([48])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
        if batch_norm:
            h_conv3 = tf.contrib.layers.batch_norm(h_conv3, is_training=is_training, trainable=True)

        W_conv4 = weight_variable([3, 3, 48, 64])
        b_conv4 = bias_variable([64])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

        W_conv5 = weight_variable([3, 3, 64, 64])
        b_conv5 = bias_variable([64])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)
        if batch_norm:
            h_conv5 = tf.contrib.layers.batch_norm(h_conv5, is_training=is_training, trainable=True)

        W_fc1 = weight_variable([1152, 1164])
        b_fc1 = bias_variable([1164])

        h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
        if batch_norm:
            h_fc1 = tf.contrib.layers.batch_norm(h_fc1, is_training=is_training, trainable=True)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1164, 100])
        b_fc2 = bias_variable([100])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='fc2')
        if batch_norm:
            h_fc2 = tf.contrib.layers.batch_norm(h_fc2, is_training=is_training, trainable=True)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        W_fc3 = weight_variable([100, 50])
        b_fc3 = bias_variable([50])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3, name='fc3')
        if batch_norm:
            h_fc3 = tf.contrib.layers.batch_norm(h_fc3, is_training=is_training, trainable=True)
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

        W_fc4 = weight_variable([50, 10])
        b_fc4 = bias_variable([10])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4, name='fc4')
        if batch_norm:
            h_fc4 = tf.contrib.layers.batch_norm(h_fc4, is_training=is_training, trainable=True)
        h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

        W_fc5 = weight_variable([10, 1])
        b_fc5 = bias_variable([1])
        y = tf.mul(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2, name='y')

        self.x = x
        self.y_ = y_
        self.y = y
        self.keep_prob = keep_prob
        self.fc2 = self.h_fc2
        self.fc3 = self.h_fc3
```

## Training
The network was initially trained on the first set of training data with no pre-processing. The parameters were chosen through manual selection and random search. It was found that starting with a dropout rate of 0.3 and learning rate of 1e-3 worked well, then lowering the learning rate to 1e-4 and 1e-5 after 50k and 100k steps, respectively. Whitening via histogram equalization was tested on the training data, but created unwanted artifacts due to the texture of the road and limitation of the window size. This is also perhaps not as important when batch normalization is used. Training the model on the images transformed from RGB to YUV color space (as presented in the NVIDIA paper) was also tested, but did not provide better results after a reasonable amount of training. Another training technique from the NVIDIA paper was changing the training data distribution to skew towards higher magnitude steering angles rather than close to 0 (driving straight, which occurs for a majority of the first training set). This provided a significant increase in test time performance.


## Further Work

Although the results fared well for the context of the challenge, further work would need to be done to get a model to perform on the road. Over-correction and feedback loops are likely to occur and can potentially be helped by providing examples of course correction in the training data. In real-world performance, speed of a forward pass plays a major role, which could be mitigated by optimizing the pre-processing and using fewer parameters. Given more time, training two-stream models with a stack of optical flow images would have performed better. Using better algorithms for optical flow, including deep learning models, may have generated better results. Jointly training the optical flow and LSTM components would also be something to explore.

I was initially hesitant when the challenges were announced, since problems in autonomous driving seem tremendous in scope and depth. Breaking it down into these smaller challenges and providing lots of support and resources made working on this project a lot of fun. Looking forward to learning more, and can’t wait to see future progress on the car!

