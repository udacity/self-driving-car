import cv2
import numpy as np

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

def region_of_interest(img):
    """
    Applies an image mask. From Udacity Self-Driving Car Nanodegree project 1
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    vert=np.array([[(0,0), (0,340),(260,250), (530,250),(640,290), (640,0)]], dtype=np.int32)
    ignore_mask_color = (255,) * 3
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vert, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img=region_of_interest(img)
    #Use only top part of image
    img=img[0:340,:,:]
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img
