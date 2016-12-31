import os
import cv2
import glob
import lmdb
import caffe
import random
import numpy as np
from caffe.proto import caffe_pb2
from img_tools import *

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def make_datum(img, label):
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_lmdb = './input/way/train_lmdb'
validation_lmdb = './input/way/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

#MV to SF
train_data1 = [img for img in glob.glob("./data/MVSF/center/*.jpg")]
#SF to MV
train_data2 = [img for img in glob.glob("./data/SFMV/center/*.jpg")]
#Shuffle train_data
random.shuffle(train_data1)
random.shuffle(train_data2)
train_data=train_data1+train_data2
random.shuffle(train_data)

print len(train_data)


in_db = lmdb.open(train_lmdb, map_size=int(1e11))
with in_db.begin(write=True) as in_txn:
    p=0
    for in_idx, img_path in enumerate(train_data):
        if (in_idx % 2 == 0) or (in_idx % 3 == 0):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'MVSF' in img_path:
            label = 0
        else:
            label = 1
        datum = make_datum(img, label)
        in_txn.put('{:0>6d}'.format(in_idx), datum.SerializeToString())
        print '{:0>6d}'.format(in_idx) + ':' + img_path+' '+str(label)
        p=in_idx
in_db.close()


in_db = lmdb.open(validation_lmdb, map_size=int(1e11))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx % 8 == 0:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if 'MVSF' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            in_txn.put('{:0>6d}'.format(in_idx), datum.SerializeToString())
            print '{:0>6d}'.format(in_idx) + ':' + img_path
in_db.close()
