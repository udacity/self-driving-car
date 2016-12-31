import os
import cv2
import glob
import lmdb
import caffe
import random
import numpy as np
import argparse
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

parser = argparse.ArgumentParser(description='Create lmdb')
parser.add_argument('-o', '--outfolder', type=str, nargs='?', default='./', help='Output folder')
parser.add_argument('-i', '--infolder', type=str, nargs='?', default='./', help='Input folder')
arg = parser.parse_args()
train_lmdb = arg.outfolder+'train_lmdb'
validation_lmdb = arg.outfolder+'validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)


train_data = [img for img in glob.glob(arg.infolder+'*.jpg')]
#Shuffle the train data
random.seed(12345)
random.shuffle(train_data)



in_db = lmdb.open(train_lmdb, map_size=int(1e11))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        #Skip every 8th image for the validation dataset
        if in_idx %  8 == 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        s=img_path.split("center/",1)[1]
        s=s.split("_",1)[0]
        datum = make_datum(img, int(s))
        in_txn.put('{:0>6d}'.format(in_idx), datum.SerializeToString())
        print '{:0>6d}'.format(in_idx) + ':' + img_path
in_db.close()


in_db = lmdb.open(validation_lmdb, map_size=int(1e11))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        #Use every 8th image for the validation dataset
        if in_idx % 8 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        s=img_path.split("center/",1)[1]
        s=s.split("_",1)[0]
        datum = make_datum(img, int(s))
        in_txn.put('{:0>6d}'.format(in_idx), datum.SerializeToString())
        print '{:0>6d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished creating DBs'
