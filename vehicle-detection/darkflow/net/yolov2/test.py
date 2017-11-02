import numpy as np
import math
import cv2
import os
#from scipy.special import expit
from utils.box import BoundBox, box_iou, prob_compare
from utils.box import prob_compare2, box_intersection
	
_thresh = dict({
	'person': .2,
	'pottedplant': .1,
	'chair': .12,
	'tvmonitor': .13
})

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	# meta
	meta = self.meta
	H, W, _ = meta['out_size']
	threshold = meta['thresh']
	C, B = meta['classes'], meta['num']
	anchors = meta['anchors']
	net_out = net_out.reshape([H, W, B, -1])

	boxes = list()
	for row in range(H):
		for col in range(W):
			for b in range(B):
				bx = BoundBox(C)
				bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
				bx.c = expit(bx.c)
				bx.x = (col + expit(bx.x)) / W
				bx.y = (row + expit(bx.y)) / H
				bx.w = math.exp(bx.w) * anchors[2 * b + 0] / W
				bx.h = math.exp(bx.h) * anchors[2 * b + 1] / H
				classes = net_out[row, col, b, 5:]
				bx.probs = _softmax(classes) * bx.c
				bx.probs *= bx.probs > threshold
				boxes.append(bx)

	# non max suppress boxes
	for c in range(C):
		for i in range(len(boxes)):
			boxes[i].class_num = c
		boxes = sorted(boxes, key = prob_compare)
		for i in range(len(boxes)):
			boxi = boxes[i]
			if boxi.probs[c] == 0: continue
			for j in range(i + 1, len(boxes)):
				boxj = boxes[j]
				if box_iou(boxi, boxj) >= .4:
					boxes[j].probs[c] = 0.


	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	for b in boxes:
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = 'object' * int(C < 2)
		label += labels[max_indx] * int(C>1)
		if max_prob > threshold:
			left  = int ((b.x - b.w/2.) * w)
			right = int ((b.x + b.w/2.) * w)
			top   = int ((b.y - b.h/2.) * h)
			bot   = int ((b.y + b.h/2.) * h)
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1
			thick = int((h+w)/300)
			cv2.rectangle(imgcv, 
				(left, top), (right, bot), 
				colors[max_indx], thick)
			mess = '{}'.format(label)
			cv2.putText(imgcv, mess, (left, top - 12), 
				0, 1e-3 * h, colors[max_indx],thick//3)

	if not save: return imgcv
	outfolder = os.path.join(self.FLAGS.test, 'out') 
	img_name = os.path.join(outfolder, im.split('/')[-1])
	cv2.imwrite(img_name, imgcv)
