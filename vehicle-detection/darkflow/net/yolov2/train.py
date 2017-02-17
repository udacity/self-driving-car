import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from ..yolo.misc import show
import numpy as np
import os
import math

def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))

def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above, 
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W # number of grid cells
    anchors = m['anchors']
    
    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C)])
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.reshape(coords, [-1, H*W, B, 4])
    adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    coords = tf.concat(3, [adjusted_coords_xy, adjusted_coords_wh])
    
    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])
    
    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

    adjusted_net_out = tf.concat(3, [adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob])
    
    wh = tf.pow(coords[:,:,:,2:4], 2) *  np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] 
    centers = coords[:,:,:,0:2] 
    floor = centers - (wh * .5) 
    ceil  = centers + (wh * .5) 

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft) 
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.mul(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])
    
    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True)) 
    best_box = tf.to_float(best_box)
    confs = tf.mul(best_box, _confs) 

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(3, 4 * [tf.expand_dims(confs, -1)])
    cooid = scoor * weight_coo
    weight_pro = tf.concat(3, C * [tf.expand_dims(confs, -1)]) 
    proid = sprob * weight_pro 

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat(3, [_coord, tf.expand_dims(confs, 3), _probs ])
    wght = tf.concat(3, [cooid, tf.expand_dims(conid, 3), proid ])

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.mul(loss, wght)
    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)


