import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from .misc import show
import numpy as np
import os

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
    S, B, C = m['side'], m['num'], m['classes']
    SS = S * S # number of grid cells

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tside    = {}'.format(m['side']))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, SS, C]
    size2 = [None, SS, B]

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
    coords = net_out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])
    wh = tf.pow(coords[:,:,:,2:4], 2) * S # unit: grid cell
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] # unit: grid cell^2 
    centers = coords[:,:,:,0:2] # [batch, SS, B, 2]
    floor = centers - (wh * .5) # [batch, SS, B, 2]
    ceil  = centers + (wh * .5) # [batch, SS, B, 2]

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
    proid = sprob * _proid

    # flatten 'em all
    probs = slim.flatten(_probs)
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)
    conid = slim.flatten(conid)
    coord = slim.flatten(_coord)
    cooid = slim.flatten(cooid)

    self.fetch += [probs, confs, conid, cooid, proid]
    true = tf.concat(1, [probs, confs, coord])
    wght = tf.concat(1, [proid, conid, cooid])

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(net_out - true, 2)
    loss = tf.mul(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
