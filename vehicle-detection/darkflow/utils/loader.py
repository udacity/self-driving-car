import tensorflow as tf
import os
import dark
import numpy as np

class loader(object):
    """
    interface to work with both .weights and .ckpt files
    in loading / recollecting / resolving mode
    """
    VAR_LAYER = ['convolutional', 'connected', 'local', 
                 'select', 'conv-select',
                 'extract', 'conv-extract']

    def __init__(self, *args):
        self.src_key = list()
        self.vals = list()
        self.load(*args)

    def __call__(self, key):
        for idx in range(len(key)):
            val = self.find(key, idx)
            if val is not None: return val
        return None
    
    def find(self, key, idx):
        up_to = min(len(self.src_key), 4)
        for i in range(up_to):
            key_b = self.src_key[i]
            if key_b[idx:] == key[idx:]:
                return self.yields(i)
        return None

    def yields(self, idx):
        del self.src_key[idx]
        temp = self.vals[idx]
        del self.vals[idx]
        return temp

class weights_loader(loader):
    """one who understands .weights files"""
    
    _W_ORDER = dict({ # order of param flattened into .weights file
        'convolutional': [
            'biases','gamma','moving_mean','moving_variance','kernel'
        ],
        'connected': ['biases', 'weights'],
        'local': ['biases', 'kernels']
    })

    def load(self, path, src_layers):
        self.src_layers = src_layers
        walker = weights_walker(path)

        for i, layer in enumerate(src_layers):
            if layer.type not in self.VAR_LAYER: continue
            self.src_key.append([layer])
            
            if walker.eof: new = None
            else: 
                args = layer.signature
                new = dark.darknet.create_darkop(*args)
            self.vals.append(new)

            if new is None: continue
            order = self._W_ORDER[new.type]
            for par in order:
                if par not in new.wshape: continue
                val = walker.walk(new.wsize[par])
                new.w[par] = val
            new.finalize(walker.transpose)

        if walker.path is not None:
            #assert walker.offset == walker.size, \
            #'expect {} bytes, found {}'.format(
            #    walker.offset, walker.size)
            print('Successfully identified {} bytes'.format(
                walker.offset))

class checkpoint_loader(loader):
    """
    one who understands .ckpt files, very much
    """
    def load(self, ckpt, ignore):
        meta = ckpt + '.meta'
        with tf.Graph().as_default() as graph:
            with tf.Session().as_default() as sess:
                saver = tf.train.import_meta_graph(meta)
                saver.restore(sess, ckpt)
                for var in tf.global_variables():
                    name = var.name.split(':')[0]
                    packet = [name, var.get_shape().as_list()]
                    self.src_key += [packet]
                    self.vals += [var.eval(sess)]

def create_loader(path, cfg = None):
    if path is None:
        load_type = weights_loader
    elif '.weights' in path:
        load_type = weights_loader
    else: 
        load_type = checkpoint_loader
    
    return load_type(path, cfg)

class weights_walker(object):
    """incremental reader of float32 binary files"""
    def __init__(self, path):
        self.eof = False # end of file
        self.path = path  # current pos
        if path is None: 
            self.eof = True
            return
        else: 
            self.size = os.path.getsize(path)# save the path
            major, minor, revision, seen = np.memmap(path,
                shape = (), mode = 'r', offset = 0,
                dtype = '({})i4,'.format(4))
            self.transpose = major > 1000 or minor > 1000
            self.offset = 16

    def walk(self, size):
        if self.eof: return None
        end_point = self.offset + 4 * size
        assert end_point <= self.size, \
        'Over-read {}'.format(self.path)

        float32_1D_array = np.memmap(
            self.path, shape = (), mode = 'r', 
            offset = self.offset,
            dtype='({})float32,'.format(size)
        )

        self.offset = end_point
        if end_point == self.size: 
            self.eof = True
        return float32_1D_array

def model_name(file_path):
    file_name = file_path.split('/')[-1]
    ext = str()
    if '.' in file_name: # exclude extension
        file_name = file_name.split('.')
        ext = file_name[-1]
        file_name = '.'.join(file_name[:-1])
    if ext == str() or ext == 'meta': # ckpt file
        file_name = file_name.split('-')
        num = int(file_name[-1])
        return '-'.join(file_name[:-1])
    if ext == 'weights':
        return file_name
