from .layer import Layer
import numpy as np

class local_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, w_, h_, activation):
        self.pad = pad * (ksize / 2)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.h_out = h_
        self.w_out = w_

        self.dnshape = [h_ * w_, n, c, ksize, ksize]
        self.wshape = dict({
            'biases': [h_ * w_ * n],
            'kernels': [h_ * w_, ksize, ksize, c, n]
        })

    def finalize(self, _):
        weights = self.w['kernels']
        if weights is None: return
        weights = weights.reshape(self.dnshape)
        weights = weights.transpose([0,3,4,2,1])
        self.w['kernels'] = weights

class conv_extract_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation,
              inp, out):
        if inp is None: inp = range(c)
        self.activation = activation
        self.batch_norm = batch_norm
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.inp = inp
        self.out = out
        self.wshape = dict({
            'biases': [len(out)], 
            'kernel': [ksize, ksize, len(inp), len(out)]
        })

    @property
    def signature(self):
        sig = ['convolutional']
        sig += self._signature[1:-2]
        return sig

    def present(self):
        args = self.signature
        self.presenter = convolutional_layer(*args)

    def recollect(self, w):
        if w is None:
            self.w = w
            return
        k = w['kernel']
        b = w['biases']
        k = np.take(k, self.inp, 2)
        k = np.take(k, self.out, 3)
        b = np.take(b, self.out)
        assert1 = k.shape == tuple(self.wshape['kernel'])
        assert2 = b.shape == tuple(self.wshape['biases'])
        assert assert1 and assert2, \
        'Dimension not matching in {} recollect'.format(
            self._signature)
        self.w['kernel'] = k
        self.w['biases'] = b


class conv_select_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation,
              keep_idx, real_n):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.keep_idx = keep_idx
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.wshape = dict({
            'biases': [real_n], 
            'kernel': [ksize, ksize, c, real_n]
        })
        if self.batch_norm:
            self.wshape.update({
                'moving_variance'  : [real_n], 
                'moving_mean': [real_n], 
                'gamma' : [real_n]
            })
            self.h['is_training'] = {
                'shape': (),
                'feed': True,
                'dfault': False
            }

    @property
    def signature(self):
        sig = ['convolutional']
        sig += self._signature[1:-2]
        return sig
    
    def present(self):
        args = self.signature
        self.presenter = convolutional_layer(*args)

    def recollect(self, w):
        if w is None:
            self.w = w
            return
        idx = self.keep_idx
        k = w['kernel']
        b = w['biases']
        self.w['kernel'] = np.take(k, idx, 3) 
        self.w['biases'] = np.take(b, idx)
        if self.batch_norm:
            m = w['moving_mean']
            v = w['moving_variance']
            g = w['gamma']
            self.w['moving_mean'] = np.take(m, idx)
            self.w['moving_variance'] = np.take(v, idx)
            self.w['gamma'] = np.take(g, idx)

class convolutional_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.dnshape = [n, c, ksize, ksize] # darknet shape
        self.wshape = dict({
            'biases': [n], 
            'kernel': [ksize, ksize, c, n]
        })
        if self.batch_norm:
            self.wshape.update({
                'moving_variance'  : [n], 
                'moving_mean': [n], 
                'gamma' : [n]
            })
            self.h['is_training'] = {
                'feed': True,
                'dfault': False,
                'shape': ()
            }

    def finalize(self, _):
        """deal with darknet"""
        kernel = self.w['kernel']
        if kernel is None: return
        kernel = kernel.reshape(self.dnshape)
        kernel = kernel.transpose([2,3,1,0])
        self.w['kernel'] = kernel