import tensorflow.contrib.slim as slim
from .baseop import BaseOp
import tensorflow as tf
import numpy as np

class reorg(BaseOp):
    def _forward(self):
        inp = self.inp.out
        shape = inp.get_shape().as_list()
        _, h, w, c = shape
        s = self.lay.stride
        out = list()
        for i in range(h/s):
            row_i = list()
            for j in range(w/s):
                si, sj = s * i, s * j
                boxij = inp[:, si: si+s, sj: sj+s,:]
                flatij = tf.reshape(boxij, [-1,1,1,c*s*s])
                row_i += [flatij]
            out += [tf.concat(2, row_i)]
        self.out = tf.concat(1, out)

    def forward(self):
        inp = self.inp.out
        s = self.lay.stride
        self.out = tf.extract_image_patches(
            inp, [1,s,s,1], [1,s,s,1], [1,1,1,1], 'VALID')

    def speak(self):
        args = [self.lay.stride] * 2
        msg = 'local flatten {}x{}'
        return msg.format(*args)


class local(BaseOp):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])

        k = self.lay.w['kernels']
        ksz = self.lay.ksize
        half = ksz/2
        out = list()
        for i in range(self.lay.h_out):
            row_i = list()
            for j in range(self.lay.w_out):
                kij = k[i * self.lay.w_out + j]
                i_, j_ = i + 1 - half, j + 1 - half
                tij = temp[:, i_ : i_ + ksz, j_ : j_ + ksz,:]
                row_i.append(
                    tf.nn.conv2d(tij, kij, 
                        padding = 'VALID', 
                        strides = [1] * 4))
            out += [tf.concat(2, row_i)]

        self.out = tf.concat(1, out)

    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.activation]
        msg = 'loca {}x{}p{}_{}  {}'.format(*args)
        return msg

class convolutional(BaseOp):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])
        temp = tf.nn.conv2d(temp, self.lay.w['kernel'], padding = 'VALID', 
            name = self.scope, strides = [1] + [self.lay.stride] * 2 + [1])
        if self.lay.batch_norm: 
            temp = self.batchnorm(self.lay, temp)
        self.out = tf.nn.bias_add(temp, self.lay.w['biases'])

    def batchnorm(self, layer, inp):
        if not self.var:
            temp = (inp - layer.w['moving_mean'])
            temp /= (np.sqrt(layer.w['moving_variance']) + 1e-5)
            temp *= layer.w['gamma']
            return temp
        else:
            args = dict({
                'center' : False, 'scale' : True,
                'epsilon': 1e-5, 'scope' : self.scope,
                'updates_collections' : None,
                'is_training': layer.h['is_training']
                })
            v = tf.__version__.split('.')[1]
            if int(v) < 12: key = 'initializers'
            else: key = 'param_initializers'
            args.update({key : layer.w})
            return slim.batch_norm(inp, **args)

    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'conv {}x{}p{}_{}  {}  {}'.format(*args)
        return msg

class conv_select(convolutional):
    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'sele {}x{}p{}_{}  {}  {}'.format(*args)
        return msg

class conv_extract(convolutional):
    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'extr {}x{}p{}_{}  {}  {}'.format(*args)
        return msg