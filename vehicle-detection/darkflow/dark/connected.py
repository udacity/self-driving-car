from .layer import Layer
import numpy as np

class extract_layer(Layer):
    def setup(self, old_inp, old_out,
              activation, inp, out):
        if inp is None: inp = range(old_inp)
        self.activation = activation
        self.old_inp = old_inp
        self.old_out = old_out
        self.inp = inp
        self.out = out
        self.wshape = {
            'biases': [len(self.out)],
            'weights': [len(self.inp), len(self.out)]
        }

    @property
    def signature(self):
        sig = ['connected']
        sig += self._signature[1:-2]
        return sig

    def present(self):
        args = self.signature
        self.presenter = connected_layer(*args)

    def recollect(self, val):
        w = val['weights']
        b = val['biases']
        if w is None: self.w = val; return
        w = np.take(w, self.inp, 0)
        w = np.take(w, self.out, 1)
        b = np.take(b, self.out)
        assert1 = w.shape == tuple(self.wshape['weights'])
        assert2 = b.shape == tuple(self.wshape['biases'])
        assert assert1 and assert2, \
        'Dimension does not match in {} recollect'.format(
            self._signature)
        
        self.w['weights'] = w
        self.w['biases'] = b
    


class select_layer(Layer):
    def setup(self, inp, old, 
              activation, inp_idx,
              out, keep, train):
        self.old = old
        self.keep = keep
        self.train = train
        self.inp_idx = inp_idx
        self.activation = activation
        inp_dim = inp
        if inp_idx is not None:
            inp_dim = len(inp_idx)
        self.inp = inp_dim
        self.out = out
        self.wshape = {
            'biases': [out],
            'weights': [inp_dim, out]
        }

    @property
    def signature(self):
        sig = ['connected']
        sig += self._signature[1:-4]
        return sig

    def present(self):
        args = self.signature
        self.presenter = connected_layer(*args)

    def recollect(self, val):
        w = val['weights']
        b = val['biases']
        if w is None: self.w = val; return
        if self.inp_idx is not None:
            w = np.take(w, self.inp_idx, 0)
            
        keep_b = np.take(b, self.keep)
        keep_w = np.take(w, self.keep, 1)
        train_b = b[self.train:]
        train_w = w[:, self.train:]
        self.w['biases'] = np.concatenate(
            (keep_b, train_b), axis = 0)
        self.w['weights'] = np.concatenate(
            (keep_w, train_w), axis = 1)


class connected_layer(Layer):
    def setup(self, input_size, 
              output_size, activation):
        self.activation = activation
        self.inp = input_size
        self.out = output_size
        self.wshape = {
            'biases': [self.out],
            'weights': [self.inp, self.out]
        }

    def finalize(self, transpose):
        weights = self.w['weights']
        if weights is None: return
        shp = self.wshape['weights']
        if not transpose:
            weights = weights.reshape(shp[::-1])
            weights = weights.transpose([1,0])
        else: weights = weights.reshape(shp)
        self.w['weights'] = weights