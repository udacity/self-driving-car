from utils import loader
import numpy as np

class Layer(object):

    def __init__(self, *args):
        self._signature = list(args)
        self.type = list(args)[0]
        self.number = list(args)[1]

        self.w = dict() # weights
        self.h = dict() # placeholders
        self.wshape = dict() # weight shape
        self.wsize = dict() # weight size
        self.setup(*args[2:]) # set attr up
        self.present()
        for var in self.wshape:
            shp = self.wshape[var]
            size = np.prod(shp)
            self.wsize[var] = size

    def load(self, src_loader):
        var_lay = src_loader.VAR_LAYER
        if self.type not in var_lay: return

        src_type = type(src_loader)
        if src_type is loader.weights_loader:
            wdict = self.load_weights(src_loader)
        else: 
            wdict = self.load_ckpt(src_loader)
        if wdict is not None:
            self.recollect(wdict)

    def load_weights(self, src_loader):
        val = src_loader([self.presenter])
        if val is None: return None
        else: return val.w

    def load_ckpt(self, src_loader):
        result = dict()
        presenter = self.presenter
        for var in presenter.wshape:
            name = presenter.varsig(var)
            shape = presenter.wshape[var]
            key = [name, shape]
            val = src_loader(key)
            result[var] = val
        return result

    @property
    def signature(self):
        return self._signature

    # For comparing two layers
    def __eq__(self, other):
        return self.signature == other.signature
    def __ne__(self, other):
        return not self.__eq__(other)

    def varsig(self, var):
        if var not in self.wshape:
            return None
        sig = str(self.number)
        sig += '-' + self.type
        sig += '/' + var
        return sig

    def recollect(self, w): self.w = w
    def present(self): self.presenter = self
    def setup(self, *args): pass
    def finalize(self): pass 