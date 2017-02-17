from cfg.process import cfg_yielder
from .darkop import create_darkop
from utils import loader
import warnings
import time
import os

class Darknet(object):

    _EXT = '.weights'

    def __init__(self, FLAGS):
        self.get_weight_src(FLAGS)
        self.modify = False

        print('Parsing {}'.format(self.src_cfg))
        src_parsed = self.parse_cfg(self.src_cfg, FLAGS)
        self.src_meta, self.src_layers = src_parsed
        
        if self.src_cfg == FLAGS.model:
            self.meta, self.layers = src_parsed
        else: 
        	print('Parsing {}'.format(FLAGS.model))
        	des_parsed = self.parse_cfg(FLAGS.model, FLAGS)
        	self.meta, self.layers = des_parsed

        self.load_weights()

    def get_weight_src(self, FLAGS):
        """
        analyse FLAGS.load to know where is the 
        source binary and what is its config.
        can be: None, FLAGS.model, or some other
        """
        self.src_bin = FLAGS.model + self._EXT
        self.src_bin = FLAGS.binary + self.src_bin
        self.src_bin = os.path.abspath(self.src_bin)
        exist = os.path.isfile(self.src_bin)

        if FLAGS.load == str(): FLAGS.load = int()
        if type(FLAGS.load) is int:
            self.src_cfg = FLAGS.model
            if FLAGS.load: self.src_bin = None
            elif not exist: self.src_bin = None
        else:
            assert os.path.isfile(FLAGS.load), \
            '{} not found'.format(FLAGS.load)
            self.src_bin = FLAGS.load
            name = loader.model_name(FLAGS.load)
            cfg_path = FLAGS.config+name+'.cfg'
            if not os.path.isfile(cfg_path):
                warnings.warn(
                    '{} not found, use {} instead'.format(
                    cfg_path, FLAGS.model))
                cfg_path = FLAGS.model
            self.src_cfg = cfg_path
            FLAGS.load = int()


    def parse_cfg(self, model, FLAGS):
        """
        return a list of `layers` objects (darkop.py)
        given path to binaries/ and configs/
        """
        args = [model, FLAGS.binary]
        cfg_layers = cfg_yielder(*args)
        meta = dict(); layers = list()
        for i, info in enumerate(cfg_layers):
            if i == 0: meta = info; continue
            else: new = create_darkop(*info)
            layers.append(new)
        return meta, layers

    def load_weights(self):
        """
        Use `layers` and Loader to load .weights file
        """
        print('Loading {} ...'.format(self.src_bin))
        start = time.time()

        args = [self.src_bin, self.src_layers]
        wgts_loader = loader.create_loader(*args)
        for layer in self.layers: layer.load(wgts_loader)
        
        stop = time.time()
        print('Finished in {}s'.format(stop - start))