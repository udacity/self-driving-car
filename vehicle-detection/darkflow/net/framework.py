from . import yolo
from . import yolov2
from . import vanilla

class framework(object):
    constructor = vanilla.constructor
    loss = vanilla.train.loss
    
    def __init__(self, meta, FLAGS):
        model = meta['model'].split('/')[-1]
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        
        self.constructor(meta, FLAGS)

    def is_inp(self):
        return True

class YOLO(framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    preprocess = yolo.test.preprocess
    postprocess = yolo.test.postprocess
    loss = yolo.train.loss
    is_inp = yolo.misc.is_inp
    profile = yolo.misc.profile
    _batch = yolo.data._batch

class YOLOv2(framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolov2.data.shuffle
    preprocess = yolo.test.preprocess
    loss = yolov2.train.loss
    is_inp = yolo.misc.is_inp
    postprocess = yolov2.test.postprocess
    _batch = yolov2.data._batch

"""
framework factory
"""

types = {
    '[detection]': YOLO,
    '[region]': YOLOv2
}

def create_framework(meta, FLAGS):
    net_type = meta['type']
    this = types.get(net_type, framework)
    return this(meta, FLAGS)