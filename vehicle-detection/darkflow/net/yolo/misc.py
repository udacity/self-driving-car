import pickle
import numpy as np
import cv2
import os

labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]

# 8, 14, 15, 19

voc_models = ['yolo-full', 'yolo-tiny', 'yolo-small',  # <- v1
              'yolov1', 'tiny-yolov1', # <- v1.1 
              'tiny-yolo-voc', 'yolo-voc'] # <- v2

coco_models = ['tiny-coco', 'yolo-coco',  # <- v1.1
               'yolo', 'tiny-yolo'] # <- v2

coco_names = 'coco.names'

def labels(meta):    
    model = meta['name']
    if model in voc_models: 
        meta['labels'] = labels20
    else:
        file = 'labels.txt'
        if model in coco_models: 
            file = os.path.join('cfg',coco_names)
        with open(file, 'r') as f:
            meta['labels'] = list()
            labs = [l.strip() for l in f.readlines()]
            for lab in labs:
                if lab == '----': break
                meta['labels'] += [lab]
    if len(meta['labels']) == 0: 
        meta['labels'] = labels20

def is_inp(self, name): 
    return name[-4:] in ['.jpg','.JPG', '.jpeg', '.JPEG']

def show(im, allobj, S, w, h, cellx, celly):
    for obj in allobj:
        a = obj[5] % S
        b = obj[5] // S
        cx = a + obj[1]
        cy = b + obj[2]
        centerx = cx * cellx
        centery = cy * celly
        ww = obj[3]**2 * w
        hh = obj[4]**2 * h
        cv2.rectangle(im,
            (int(centerx - ww/2), int(centery - hh/2)),
            (int(centerx + ww/2), int(centery + hh/2)),
            (0,0,255), 2)
    cv2.imshow("result", im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def show2(im, allobj):
    for obj in allobj:
        cv2.rectangle(im,
            (obj[1], obj[2]), 
            (obj[3], obj[4]), 
            (0,0,255),2)
    cv2.imshow('result', im)
    cv2.waitKey()
    cv2.destroyAllWindows()


_MVA = .05

def profile(self, net):
    pass
#     data = self.parse(exclusive = True)
#     size = len(data); batch = self.FLAGS.batch
#     all_inp_ = [x[0] for x in data]
#     net.say('Will cycle through {} examples {} times'.format(
#         len(all_inp_), net.FLAGS.epoch))

#     fetch = list(); mvave = list(); names = list();
#     this = net.top
#     conv_lay = ['convolutional', 'connected', 'local', 'conv-select']
#     while this.inp is not None:
#         if this.lay.type in conv_lay:
#             fetch = [this.out] + fetch
#             names = [this.lay.signature] + names
#             mvave = [None] + mvave 
#         this = this.inp
#     print(names)

#     total = int(); allofthem = len(all_inp_) * net.FLAGS.epoch
#     batch = min(net.FLAGS.batch, len(all_inp_))
#     for count in range(net.FLAGS.epoch):
#         net.say('EPOCH {}'.format(count))
#         for j in range(len(all_inp_)/batch):
#             inp_feed = list(); new_all = list()
#             all_inp = all_inp_[j*batch: (j*batch+batch)]
#             for inp in all_inp:
#                 new_all += [inp]
#                 this_inp = os.path.join(net.FLAGS.dataset, inp)
#                 this_inp = net.framework.preprocess(this_inp)
#                 expanded = np.expand_dims(this_inp, 0)
#                 inp_feed.append(expanded)
#             all_inp = new_all
#             feed_dict = {net.inp : np.concatenate(inp_feed, 0)}
#             out = net.sess.run(fetch, feed_dict)

#             for i, o in enumerate(out):
#                 oi = out[i];
#                 dim = len(oi.shape) - 1
#                 ai = mvave[i]; 
#                 mi = np.mean(oi, tuple(range(dim)))
#                 vi = np.var(oi, tuple(range(dim)))
#                 if ai is None: mvave[i] = [mi, vi]
#                 elif 'banana ninja yada yada':
#                     ai[0] = (1 - _MVA) * ai[0] + _MVA * mi
#                     ai[1] = (1 - _MVA) * ai[1] + _MVA * vi
#             total += len(inp_feed)
#             net.say('{} / {} = {}%'.format(
#                 total, allofthem, 100. * total / allofthem))

#         with open('profile', 'wb') as f:
#             pickle.dump([mvave], f, protocol = -1)
