import numpy as np

class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

def prob_compare(box):
    return box.probs[box.class_num]

def prob_compare2(boxa, boxb):
    if (boxa.pi < boxb.pi):
        return 1
    elif(boxa.pi == boxb.pi):
        return 0
    else:
        return -1