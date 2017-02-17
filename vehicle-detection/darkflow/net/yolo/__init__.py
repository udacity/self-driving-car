from . import train
from . import test
from . import data
from . import misc
import numpy as np


""" YOLO framework __init__ equivalent"""

def constructor(self, meta, FLAGS):

	def _to_color(indx, base):
		""" return (b, r, g) tuple"""
		base2 = base * base
		b = 2 - indx / base2
		r = 2 - (indx % base2) / base
		g = 2 - (indx % base2) % base
		return (b * 127, r * 127, g * 127)

	misc.labels(meta)
	assert len(meta['labels']) == meta['classes'], (
		'labels.txt and {} indicate' + ' '
		'inconsistent class numbers'
	).format(meta['model'])
	colors = list()
	base = int(np.ceil(pow(meta['classes'], 1./3)))
	for x in range(len(meta['labels'])): 
		colors += [_to_color(x, base)]
	meta['colors'] = colors
	self.fetch = list()
	self.meta, self.FLAGS = meta, FLAGS