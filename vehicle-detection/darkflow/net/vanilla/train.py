_LOSS_TYPE = ['sse','l2', 'smooth', 
			  'sparse', 'l1', 'softmax',
			  'svm', 'fisher']

def loss(self, net_out):
	m = self.meta
	loss_type = self.meta['type']
	assert loss_type in _LOSS_TYPE, \
	'Loss type {} not implemented'.format(loss_type)

	out = net_out
	out_shape = out.get_shape()
	out_dtype = out.dtype.base_dtype
	_truth = tf.placeholders(out_dtype, out_shape)

	self.placeholders = dict({
			'truth': _truth
		})

	diff = _truth - out
	if loss_type in ['sse','12']:
		loss = tf.nn.l2_loss(diff)

	elif loss_type == ['smooth']:
		small = tf.cast(diff < 1, tf.float32)
		large = 1. - small
		l1_loss = tf.nn.l1_loss(tf.mul(diff, large))
		l2_loss = tf.nn.l2_loss(tf.mul(diff, small))
		loss = l1_loss + l2_loss

	elif loss_type in ['sparse', 'l1']:
		loss = l1_loss(diff)

	elif loss_type == 'softmax':
		loss = tf.nn.softmax_cross_entropy_with_logits(logits, y)
		loss = tf.reduce_mean(loss)

	elif loss_type == 'svm':
		assert 'train_size' in m, \
		'Must specify'
		size = m['train_size']
		self.nu = tf.Variable(tf.ones([train_size, num_classes]))