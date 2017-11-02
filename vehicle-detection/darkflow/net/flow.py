import os
import time
import numpy as np
import tensorflow as tf
import pickle

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op] 
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)


def predict(self):
    inp_path = self.FLAGS.test
    all_inp_ = os.listdir(inp_path)
    all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]
    if not all_inp_:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inp_))

    for j in range(len(all_inp_) // batch):
        inp_feed = list(); new_all = list()
        all_inp = all_inp_[j*batch: (j*batch+batch)]
        for inp in all_inp:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        all_inp = new_all

        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            self.framework.postprocess(prediction,
                os.path.join(inp_path, all_inp[i]))
        stop = time.time(); last = stop - start

        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))