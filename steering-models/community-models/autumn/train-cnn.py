import os
import tensorflow as tf
from autumn import ConvModel, DataReader
import argparse

BATCH_SIZE = 100
DATA_DIR = '/vol/data'
LOGDIR = './train_model'
CHECKPOINT_EVERY = 100
NUM_STEPS = int(1e5)
CKPT_FILE = 'model.ckpt'
LEARNING_RATE = 1e-3
KEEP_PROB = 0.8
L2_REG = 0
EPSILON = 0.001
MOMENTUM = 0.9


def get_arguments():
    parser = argparse.ArgumentParser(description='ConvNet training')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch.')
    parser.add_argument('--data_dir', '--data', type=str, default=DATA_DIR,
                        help='The directory containing the training data.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Storing debug information for TensorBoard.')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Directory for log files.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--keep_prob', type=float, default=KEEP_PROB,
                        help='Dropout keep probability.')
    parser.add_argument('--l2_reg', type=float,
                        default=L2_REG)
    return parser.parse_args()


def main():
    args = get_arguments()
    sess = tf.Session()

    model = ConvModel()
    train_vars = tf.trainable_variables()
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.y_, model.y)))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * args.l2_reg
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    start_step = 0
    if args.restore_from is not None:
        saver.restore(sess, args.logdir + '/' + args.restore_from)
        start_step = float(args.restore_from.split('step-')[0].split('-')[-1])
        print('Model restored from ' + args.logdir + '/' + args.restore_from)

    if args.store_metadata:
        tf.scalar_summary("loss", loss)
        merged_summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(args.logdir, graph=tf.get_default_graph())

    min_loss = 1.0
    data_reader = DataReader()

    for i in range(start_step, start_step + args.num_steps):
        xs, ys = data_reader.load_train_batch(args.batch_size)
        train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob})
        train_error = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        print("Step %d, train loss %g" % (i, train_error))

        if i % 10 == 0:
            xs, ys = data_reader.load_val_batch(args.batch_size)
            val_error = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
            print("Step %d, val loss %g" % (i, val_error))
            if i > 0 and i % args.checkpoint_every == 0:
                if not os.path.exists(args.logdir):
                    os.makedirs(args.logdir)
                    checkpoint_path = os.path.join(args.logdir, "model-step-%d-val-%g.ckpt" % (i, val_error))
                    filename = saver.save(sess, checkpoint_path)
                    print("Model saved in file: %s" % filename)
                elif val_error < min_loss:
                    min_loss = val_error
                    if not os.path.exists(args.logdir):
                        os.makedirs(args.logdir)
                    checkpoint_path = os.path.join(args.logdir, "model-step-%d-val-%g.ckpt" % (i, val_error))
                    filename = saver.save(sess, checkpoint_path)
                    print("Model saved in file: %s" % filename)

if __name__ == '__main__':
    main()
