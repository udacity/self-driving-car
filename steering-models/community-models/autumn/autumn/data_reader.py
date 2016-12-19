import scipy.misc
import random
import csv

DATA_DIR = '/vol/data/'
FILE_EXT = '.png'


class DataReader(object):
    def __init__(self, data_dir=DATA_DIR, file_ext=FILE_EXT, sequential=False):
        self.load()

    def load(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        total = 0
        count01 = count005 = count002 = count0 = 0

        with open('interpolated_center.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                angle = float(row['steering_angle'])
                if angle > 0.1 or angle < -0.1 and random.random() > 0.2:
                    xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                    ys.append(row['steering_angle'])
                    count01 += 1
                elif (angle > 0.05 or angle < -0.5) and random.random() > 0.2:
                    xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                    ys.append(row['steering_angle'])
                    count005 += 1
                elif (angle > 0.02 or angle < -0.02) and random.random() > 0.7:
                    xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                    ys.append(row['steering_angle'])
                    count002 += 1
                elif random.random() > 0.8:
                    xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                    ys.append(row['steering_angle'])
                    count0 += 1
                total += 1

        with open('train_center.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                angle = float(row['steering_angle'])
                xs.append(DATA_DIR + 'Ch2_Train/center/flow_7_local/' + row['frame_id'] + FILE_EXT)
                ys.append(row['steering_angle'])
                total += 1

        print('> 0.1 or < -0.1: ' + str(count01))
        print('> 0.05 or < -0.05: ' + str(count005))
        print('> 0.02 or < -0.02: ' + str(count002))
        print('~0: ' + str(count0))
        print('Total data: ' + str(total))

        self.num_images = len(xs)

        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 0.8)]
        self.train_ys = ys[:int(len(xs) * 0.8)]

        self.val_xs = xs[-int(len(xs) * 0.2):]
        self.val_ys = ys[-int(len(xs) * 0.2):]

        self.num_train_images = len(self.train_xs)
        self.num_val_images = len(self.val_xs)

    def load_train_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out

    def load_val_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.val_xs[(self.val_batch_pointer + i) % self.num_val_images])
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.val_ys[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return x_out, y_out

    def load_seq(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        print('LSTM Data')

        with open('train_center.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(DATA_DIR + 'Ch2_Train/center/flow_7_local/' + row['frame_id'] + FILE_EXT)
                ys.append(row['steering_angle'])

        c = list(zip(xs, ys))
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 1.0)]
        self.train_ys = ys[:int(len(xs) * 1.0)]

        self.num_images = len(self.train_xs)
        print('total: ' + str(self.num_images))

        self.num_train_images = len(self.train_xs)

    def load_seq_2(self):
        xs = []
        ys = []

        self.train_batch_pointer = 0
        print('LSTM Data')

        with open('interpolated_center.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(DATA_DIR + 'output/left/flow_7_cart/' + row['frame_id'] + FILE_EXT)
                ys.append(row['steering_angle'])

        c = list(zip(xs, ys))
        xs, ys = zip(*c)

        self.train_xs = xs[:int(len(xs) * 1.0)]
        self.train_ys = ys[:int(len(xs) * 1.0)]

        self.num_images = len(self.train_xs)
        print('total: ' + str(self.num_images))

        self.num_train_images = len(self.train_xs)

    def skip(self, num):
        self.train_batch_pointer += num

    def load_seq_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_train_images])
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out
