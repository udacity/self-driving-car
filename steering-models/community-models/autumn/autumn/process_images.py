import cv2
import scipy.misc
import numpy as np
import csv
import os
import argparse

DATA_DIR = '/vol/data'
INPUT_CSV = 'train_center.csv'
WINDOW_SIZE = 7
OUTPUT_DIR = 'flow_%d_local' % WINDOW_SIZE

parser = argparse.ArgumentParser(description='Convert files to 3-channel mean dense optical flow')
parser.add_argument('--data-dir', '--data', action='store', dest='data_dir',
                    default=DATA_DIR, help='Directory containing original images')
parser.add_argument('--input-csv', '--input', '-i', action='store', dest='input_csv',
                    default=INPUT_CSV, help='CSV file containing list of file names')
parser.add_argument('--input-type', '--input-ext', action='store', dest='input_type',
                    default='jpg', help='File type extension of input images')
parser.add_argument('--output-dir', '--output', '-o', action='store', dest='output_dir',
                    default=OUTPUT_DIR, help='Name of directory to store converted images in')
parser.add_argument('--window-size', '--window', action='store', dest='window_size', default=WINDOW_SIZE)
parser.add_argument('--show', action='store_true', dest='show_image')
parser.add_argument('--average-polar', action='store_true', dest='average_polar')
args = parser.parse_args()

files = []
input_type = '.' + args.input_type

with open(args.input_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['frame_id']
        files.append(filename)

last = []
prev_image = None

for i, filename in enumerate(files):
    img = scipy.misc.imread(args.data_dir + '/' + files[i] + input_type, mode='RGB')
    prev = prev_image if prev_image is not None else img
    prev_image = img
    prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    last.append(flow)
    if len(last) > args.window_size:
        last.pop(0)

    avg_flow = sum(last) / len(last)
    mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])

    hsv = np.zeros_like(prev_image)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if args.show_image:
        cv2.imshow('flow', bgr)
        cv2.waitKey(1)

    if not os.path.exists(args.data_dir + '/' + args.output_dir):
        os.makedirs(args.data_dir + '/' + args.output_dir)
    cv2.imwrite(args.data_dir + '/' + args.output_dir + '/' + files[i] + '.png', bgr)
    print('Saving to ' + args.data_dir + '/' + args.output_dir + '/' + files[i] + '.png')
