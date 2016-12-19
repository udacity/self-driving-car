"""
Evaluate the final test set.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from math import cos, sin, pi

import cv2
import numpy as np
from progress.bar import IncrementalBar
from scipy.stats.mstats import mquantiles

from datasets import load_dataset
from models import load_from_config

FNULL = open(os.devnull, 'w')

def main():
    default_model_config = {
        'timesteps': 10,
        'scale': 16.0,
        'type': 'transfer-lstm',
        'transform_model_config': {
            'model_uri': '/models/output/1480050604.h5',
            'scale': 16,
            'type': 'regression'
        },
        'model_uri': '/models/output/1480098607.h5'
    }

    ts = int(time.time())
    default_submission_path = '/submissions/csv/submission.%s.csv' % ts
    default_video_path = '/submissions/videos/video.%s.mp4' % ts

    parser = argparse.ArgumentParser(
        description='Build submission csv and video overlay')
    parser.add_argument('--submission_path', type=str,
                        default=default_submission_path,
                        help='Path to write submission output file to')
    parser.add_argument('--video_path', type=str,
                        default=default_video_path,
                        help='Path to write overlay video to')
    parser.add_argument('--test_dir', type=str,
                        default='/datasets/showdown_raw/test/center/',
                        help='Directory containing test images')
    parser.add_argument('--model_config', type=str,
                        default=json.dumps(default_model_config),
                        help='Model config JSON to use')
    parser.add_argument('--regenerate', type=bool,
                        default=False,
                        help='If true, generates submission file if it exists')
    args = parser.parse_args()

    generate_submission(
        args.model_config,
        args.test_dir,
        args.submission_path,
        args.video_path,
        regenerate=args.regenerate)

def generate_submission(model_config,
                        images_path,
                        submission_path,
                        video_path,
                        regenerate=True):

    if regenerate or not os.path.exists(submission_path):
        print 'Generating submission file at', submission_path
        model_config = json.loads(model_config)
        model_type = model_config['type']
        model = load_from_config(model_config)

        if model_type == 'lstm':
            # NOTE: we could migrate this to make_stateful_predictor
            model.model.summary()
            timesteps = model.timesteps
            predictor = lambda x: model.predict_on_batch(x)[0][0]
            generate_lstm_submission(
                predictor,
                timesteps,
                images_path,
                submission_path)
        else:
            predictor = model.make_stateful_predictor()
            predictor = smoothing_predictor(predictor)

            generate_submission_csv(
                predictor,
                images_path,
                submission_path)

    temp_dir = tempfile.mkdtemp()
    try:
        generate_video(
            submission_path,
            images_path,
            video_path,
            temp_dir)
    finally:
        shutil.rmtree(temp_dir)


def generate_submission_csv(predictor, images_path, output_path):
    filenames = sorted(os.listdir(images_path))
    progress_bar = IncrementalBar(
        'Generating submission',
        max=len(filenames),
        suffix='%(percent).1f%% - %(eta)ds')

    with open(output_path, 'w') as f:
        f.write('frame_id,steering_angle\n')
        for filename in sorted(os.listdir(images_path)):
            src = os.path.join(images_path, filename)
            x = load_test_image(src).reshape((1, 120, 320, 3))
            y_pred = predictor(x)
            filename = filename.split('.')[0]
            f.write('%s,%s\n' % (filename, y_pred))
            progress_bar.next()

    print ''


def generate_lstm_submission(predictor, timesteps, images_path, output_path):
    input_shape = (120, 320, 3)
    with open(output_path, 'w') as f:
        f.write('frame_id,steering_angle\n')
        filenames = sorted(os.listdir(images_path))
        for ind in xrange(len(filenames)):
            arr = np.empty([1, timesteps] + list(input_shape))
            for step in xrange(timesteps):
                step_index = ind - step
                if 0 <= step_index <= ind:
                  src = os.path.join(images_path, filenames[step_index])
                  arr[0, timesteps - step - 1, :, :, :] = load_test_image(src)
            y_pred = (predictor(arr) * (3.14 / 180))
            filename = filenames[ind].split('.')[0]
            f.write('%s,%s\n' % (filename, y_pred))


def generate_video(submission_path,
                   images_path,
                   video_path,
                   temp_dir):

    assert video_path.endswith('.mp4'), 'h264 pls'
    safe_makedirs(os.path.dirname(video_path))

    filename_angles = []
    with open(submission_path) as f:
        for line in f:
            if "frame" not in line:
                ts, angle = line.strip().split(',')
                filename_angles.append((ts, angle))

    progress_bar = IncrementalBar(
        'Generating overlay',
        max=len(filename_angles),
        suffix='%(percent).1f%% - %(eta)ds')

    for filename, angle in filename_angles:
        img_path = os.path.join(images_path, filename + '.jpg')
        cv_image = overlay_angle(img_path, float(angle))
        cv2.imwrite(os.path.join(temp_dir, filename + '.png'), cv_image)
        progress_bar.next()

    print '\nGenerating mpg video'
    _, mpg_path = tempfile.mkstemp()
    subprocess.check_call([
        'mencoder',
        'mf://%s/*.png' % temp_dir,
        '-mf',
        'type=png:fps=20',
        '-o', mpg_path,
        '-speed', '1',
        '-ofps', '20',
        '-ovc', 'lavc',
        '-lavcopts', 'vcodec=mpeg2video:vbitrate=2500',
        '-oac', 'copy',
        '-of', 'mpeg'
    ], stdout=FNULL, stderr=subprocess.STDOUT)

    print 'Converting mpg video to mp4'
    try:
        subprocess.check_call([
            'ffmpeg',
            '-i', mpg_path,
            video_path
        ], stdout=FNULL, stderr=subprocess.STDOUT)
    finally:
        os.remove(mpg_path)

    print 'Wrote final overlay video to', video_path


def overlay_angle(img_path, angle):
    center=(320, 400)
    radius=50
    cv_image = cv2.imread(img_path)
    cv2.circle(cv_image, center, radius, (255, 255, 255), thickness=4, lineType=8)
    x, y = point_on_circle(center, radius, -angle)
    cv2.circle(cv_image, (x,y), 6, (255, 0, 0), thickness=6, lineType=8)
    cv2.putText(
        cv_image,
        'angle: %.5f' % get_degrees(angle),
        (50, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255))

    return cv_image


def get_degrees(radians):
    return (radians * 180.0) / 3.14


def point_on_circle(center, radius, angle):
    """ Finding the x,y coordinates on circle, based on given angle
    """
    # center of circle, angle in degree and radius of circle
    shift_angle = -3.14 / 2
    x = center[0] + (radius * cos(shift_angle + angle))
    y = center[1] + (radius * sin(shift_angle + angle))

    return int(x), int(y)


def smoothing_predictor(predictor,
                        smoothing=True,
                        smoothing_steps=3,
                        interpolation_weight=0.5,
                        max_abs_delta=None):
    if smoothing:
        assert smoothing_steps >= 2
        X = np.linspace(0, 1, smoothing_steps)
        x1 = 1 + X[1] - X[0]
        prev = deque()

    def predict_fn(x):
        p = predictor(x)

        if not smoothing:
            return p

        if len(prev) == smoothing_steps:
            m, b = np.polyfit(X, prev, 1)
            if abs(m) > 0.01:
                m *= 1.3
            p_interp = b + m * x1
            p_weighted = ((1 - interpolation_weight) * p
                          + interpolation_weight * p_interp)
            prev.popleft()
        else:
            p_weighted = p

        prev.append(p)

        if max_abs_delta is not None and len(prev) >= 2:
            p_last = prev[-2]
            delta = np.clip(
                p_weighted - p_last,
                -max_abs_delta,
                max_abs_delta)

            p_final = p_last + delta
        else:
            p_final = p_weighted

        return p_final

    return predict_fn


def load_test_image(src):
    cv_image = cv2.imread(src)
    cv_image = cv2.resize(cv_image, (320, 240))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
    cv_image = cv_image[120:240, :, :]
    cv_image[:,:,0] = cv2.equalizeHist(cv_image[:,:,0])
    cv_image = ((cv_image-(255.0/2))/255.0)
    return cv_image


def safe_makedirs(path):
    try: os.makedirs(path)
    except: pass


if __name__ == '__main__':
    main()
