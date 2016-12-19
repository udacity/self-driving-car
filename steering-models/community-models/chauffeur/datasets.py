"""
Loading/saving datasets.
"""
import logging
import multiprocessing
import os
import pickle
import shutil
import subprocess
import threading
import traceback

import cv2
import pandas as pd

from keras.utils.np_utils import to_categorical
import numpy as np
from progress.bar import IncrementalBar
import requests
from scipy.stats.mstats import mquantiles

from models import get_output_dim
from util import download_dir, parse_s3_uri, upload_dir

logger = logging.getLogger(__name__)


class Dataset(object):
    """
    Used to create (image, label) tuple generators of an image dataset.
    """
    def __init__(self,
                 images_base_path,
                 labels,
                 training_indexes,
                 testing_indexes,
                 validation_indexes,
                 image_file_fmt='%d.png.npy'):
        """
        @param images_base_path - path to image files
        @param labels - 2d array of all label data
        @param training_indexes - 1d array of training indexes
        @param testing_indexes - 1d array of testing indexes
        @param validation_indexes - 1d array of validation indexes
        @param image_file_fmt - format string for image file names
        """
        self.labels = labels
        self.training_indexes = training_indexes
        self.testing_indexes = testing_indexes
        self.validation_indexes = validation_indexes
        self.images_base_path = images_base_path
        self.image_file_fmt = image_file_fmt

    def get_image_shape(self):
        """
        @return - image dimensions shape
        """
        return self.load_image(self.training_indexes[0]).shape

    def get_training_size(self):
        """
        @return - number of training samples
        """
        return len(self.training_indexes)

    def get_testing_size(self):
        """
        @return - number of testing samples
        """
        return len(self.testing_indexes)

    def get_validation_size(self):
        """
        @return - number of validation samples
        """
        return len(self.validation_indexes)

    def get_training_labels(self):
        """
        @return - numpy array of labels for training set
        """
        return self.labels[self.training_indexes - 1]

    def get_testing_labels(self):
        """
        @return - numpy array of labels for testing set
        """
        return self.labels[self.testing_indexes - 1]

    def get_validation_labels(self):
        """
        @return - numpy array of labels for validation set
        """
        return self.labels[self.validation_indexes - 1]

    def final_generator(self, batch_size):
        """
        Generator over all non-validation samples.

        @param batch_size - images per batch
        @return - generator returning (images, labels) batches
        """
        indexes = np.concatenate([
            self.training_indexes,
            self.testing_indexes])

        return self.get_generator(batch_size, indexes, True)

    def training_generator(self, batch_size):
        """
        Generator over training samples.

        @param batch_size - images per batch
        @return - generator returning (images, labels) batches
        """
        return self.get_generator(batch_size, self.training_indexes, True)

    def testing_generator(self, batch_size):
        """
        Generator over testing samples.

        @param batch_size - images per batch
        @return - generator returning (images, labels) batches
        """
        return self.get_generator(batch_size, self.testing_indexes, True)

    def validation_generator(self, batch_size):
        """
        Generator over validation samples.

        @param batch_size - images per batch
        @return - generator returning (images, labels) batches
        """
        return self.get_generator(batch_size, self.validation_indexes, True)

    def sequential_generator(self, batch_size):
        """
        Generator which iterates over each image in sequential order.

        @param batch_size - images per batch
        @return - generator returning (images, labels) batches
        """
        max_index = np.max([
            self.training_indexes.max(),
            self.testing_indexes.max(),
            self.validation_indexes.max()])

        # generate a sequential list of all indexes
        indexes = np.arange(1, max_index + 1)

        # don't shuffle indexes on each pass to maintain order
        return self.get_generator(batch_size, indexes, False)

    def get_generator(self, batch_size, indexes, shuffle_on_exhaust):
        """
        Helper to get an infinite image loading generator.

        @param batch_size - images per batch
        @param indexes - 1d array of indexes to include in dataset
        @param shuffle_on_exhaust - should shuffle data on each full pass
        @return - generator returning (images, labels) batches
        """
        return InfiniteImageLoadingGenerator(
            batch_size,
            indexes,
            self.labels,
            self.images_base_path,
            self.image_file_fmt,
            shuffle_on_exhaust=shuffle_on_exhaust)

    def get_baseline_mse(self):
        """
        Get the baseline MSE of a dataset using a dummy predictor.

        @return - mean squared error of dummy predictor on testing set
        """
        dummy_predictor = self.get_training_labels().mean()
        mse = ((self.get_testing_labels() - dummy_predictor) ** 2).mean()
        return mse


class InfiniteImageLoadingGenerator(object):
    """
    Iterable object which loads the next batch of (image, label) tuples
    in the data set.
    """
    def __init__(self,
                 batch_size,
                 indexes,
                 labels,
                 images_base_path,
                 image_file_fmt,
                 shuffle_on_exhaust,
                 cat_thresholds=None,
                 pctl_sampling='none',
                 pctl_thresholds=None,
                 timesteps=None,
                 timestep_delta=1,
                 precomputed=None,
                 scale=1.0,
                 concat_original=False):
        """
        @param batch_size - number of images to generate per batch
        @param indexes - array (N,) of image index IDs
        @param labels - array (M,) of all labels
        @param images_base_path - local path to image directory
        @param image_file_fmt - format string for image filenames
        @param shuffle_on_exhaust - should shuffle data on each full pass
        @param cat_thresholds - categorical label thresholds.
        @param pctl_sampling - type of percentile sampling.
        @param pctl_thresholds - override percentile thresholds.
        @param timesteps - appends this many previous labels to end of samples
        @param timestep_delta - number of frames to skip in each timestep
        @param precomputed - precomputed input data {index: data}
        @param scale - scaling factor to apply to labels
        """
        self.batch_size = batch_size
        self.indexes = indexes
        self.orig_labels = labels
        self.images_base_path = images_base_path
        self.image_file_fmt = image_file_fmt
        self.shuffle_on_exhaust = shuffle_on_exhaust
        self.cat_thresholds = cat_thresholds
        self.pctl_sampling = pctl_sampling
        self.pctl_thresholds = pctl_thresholds
        self.timesteps = timesteps
        self.timestep_delta = timestep_delta
        self.precomputed = precomputed
        self.scale = scale
        self.concat_original = concat_original

        self.image_shape = list(self.load_image(self.indexes[0]).shape)

        if pctl_sampling != 'none':
            these_labels = labels[indexes - 1]
            if cat_thresholds is not None:
                pctl_splits = (
                    [these_labels.min()] +
                    cat_thresholds +
                    [these_labels.max()])
            elif pctl_thresholds is not None:
                pctl_splits = (
                    [these_labels.min()] +
                    pctl_thresholds +
                    [these_labels.max()])
            else:
                pctl_splits = mquantiles(these_labels, np.arange(0.0, 1.01, 0.01))

            self.pctl_indexes = filter(len, [
                indexes[np.where((these_labels >= lb) & (these_labels < ub))[0]]
                for lb, ub in zip(pctl_splits[:-1], pctl_splits[1:])])

        # If cat_classes specified, map labels to discrete classes
        if cat_thresholds is not None:
            if len(cat_thresholds) == 1:
                self.labels = labels > cat_thresholds[0]
                self.label_shape = [1]
            else:
                binned_labels = np.digitize(labels, cat_thresholds)
                self.labels = to_categorical(binned_labels)
                self.label_shape = list(self.labels.shape[1:])
        else:
            self.labels = self.orig_labels * self.scale
            self.label_shape = [1]

        self.current_index = 0

    def get_batch_size(self):
        return self.batch_size

    def get_size(self):
        return len(self.indexes)

    def scale_labels(self, scale):
        return InfiniteImageLoadingGenerator(
            batch_size=self.batch_size,
            indexes=self.indexes,
            labels=self.orig_labels,
            images_base_path=self.images_base_path,
            image_file_fmt=self.image_file_fmt,
            shuffle_on_exhaust=self.shuffle_on_exhaust,
            cat_thresholds=self.cat_thresholds,
            pctl_sampling=self.pctl_sampling,
            pctl_thresholds=self.pctl_thresholds,
            timesteps=self.timesteps,
            timestep_delta=self.timestep_delta,
            precomputed=self.precomputed,
            scale=scale,
            concat_original=self.concat_original)

    def as_categorical(self, cat_thresholds):
        return InfiniteImageLoadingGenerator(
            batch_size=self.batch_size,
            indexes=self.indexes,
            labels=self.orig_labels,
            images_base_path=self.images_base_path,
            image_file_fmt=self.image_file_fmt,
            shuffle_on_exhaust=self.shuffle_on_exhaust,
            cat_thresholds=cat_thresholds,
            pctl_sampling=self.pctl_sampling,
            pctl_thresholds=self.pctl_thresholds,
            timesteps=self.timesteps,
            timestep_delta=self.timestep_delta,
            precomputed=self.precomputed,
            scale=self.scale,
            concat_original=self.concat_original)

    def with_percentile_sampling(self,
                                 pctl_sampling='uniform',
                                 pctl_thresholds=None):
        """
        Performs some sampling on the labels used in each batch. Types
        of sampling are:

          uniform - each batch will have equal representation of label pctl
          same - each batch will be from the same label pctl
          none - default, no sampling
        """
        return InfiniteImageLoadingGenerator(
            batch_size=self.batch_size,
            indexes=self.indexes,
            labels=self.orig_labels,
            images_base_path=self.images_base_path,
            image_file_fmt=self.image_file_fmt,
            shuffle_on_exhaust=self.shuffle_on_exhaust,
            cat_thresholds=self.cat_thresholds,
            pctl_sampling=pctl_sampling,
            pctl_thresholds=pctl_thresholds,
            timesteps=self.timesteps,
            timestep_delta=self.timestep_delta,
            precomputed=self.precomputed,
            scale=self.scale,
            concat_original=self.concat_original)

    def with_timesteps(self, timesteps=None, timestep_delta=1):
        """
        Add previous input timesteps.

        @param timesteps - number of previous labels to append to samples
        @return - image-loading iterator with prev image timesteps
        """
        return InfiniteImageLoadingGenerator(
            batch_size=self.batch_size,
            indexes=self.indexes,
            labels=self.orig_labels,
            images_base_path=self.images_base_path,
            image_file_fmt=self.image_file_fmt,
            shuffle_on_exhaust=self.shuffle_on_exhaust,
            cat_thresholds=self.cat_thresholds,
            pctl_sampling=self.pctl_sampling,
            pctl_thresholds=self.pctl_thresholds,
            timesteps=timesteps,
            timestep_delta=timestep_delta,
            precomputed=self.precomputed,
            scale=self.scale,
            concat_original=self.concat_original)

    def precompute_transform(self, transform_model):
        """
        Precompute an encoder transform on the whole dataset and load
        it into memory.

        @param transform_model- model supporting predict_on_batch
        @return - imag-loading iterator with precomputed transformed input
        """
        encoder = transform_model.as_encoder()
        batch_size = self.batch_size
        output_dim = get_output_dim(encoder)
        indexes = np.arange(1, len(self.labels) + 1)
        n_samples = len(indexes)
        n_batches = n_samples / batch_size
        n_batches += 1 if (n_batches * batch_size) != n_samples else 0
        progress_bar = IncrementalBar(
            'Precomputing',
            max=n_batches,
            suffix='%(percent).1f%% - %(eta)ds')
        precomputed = {}

        for batch_no in xrange(n_batches):
            start = batch_no * batch_size
            end = min((batch_no + 1) * batch_size, n_samples)
            batch_indexes = indexes[start:end]
            size = end - start

            batch = np.empty([size] + self.image_shape)
            for i, img_index in enumerate(batch_indexes):
                batch[i] = self.load_image(img_index)
            transformed = encoder.predict_on_batch(batch)

            for i, img_index in enumerate(batch_indexes):
                precomputed[img_index] = transformed[i]

            progress_bar.next()

        return InfiniteImageLoadingGenerator(
            batch_size=self.batch_size,
            indexes=self.indexes,
            labels=self.orig_labels,
            images_base_path=self.images_base_path,
            image_file_fmt=self.image_file_fmt,
            shuffle_on_exhaust=self.shuffle_on_exhaust,
            cat_thresholds=self.cat_thresholds,
            pctl_sampling=self.pctl_sampling,
            pctl_thresholds=self.pctl_thresholds,
            timesteps=self.timesteps,
            timestep_delta=self.timestep_delta,
            precomputed=precomputed,
            scale=self.scale,
            concat_original=self.concat_original)

    def with_precomputed(self, precomputed):
        return InfiniteImageLoadingGenerator(
            batch_size=self.batch_size,
            indexes=self.indexes,
            labels=self.orig_labels,
            images_base_path=self.images_base_path,
            image_file_fmt=self.image_file_fmt,
            shuffle_on_exhaust=self.shuffle_on_exhaust,
            cat_thresholds=self.cat_thresholds,
            pctl_sampling=self.pctl_sampling,
            pctl_thresholds=self.pctl_thresholds,
            timesteps=self.timesteps,
            timestep_delta=self.timestep_delta,
            precomputed=precomputed,
            scale=self.scale,
            concat_original=self.concat_original)

    def with_concat_original(self, concat_original):
        return InfiniteImageLoadingGenerator(
            batch_size=self.batch_size,
            indexes=self.indexes,
            labels=self.orig_labels,
            images_base_path=self.images_base_path,
            image_file_fmt=self.image_file_fmt,
            shuffle_on_exhaust=self.shuffle_on_exhaust,
            cat_thresholds=self.cat_thresholds,
            pctl_sampling=self.pctl_sampling,
            pctl_thresholds=self.pctl_thresholds,
            timesteps=self.timesteps,
            timestep_delta=self.timestep_delta,
            precomputed=self.precomputed,
            scale=self.scale,
            concat_original=concat_original)

    def __iter__(self):
        return self

    def load_image(self, index, original_only=False):
        """
        Load image at index.

        @param index - index of image
        @return - image data
        """
        if self.precomputed is not None and not original_only:
            return self.precomputed[index]
        else:
            return load_image(
                index, self.images_base_path, self.image_file_fmt)

    def skip(self, n):
        """
        Skip n indexes in the generator.
        """
        self.incr_index(n)
        return self

    def incr_index(self, n=1):
        """
        Increment the current index, wrapping around and possibly
        shuffling the dataset on EOF
        """
        if self.current_index + n == len(self.indexes):
            self.current_index = 0

            if self.shuffle_on_exhaust:
                # each full pass over data is a random permutation
                np.random.shuffle(self.indexes)
        else:
            self.current_index += n

        return self.current_index

    def next(self):
        labels = np.empty([self.batch_size] + self.label_shape)
        if self.timesteps is not None:
            timesteps_shape = (
                [self.batch_size, self.timesteps] + self.image_shape)
            samples = np.empty(timesteps_shape)
        else:
            samples = np.empty([self.batch_size] + self.image_shape)

        originals = []

        if self.pctl_sampling == 'none':
            next_indexes = [
                self.indexes[self.incr_index()]
                for _ in xrange(self.batch_size)]
        elif self.pctl_sampling == 'uniform':
            max_bins = max(self.batch_size, len(self.pctl_indexes))
            per_bin = max(self.batch_size / max_bins, 1)
            index_bins = np.random.choice(self.pctl_indexes, max_bins)
            next_indexes = []
            for index_bin in index_bins:
                remaining = self.batch_size - len(next_indexes)
                for_this_bin = min(remaining, per_bin)
                next_indexes.extend(np.random.choice(index_bin, for_this_bin))
        elif self.pctl_sampling == 'same':
            chosen_bin = np.random.choice(self.pctl_indexes, 1)[0]
            with_replacement = len(chosen_bin) < self.batch_size
            next_indexes = np.random.choice(
                chosen_bin, self.batch_size, with_replacement)
        else:
            raise NotImplementedError

        for i, next_image_index in enumerate(next_indexes):
            # image indexes are 1-indexed
            next_label_index = next_image_index - 1
            labels[i] = self.labels[next_label_index]

            if self.concat_original:
                originals.append(self.load_image(next_image_index, True))

            if self.timesteps is None:
                samples[i] = self.load_image(next_image_index)
            else:
                for step in xrange(self.timesteps):
                    delta = step * self.timestep_delta
                    step_index = max(1, next_image_index - delta)
                    img = self.load_image(step_index)
                    samples[i, self.timesteps - step - 1] = img

        if self.concat_original:
            originals_np = np.empty((self.batch_size, ) + originals[0].shape)
            for i, orig in enumerate(originals):
                originals_np[i] = orig
            return ([samples, originals_np], labels)
        else:
            return (samples, labels)

def load_image(index, images_base_path, image_file_fmt):
    """
    Load image from disk.

    @param index - image index
    @param images_base_path - base images path
    @param image_file_fmt - image file fmt string
    @return - 3d image numpy array
    """
    image_path = os.path.join(images_base_path, image_file_fmt % index)
    image = np.load(image_path)
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    return ((image-(255.0/2))/255.0)


def load_dataset(dataset_path):
    """
    Downloads and loads an image dataset.

    A dataset directory should have this structure:
      /
          training_indexes.npy
          testing_indexes.npy
          validation_indexes.npy
          labels.npy
          images/
              1.png.npy
              2.png.npy
              ...
              N.png.npy

    @param dataset_path - path to dataset directory
    """
    assert(os.path.exists(dataset_path),
           "dataset doesn't exist: " + dataset_path)

    # Load the dataset from the local directory
    labels = np.load(os.path.join(dataset_path, 'labels.npy'))

    training_indexes = (np
        .load(os.path.join(dataset_path, 'training_indexes.npy'))
        .astype(int))
    testing_indexes = (np
        .load(os.path.join(dataset_path, 'testing_indexes.npy'))
        .astype(int))
    validation_indexes = (np
        .load(os.path.join(dataset_path, 'validation_indexes.npy'))
        .astype(int))

    images_base_path = os.path.join(dataset_path, 'images')

    # hack png vs jpg; DWIM
    try:
        load_image(1, images_base_path, '%d.png.npy')
        image_file_fmt = '%d.png.npy'
    except:
        image_file_fmt = '%d.jpg.npy'

    return Dataset(
        images_base_path=images_base_path,
        labels=labels,
        training_indexes=training_indexes,
        testing_indexes=testing_indexes,
        validation_indexes=validation_indexes,
        image_file_fmt=image_file_fmt)


def prepare_dataset(
        archive_url,
        local_output_path,
        output_s3_uri,
        training_percent=0.7,
        testing_percent=0.2,
        validation_percent=0.1,
        cache_dir='/tmp'):
    """
    Prepare dataset from nishanth's preprocessed format.

    @param archive_url - url to download dataset archive from
    @param local_output_path - where to write prepared dataset locally
    @param output_s3_uri - where to write prepared dataset in s3
    @param training_percent - percent of samples used in training set
    @param testing_percent - percent of samples used in testing set
    @param validation_percent - percent of samples used in validation set
    @param cache_dir - where to store intermediate archives
    """
    filename = archive_url.split('/')[-1]
    archive_path = os.path.join(cache_dir, filename)
    local_raw_path = os.path.join(cache_dir, filename.split('.')[0])

    # download/decompress archive if necessary
    if not os.path.exists(local_raw_path):
        logger.info('Downloading dataset archive from %s', archive_url)

        # download
        r = requests.get(archive_url, stream=True)
        with open(archive_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

        # decompress archive
        logger.info('Decompressing dataset archive %s to %s',
                     filename, local_raw_path)

        try: os.makedirs(local_raw_path)
        except: pass

        subprocess.call(['tar', 'xf', archive_path, '-C', local_raw_path])

    dir_list = list(os.listdir(local_raw_path))
    assert len(dir_list) == 2
    assert 'labels' in dir_list

    images_dirname, = [f for f in dir_list if f != 'labels']
    base_images_path = os.path.join(local_raw_path, images_dirname)
    logger.info('Using %s as base images directory', base_images_path)

    with open(os.path.join(local_raw_path, 'labels')) as labels_f:
        labels = np.array([float(line.strip()) for line in labels_f])

    n_samples = len(labels)
    n_training = int(training_percent * n_samples)
    n_testing = int(testing_percent * n_samples)
    n_validation = n_samples - n_training - n_testing

    logger.info('%d total samples in the dataset', n_samples)
    logger.info('%d samples in training set', n_training)
    logger.info('%d samples in testing set', n_testing)
    logger.info('%d samples in validation set', n_validation)

    indexes = np.arange(1, n_samples + 1)
    np.random.shuffle(indexes)

    training_indexes = indexes[:n_training]
    testing_indexes = indexes[n_training:(n_training + n_testing)]
    validation_indexes = indexes[-n_validation:]

    shutil.rmtree(local_output_path, ignore_errors=True)
    os.makedirs(local_output_path)

    # create the properly-formatted dataset directory
    np.save(os.path.join(local_output_path, 'labels.npy'), labels)
    np.save(
        os.path.join(local_output_path, 'training_indexes.npy'),
        training_indexes)
    np.save(
        os.path.join(local_output_path, 'testing_indexes.npy'),
        testing_indexes)
    np.save(
        os.path.join(local_output_path, 'validation_indexes.npy'),
        validation_indexes)
    shutil.copytree(
        base_images_path,
        os.path.join(local_output_path, 'images'))

    # upload dataset directory to s3
    upload_dir(local_output_path, output_s3_uri)

def prepare_local_dataset(
        local_raw_path,
        local_output_path,
        training_percent=0.7,
        testing_percent=0.2,
        validation_percent=0.1):
    """
    Prepare dataset from nishanth's preprocessed format.

    @param local_raw_path - input path to a dataset directory containing images and a labels file
    @param local_output_path - where to write prepared dataset locally
    @param training_percent - percent of samples used in training set
    @param testing_percent - percent of samples used in testing set
    @param validation_percent - percent of samples used in validation set
    """
    dir_list = list(os.listdir(local_raw_path))
    assert len(dir_list) == 2
    assert 'labels' in dir_list

    images_dirname, = [f for f in dir_list if f != 'labels']
    base_images_path = os.path.join(local_raw_path, images_dirname)
    logger.info('Using %s as base images directory', base_images_path)

    with open(os.path.join(local_raw_path, 'labels')) as labels_f:
        labels = np.array([float(line.strip()) for line in labels_f])

    n_samples = len(labels)
    n_training = int(training_percent * n_samples)
    n_testing = int(testing_percent * n_samples)
    n_validation = n_samples - n_training - n_testing

    logger.info('%d total samples in the dataset', n_samples)
    logger.info('%d samples in training set', n_training)
    logger.info('%d samples in testing set', n_testing)
    logger.info('%d samples in validation set', n_validation)

    indexes = np.arange(1, n_samples + 1)
    np.random.shuffle(indexes)

    training_indexes = indexes[:n_training]
    testing_indexes = indexes[n_training:(n_training + n_testing)]
    validation_indexes = indexes[-n_validation:]

    shutil.rmtree(local_output_path, ignore_errors=True)
    os.makedirs(local_output_path)

    # create the properly-formatted dataset directory
    np.save(os.path.join(local_output_path, 'labels.npy'), labels)
    np.save(
        os.path.join(local_output_path, 'training_indexes.npy'),
        training_indexes)
    np.save(
        os.path.join(local_output_path, 'testing_indexes.npy'),
        testing_indexes)
    np.save(
        os.path.join(local_output_path, 'validation_indexes.npy'),
        validation_indexes)
    subprocess.call([
        'ln', '-s',
        os.path.join(local_raw_path, 'images'),
        os.path.join(local_output_path, 'images')])

def prepare_final_dataset(
        local_raw_path,
        local_output_path,
        training_percent=0.7,
        testing_percent=0.2,
        validation_percent=0.1):
    train_path = os.path.join(local_raw_path, 'Train')

    # ensure images path exists
    images_path = os.path.join(local_output_path, 'images')
    logger.info('Using %s as base images directory', images_path)
    try: os.makedirs(images_path)
    except: pass

    part_dfs = []
    for part_no in os.listdir(train_path):
        part_path = os.path.join(train_path, str(part_no))
        sensor_csv_path = os.path.join(part_path, 'interpolated.csv')
        sensor_df = pd.DataFrame.from_csv(sensor_csv_path)
        center_df = sensor_df[sensor_df['frame_id'] == 'center_camera'].copy()
        center_df['filename'] = (
            (part_path + '/') + center_df.filename.astype(str))

        part_dfs.append(center_df[['timestamp', 'filename', 'angle']])

    # concat all the path directory csvs
    master_df = pd.concat(part_dfs).sort_values('timestamp')

    n_original_samples = len(master_df)
    n_samples = len(master_df) * 2
    n_training = int(training_percent * n_samples)
    n_testing = int(testing_percent * n_samples)
    n_validation = n_samples - n_training - n_testing

    logger.info('%d total samples in the dataset', n_samples)
    logger.info('%d samples in training set', n_training)
    logger.info('%d samples in testing set', n_testing)
    logger.info('%d samples in validation set', n_validation)

    labels = np.empty(n_samples)
    tasks = []
    for image_index, (_, row) in enumerate(master_df.iterrows()):
        labels[image_index] = row.angle
        labels[image_index + n_original_samples] = -row.angle
        tasks.append(
            (row.filename, images_path, image_index + 1, image_index + n_original_samples + 1))

    indexes = np.arange(1, n_samples + 1)
    np.random.shuffle(indexes)

    training_indexes = indexes[:n_training]
    testing_indexes = indexes[n_training:(n_training + n_testing)]
    validation_indexes = indexes[-n_validation:]

    np.save(os.path.join(local_output_path, 'labels.npy'), labels)
    np.save(
        os.path.join(local_output_path, 'training_indexes.npy'),
        training_indexes)
    np.save(
        os.path.join(local_output_path, 'testing_indexes.npy'),
        testing_indexes)
    np.save(
        os.path.join(local_output_path, 'validation_indexes.npy'),
        validation_indexes)

    pool = multiprocessing.Pool(4)
    pool.map(process_final_image, tasks)


def process_final_image(args):
    src_path, dest_dir, image_index, flipped_image_index = args
    normal_path = os.path.join(dest_dir, '%d.png.npy' % image_index)
    flipped_path = os.path.join(dest_dir, '%d.png.npy' % flipped_image_index)

    cv_image = cv2.imread(src_path)
    cv_image = cv2.resize(cv_image, (320, 240))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
    cv_image = cv_image[120:240, :, :]

    np.save(normal_path, cv_image)

    # flip the image over the y axis to equalize left/right turns
    cv_image = cv_image[:, ::-1, :]
    np.save(flipped_path, cv_image)


def prepare_thresholded_dataset(src_path,
                                local_output_path,
                                min_threshold=None,
                                max_threshold=None,
                                sample_region=None,
                                training_percent=0.7,
                                testing_percent=0.2,
                                validation_percent=0.1):
    try: os.makedirs(local_output_path)
    except: pass

    subprocess.call([
        'ln', '-s',
        os.path.join(src_path, 'images'),
        os.path.join(local_output_path, 'images')])

    labels = np.load(os.path.join(src_path, 'labels.npy'))
    logger.info('%d samples in original dataset', len(labels))

    cond = (labels > -99999999)  # hack;
    if min_threshold is not None:
        cond &= (labels >= min_threshold)
    if max_threshold is not None:
        cond &= (labels <= max_threshold)

    indexes = np.arange(1, len(labels) + 1)
    indexes = indexes[np.where(cond)[0]]

    logger.info('%d samples after applying thresholds', len(indexes))

    if sample_region is not None:
        new_labels = labels[np.where(cond)[0]]
        lb, ub = sample_region
        cond = (new_labels >= lb) & (new_labels <= ub)
        in_region = indexes[np.where(cond)[0]]
        out_region = indexes[np.where(~cond)[0]]
        target_size = len(out_region)
        in_region = np.random.choice(in_region, target_size, replace=False)
        indexes = np.concatenate((in_region, out_region))

        logger.info('%d samples in region, %d outside',
                    len(in_region), len(out_region))

    logger.info('Final dataset size %d', len(indexes))

    np.random.shuffle(indexes)
    n_samples = len(indexes)
    n_training = int(training_percent * n_samples)
    n_testing = int(testing_percent * n_samples)
    n_validation = n_samples - n_training - n_testing

    training_indexes = indexes[:n_training]
    testing_indexes = indexes[n_training:(n_training + n_testing)]
    validation_indexes = indexes[-n_validation:]

    # create the properly-formatted dataset directory
    np.save(os.path.join(local_output_path, 'labels.npy'), labels)
    np.save(
        os.path.join(local_output_path, 'training_indexes.npy'),
        training_indexes)
    np.save(
        os.path.join(local_output_path, 'testing_indexes.npy'),
        testing_indexes)
    np.save(
        os.path.join(local_output_path, 'validation_indexes.npy'),
        validation_indexes)


def combine_datasets(dataset_path1,
                     dataset_path2,
                     output_path,
                     percent1,
                     percent2,
                     min_sequence=1000,
                     training_percent=0.7,
                     testing_percent=0.2,
                     validation_percent=0.1):
    """
    """
    try: shutil.rmtree(output_path)
    except: pass

    images_path = os.path.join(output_path, 'images')
    try: os.makedirs(images_path)
    except: pass

    labels1 = np.load(os.path.join(dataset_path1, 'labels.npy'))
    labels2 = np.load(os.path.join(dataset_path2, 'labels.npy'))

    indexes1 = np.random.choice(
        np.arange(1, len(labels1) + 1), int(percent1 * len(labels1)))

    n_indexes2 = int(percent2 * len(labels2))
    indexes2 = []
    exclude2 = set()
    while len(indexes2) < n_indexes2:
        index = int(np.random.random() * (len(labels2) - min_sequence))
        indexes2.extend(np.arange(index, index + min_sequence))
        exclude2.update(np.arange(index, index + 100))

    out_img_index = 1
    new_labels = np.empty(len(indexes1) + len(indexes2))
    new_exclude = set()

    for index in indexes1:
        src_path = os.path.join(dataset_path1, 'images', '%s.png.npy' % index)
        dest_path = os.path.join(images_path, '%s.png.npy' % out_img_index)
        subprocess.check_call(['ln', '-s', src_path, dest_path])
        new_labels[out_img_index - 1] = labels1[index - 1]
        out_img_index += 1

    for index in indexes2:
        src_path = os.path.join(dataset_path2, 'images', '%s.png.npy' % index)
        dest_path = os.path.join(images_path, '%s.png.npy' % out_img_index)
        subprocess.check_call(['ln', '-s', src_path, dest_path])
        new_labels[out_img_index - 1] = labels2[index - 1]

        if index in exclude2:
            new_exclude.add(out_img_index)

        out_img_index += 1

    # exclude the first 100 images from training from each random sequence
    new_indexes = np.array([
        x for x in np.arange(1, len(new_labels) + 1)
        if x not in new_exclude])
    np.random.shuffle(new_indexes)
    n_samples = len(new_indexes)
    n_training = int(training_percent * n_samples)
    n_testing = int(testing_percent * n_samples)
    n_validation = n_samples - n_training - n_testing

    training_indexes = new_indexes[:n_training]
    testing_indexes = new_indexes[n_training:(n_training + n_testing)]
    validation_indexes = new_indexes[-n_validation:]

    # create the properly-formatted dataset directory
    np.save(os.path.join(output_path, 'labels.npy'), new_labels)
    np.save(
        os.path.join(output_path, 'training_indexes.npy'),
        training_indexes)
    np.save(
        os.path.join(output_path, 'testing_indexes.npy'),
        testing_indexes)
    np.save(
        os.path.join(output_path, 'validation_indexes.npy'),
        validation_indexes)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    combine_datasets(
        '/datasets/showdown_full',
        '/datasets/finale_full',
        '/datasets/shinmicro_full',
        1.0,
        0.1)
