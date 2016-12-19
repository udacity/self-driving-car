"""
This is the worker process which reads from the task queue, trains the
model, validates it, and writes the results to the db.
"""
import cProfile
import logging
import os
import signal
import StringIO
import pstats
import time

from keras.backend import binary_crossentropy
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf

from callbacks import SnapshotCallback
from models import (
    load_from_config,
    CategoricalModel, EnsembleModel, LstmModel, RegressionModel,
    TransferLstmModel)
from datasets import load_dataset

logger = logging.getLogger(__name__)

PROFILING = False


def profiling_sigint_handler(signal, frame):
    pr.disable()
    s = StringIO.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(.2)
    print s.getvalue()

    print '------------------------'
    if raw_input('continue? (y/n) ') != 'y':
        exit(0)

if PROFILING:
    pr = cProfile.Profile()
    pr.enable()
    signal.signal(signal.SIGINT, profiling_sigint_handler)


def handle_task(task,
                datasets_dir='/datasets',
                models_path='/models'):
    """
    Runs a tensorflow task.
    """
    model_config = task['model_config']
    model_type = model_config['type']
    logger.info('loading model with config %s', task['model_config'])
    model = load_from_config(task['model_config'])
    dataset_path = os.path.join(datasets_dir, task['dataset_path'])
    dataset = load_dataset(dataset_path)
    baseline_mse = dataset.get_baseline_mse()

    snapshot_dir = os.path.join(
        models_path, 'snapshots', model_type, task['task_id'])
    snapshot = SnapshotCallback(
        model,
        snapshot_dir=snapshot_dir,
        score_metric=task.get('score_metric', 'val_rmse'))

    earlystop = EarlyStopping(
        monitor=task.get('score_metric', 'val_rmse'),
        patience=12,
        mode='min')

    callbacks = [snapshot, earlystop]

    logger.info('Baseline mse = %.4f  rmse = %.4f' % (
        baseline_mse, np.sqrt(baseline_mse)))
    model.fit(
        dataset,
        task['training_args'],
        final=task.get('final', False),
        callbacks=callbacks)

    output_model_path = os.path.join(
        models_path, 'output', '%s.h5' % task['task_id'])
    output_config = model.save(output_model_path)
    logger.info('Maximum snapshot had score %s=%.6f, saved to %s',
                snapshot.score_metric,
                snapshot.max_score,
                snapshot.max_path)
    logger.info('Minimum snapshot had score %s=%.6f, saved to %s',
                snapshot.score_metric,
                snapshot.min_score,
                snapshot.min_path)
    logger.info('Wrote final model to %s', output_model_path)

    # assume evaluation is mse
    evaluation = model.evaluate(dataset)
    training_mse = evaluation[0]

    improvement = -(training_mse - baseline_mse) / baseline_mse
    logger.info('Evaluation: %s', evaluation)
    logger.info('Baseline MSE %.5f, training MSE %.5f, improvement %.2f%%',
                baseline_mse, training_mse, improvement * 100)
    logger.info('output config: %s' % output_config)

    if model.output_dim() == 1:
        example_ranges = 10
        range_size = 20
        testing_size = dataset.get_testing_size()
        for _ in xrange(example_ranges):
            # print out some sample prediction/label pairs
            skip_to = int(np.random.random() * (testing_size - range_size))
            example_images, example_labels = (dataset
                .sequential_generator(range_size)
                .skip(skip_to)
                .next())

            predictions = model.predict_on_batch(example_images)
            for pred, label in zip(predictions, example_labels):
                logger.info('p=%.5f  l=%.5f', pred, label)


def main():
    logging.basicConfig(level=logging.INFO)
    task_id = str(int(time.time()))
    tmp_model_path = os.path.join('/tmp', '%s.h5' % task_id)

    if True:
        task = {
            'task_id': task_id,
            'score_metric': 'val_rmse',
            'dataset_path': 'showdown_full',
            'final': True,
            'model_config': TransferLstmModel.create_cnn(
                tmp_model_path,
                transform_model_config={
                    'model_uri': '/models/snapshots/regression/1480182349/31.h5',
                    'scale': 16,
                    'type': 'regression'
                },
                timesteps=50,
                W_l2=0.001,
                scale=16.,
                input_shape=(120, 320, 3)),
            'training_args': {
                'batch_size': 32,
                'epochs': 100,
            },
        }

    if False:
        task = {
            'task_id': task_id,
            'score_metric': 'loss',
            'dataset_path': 'shinale_full',
            'final': False,
            'model_config': RegressionModel.create_resnet_inception_v2(
                tmp_model_path,
                learning_rate=0.001,
                input_shape=(120, 320, 3)),
            'training_args': {
                'batch_size': 16,
                'epochs': 100,
                'pctl_sampling': 'uniform',
                'pctl_thresholds': showdown_percentiles(),
            },
        }

    if False:
        task = {
            'task_id': task_id,
            'score_metric': 'loss',
            'dataset_path': 'showdown_full',
            'final': True,
            'model_config': {
                'model_uri': '/models/output/1480004259.h5',
                'scale': 16,
                'type': 'regression'
            },
            'training_args': {
                'batch_size': 32,
                'epochs': 40,
            },
        }


    if False:
        # sharp left vs center vs sharp right
        task = {
            'task_id': task_id,
            'dataset_path': 'finale_full',
            'score_metric': 'val_categorical_accuracy',
            'model_config': CategoricalModel.create(
                tmp_model_path,
                use_adadelta=True,
                W_l2=0.001,
                thresholds=[-0.061, 0.061]
            ),
            'training_args': {
                'batch_size': 32,
                'epochs': 30,
                'pctl_sampling': 'uniform',
            },
        }


    if False:
        # half degree model
        task = {
            'task_id': task_id,
            'dataset_path': 'finale_center',
            'model_config': CategoricalModel.create(
                tmp_model_path,
                use_adadelta=True,
                learning_rate=0.001,
                thresholds=np.linspace(-0.061, 0.061, 14)[1:-1],
                input_shape=(120, 320, 3)),
            'training_args': {
                'pctl_sampling': 'uniform',
                'batch_size': 32,
                'epochs': 20,
            },
        }

    if False:
        input_model_config = {
            'model_uri': 's3://sdc-matt/simple/1477715388/model.h5',
            'type': 'simple',
            'cat_classes': 5
        }

        ensemble_model_config = EnsembleModel.create(
            tmp_model_path,
            input_model_config,
            timesteps=3,
            timestep_noise=0.1,
            timestep_dropout=0.5)

        task = {
            'task_id': task_id,
            'dataset_path': 'final_training',
            'model_config': ensemble_model_config,
            'training_args': {
                'batch_size': 64,
                'epochs': 3
            },
        }

    if False:
        lstm_model_config = LstmModel.create(
            tmp_model_path,
            (10, 120, 320, 3),
            timesteps=10,
            W_l2=0.0001,
            scale=60.0)

        task = {
            'task_id': task_id,
            'dataset_path': 'showdown_full',
            'final': True,
            'model_config': lstm_model_config,
            'training_args': {
                'pctl_sampling': 'uniform',
                'batch_size': 32,
                'epochs': 10,
            },
        }

    handle_task(task)


def showdown_percentiles():
    return np.array([
        -9.19788539e-01,  -6.07374609e-01,  -5.13126791e-01,
        -4.59021598e-01,  -4.13643032e-01,  -3.63028497e-01,
        -3.17649931e-01,  -2.91469991e-01,  -2.67035365e-01,
        -2.46091425e-01,  -2.28638127e-01,  -2.09439516e-01,
        -1.88495561e-01,  -1.67551607e-01,  -1.50098309e-01,
        -1.34390354e-01,  -1.22173049e-01,  -1.13446400e-01,
        -1.04719758e-01,  -9.77384374e-02,  -9.25024524e-02,
        -8.55211318e-02,  -7.85398185e-02,  -7.15584978e-02,
        -6.63225129e-02,  -5.93411960e-02,  -5.41052073e-02,
        -5.06145470e-02,  -4.53785621e-02,  -4.01425734e-02,
        -3.66519131e-02,  -3.31612565e-02,  -2.79252678e-02,
        -2.61799395e-02,  -2.26892810e-02,  -2.09439509e-02,
        -1.91986226e-02,  -1.74532924e-02,  -1.57079641e-02,
        -1.39626339e-02,  -1.22173047e-02,  -1.04719754e-02,
        -8.72664619e-03,  -6.98131695e-03,  -5.23598772e-03,
        -3.49065848e-03,  -3.49065848e-03,  -1.74532924e-03,
        -1.74532924e-03,   0.00000000e+00,   1.74532924e-03,
         1.74532924e-03,   3.49065848e-03,   3.49065848e-03,
         5.23598772e-03,   6.98131695e-03,   8.72664619e-03,
         1.04719754e-02,   1.22173047e-02,   1.39626339e-02,
         1.57079641e-02,   1.74532924e-02,   1.91986226e-02,
         2.09439509e-02,   2.26892810e-02,   2.61799395e-02,
         2.79252678e-02,   3.31612565e-02,   3.66519131e-02,
         4.01425734e-02,   4.53785621e-02,   5.06145470e-02,
         5.41052073e-02,   5.93411960e-02,   6.63225129e-02,
         7.15584978e-02,   7.85398185e-02,   8.55211318e-02,
         9.25024524e-02,   9.77384374e-02,   1.04719758e-01,
         1.13446400e-01,   1.22173049e-01,   1.34390354e-01,
         1.50098309e-01,   1.67551607e-01,   1.88495561e-01,
         2.09439516e-01,   2.28638127e-01,   2.46091425e-01,
         2.67035365e-01,   2.91469991e-01,   3.17649931e-01,
         3.63028497e-01,   4.13643032e-01,   4.59021598e-01,
         5.13126791e-01,   6.07374609e-01,   9.19788539e-01,
         2.05076194e+00])


if __name__ == '__main__':
    main()
