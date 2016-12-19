"""
Creating tensorflow models.
"""
from collections import deque
import logging, os, tempfile

from keras import backend as K
from keras import metrics
from keras.engine.topology import Merge
from keras.layers import (
    Activation, BatchNormalization, Dense, Dropout, Flatten,
    Input, SpatialDropout2D, merge)
from keras.layers.convolutional import (
    AveragePooling2D, Convolution2D, MaxPooling2D)
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.models import load_model as keras_load_model
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
import numpy as np
from scipy.stats.mstats import mquantiles
import tensorflow as tf

from util import download_file, parse_s3_uri, upload_file

logger = logging.getLogger(__name__)


class BaseModel(object):
    """
    Abstraction over model which allows fit/evaluate/predict
    given input data.
    """
    def save(self, model_path):
        """
        Save model and return the new model config.

        @param model_path - path to write model to
        @return - model config
        """
        raise NotImplemented

    def fit(self, dataset, training_args, callbacks=None, final=False):
        """
        Fit the model with given dataset/args.

        @param dataset - a Dataset
        @param training_args - dict of training args
        @param callbacks - optional list of callbacks to use
        @param final - if True, train on all available data
        """
        raise NotImplemented

    def evaluate(self, dataset):
        """
        Evaluate the model given dataset.

        @param dataset - a Dataset
        @return - evaluation results (type specific to each model)
        """
        raise NotImplemented

    def predict_on_batch(self, batch):
        """
        Predict given batch of input samples.

        @param batch - batch of input samples
        @return - output predictions
        """
        raise NotImplemented

    def as_encoder(self):
        """
        Transform this model into an encoder model.

        @return - tensorflow model
        """
        raise NotImplemented

    def output_dim(self):
        """
        @return - output dimension
        """
        raise NotImplemented

    def predict_all(self, dataset, test_only=False):
        """
        todo support multi-class output

        @returns - (y_pred, y_true)
        """
        batch_size = 32
        if test_only:
            generator = dataset.testing_generator(batch_size)
            n_total = dataset.testing_size()
        else:
            generator = dataset.sequential_generator(batch_size)
            n_total = len(dataset.labels)

        output_dim = self.output_dim()
        n_batches = n_total / batch_size
        y_pred = np.empty(n_batches * batch_size)
        y_true = np.empty(n_batches * batch_size)

        for i in xrange(n_batches):
            x, y = generator.next()
            start, end = (i * batch_size, (i + 1) * batch_size)
            y_pred[start:end] = self.predict_on_batch(x).ravel()
            y_true[start:end] = y.ravel()

        return y_pred, y_true


class CategoricalModel(BaseModel):
    TYPE = 'categorical'

    def __init__(self, model_config):
        self.model = load_model_from_uri(model_config['model_uri'])
        self.thresholds = model_config.get('thresholds')

        if 'thresholds' not in model_config:
            logger.warning('Using old-style config, any use other than '
                           'as an encoder will fail')

    def fit(self, dataset, training_args, callbacks=None, final=False):
        batch_size = training_args.get('batch_size', 100)
        epochs = training_args.get('epochs', 5)

        self.model.summary()

        training_generator = (
            dataset.final_generator(batch_size) if final
            else dataset.training_generator(batch_size))

        training_generator = training_generator.as_categorical(self.thresholds)

        if 'pctl_sampling' in training_args:
            pctl_sampling = training_args['pctl_sampling']
            pctl_thresholds = training_args.get('pctl_thresholds')
            training_generator = (training_generator
                .with_percentile_sampling(pctl_sampling, pctl_thresholds))

        validation_generator = (dataset
            .validation_generator(batch_size)
            .as_categorical(self.thresholds))

        if 'pctl_sampling' in training_args:
            pctl_sampling = training_args['pctl_sampling']
            pctl_thresholds = training_args.get('pctl_thresholds')
            training_generator = (training_generator
                .with_percentile_sampling(pctl_sampling, pctl_thresholds))

        validation_size = training_args.get(
            'validation_size', validation_generator.get_size())
        epoch_size = training_args.get(
            'epoch_size', training_generator.get_size())

        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=(callbacks or []))

    def evaluate(self, dataset):
        n_batches = 32 * (dataset.get_testing_size() / 32)
        testing_generator = (dataset
            .testing_generator(32)
            .as_categorical(self.thresholds))

        return self.model.evaluate_generator(
            testing_generator, n_batches)

    def predict_on_batch(self, batch):
        return self.model.predict_on_batch(batch)

    def as_encoder(self):
        # remove the last output layers to retain the feature maps
        deep_copy = deep_copy_model(self.model)
        for _ in xrange(4):
            deep_copy.pop()
        return deep_copy_model_weights(deep_copy)

    def output_dim(self):
        return get_output_dim(self.model)

    def save(self, model_path):
        save_model(self.model, model_path)

        return {
            'type': CategoricalModel.TYPE,
            'model_uri': model_path,
            'thresholds': self.thresholds,
        }

    @classmethod
    def create(cls,
               model_path,
               thresholds=[-0.03664, 0.03664],
               input_shape=(120, 320, 3),
               use_adadelta=True,
               learning_rate=0.01,
               W_l2=0.0001):
        """
        """
        output_dim = 1 if len(thresholds) == 1 else len(thresholds) + 1

        model = Sequential()
        model.add(Convolution2D(16, 5, 5,
            input_shape=input_shape,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(20, 5, 5,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(100, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=256,
            init='he_normal',
            activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=output_dim,
            init='he_normal',
            W_regularizer=l2(W_l2),
            activation='sigmoid'))

        optimizer = ('adadelta' if use_adadelta
                     else SGD(lr=learning_rate, momentum=0.9))

        if len(thresholds) == 1:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['categorical_accuracy', 'top_2']

	model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)

        # Write model to designated path
        model.save(model_path)

        # Return model_config params compatible with constructor
        return {
            'type': CategoricalModel.TYPE,
            'model_uri': model_path,
            'thresholds': thresholds,
        }

class RegressionModel(BaseModel):
    TYPE = 'regression'

    def __init__(self, model_config):
        self.model = load_model_from_uri(model_config['model_uri'])
        self.scale = model_config.get('scale', 1.0)

    def fit(self, dataset, training_args, callbacks=None, final=False):
        batch_size = training_args.get('batch_size', 100)
        epochs = training_args.get('epochs', 5)

        self.model.summary()

        training_generator = (
            dataset.final_generator(batch_size) if final
            else dataset.training_generator(batch_size))

        if 'pctl_sampling' in training_args:
            pctl_sampling = training_args['pctl_sampling']
            pctl_thresholds = training_args.get('pctl_thresholds')
            training_generator = (training_generator
                .with_percentile_sampling(pctl_sampling, pctl_thresholds))

        training_generator = training_generator.scale_labels(self.scale)
        validation_generator = (dataset
            .validation_generator(batch_size)
            .scale_labels(self.scale))

        validation_size = training_args.get(
            'validation_size', validation_generator.get_size())
        epoch_size = training_args.get(
            'epoch_size', training_generator.get_size())

        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=(callbacks or []))

        return history

    def evaluate(self, dataset):
        generator = dataset.testing_generator(32)
        return std_evaluate(self, generator)

    def predict_on_batch(self, batch):
        return self.model.predict_on_batch(batch) / self.scale

    def make_stateful_predictor(self):
        return lambda x: self.predict_on_batch(x)[0, 0]

    def as_encoder(self):
        deep_copy = deep_copy_model(self.model)

        if isinstance(deep_copy, Sequential):
            # remove the last output layers to retain the feature maps
            while 'flatten' not in deep_copy.layers[-1].name:
                deep_copy.pop()
            return deep_copy_model_weights(deep_copy)
        else:
            deep_copy.layers.pop()
            deep_copy.outputs = [deep_copy.layers[-1].output]
            deep_copy.layers[-1].outbound_nodes = []

        return deep_copy

    def output_dim(self):
        return get_output_dim(self.model)

    def save(self, model_path):
        save_model(self.model, model_path)

        return {
            'type': RegressionModel.TYPE,
            'model_uri': model_path,
            'scale': self.scale,
        }

    @classmethod
    def create(cls,
               model_path,
               input_shape=(120, 320, 3),
               use_adadelta=True,
               learning_rate=0.01,
               W_l2=0.0001,
               scale=16):
        """
        """
        model = Sequential()
        model.add(Convolution2D(16, 5, 5,
            input_shape=input_shape,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(20, 5, 5,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(40, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(60, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(80, 2, 2,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(SpatialDropout2D(0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 2, 2,
            init= "he_normal",
            activation='relu',
            border_mode='same'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        optimizer = ('adadelta' if use_adadelta
                     else SGD(lr=learning_rate, momentum=0.9))

	model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['rmse'])

        # Write model to designated path
        model.save(model_path)

        # Return model_config params compatible with constructor
        return {
            'type': RegressionModel.TYPE,
            'model_uri': model_path,
            'scale': scale,
        }

    @classmethod
    def create_resnet_inception_v1(cls,
                                   model_path,
                                   input_shape=(120, 320, 3),
                                   learning_rate=0.01,
                                   W_l2=0.0001,
                                   scale=16):
        """
        """
        img_input = Input(shape=input_shape)

        # stem
        output = conv_5x5(32, subsample=(2, 2))(img_input)
        output = conv_5x5(64)(output)
        output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(output)
        output = conv_1x1(80)(output)
        output = conv_3x3(192)(output)
        output = conv_3x3(256, subsample=(2, 2))(output)
        output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(output)

        if False:
            # inception 1
            output = inception_layer(output, 20)
            output = BatchNormalization(axis=3)(output)
            output = inception_layer(output, 24)
            output = BatchNormalization(axis=3)(output)
        else:
            # resnet-inception 1
            output = inception_resnet_a()(output)
            output = inception_resnet_a()(output)
            output = inception_resnet_a()(output)

        # reduce 1
        output = MaxPooling2D(pool_size=(2, 2))(output)

        # conv 1
        output = conv_3x3(40)(output)

        # reduce 2
        output = MaxPooling2D(pool_size=(2, 2))(output)

        # conv 2
        output = conv_3x3(60)(output)
        output = conv_2x2(80)(output)
        output = conv_2x2(128)(output)

        # reduce 3
        output = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(output)

        # output
        output = Flatten()(output)
        output = Dropout(0.2)(output)
        output = Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2))(output)

        optimizer = SGD(momentum=0.9, lr=0.0045)
        optimizer = 'adadelta'

        model = Model(input=img_input, output=output)
	model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['rmse'])

        # Write model to designated path
        model.save(model_path)

        # Return model_config params compatible with constructor
        return {
            'type': RegressionModel.TYPE,
            'model_uri': model_path,
            'scale': scale,
        }

    @classmethod
    def create_resnet_inception_v2(cls,
                                   model_path,
                                   input_shape=(120, 320, 3),
                                   learning_rate=0.01,
                                   W_l2=0.0001,
                                   scale=16):
        """
        """
        img_input = Input(shape=input_shape)

        # smaller stem
        output = conv_5x5(64, subsample=(2, 2))(img_input)
        output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(output)
        output = conv_1x1(80)(output)
        output = conv_3x3(256, subsample=(2, 2))(output)
        output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(output)

        # resnet-inception a
        output = inception_resnet_a()(output)
        output = inception_resnet_a()(output)
        output = inception_resnet_a()(output)

        # reduce
        output = MaxPooling2D(pool_size=(2, 2))(output)

        # resnet-inception b
        output = inception_resnet_b()(output)
        output = inception_resnet_b()(output)
        output = inception_resnet_b()(output)

        # reduce
        output = MaxPooling2D(pool_size=(2, 2))(output)

        # conv 2
        output = conv_2x2(80)(output)
        output = conv_2x2(128)(output)

        # reduce 3
        output = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(output)

        # output
        output = Flatten()(output)
        output = Dropout(0.2)(output)
        output = Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2))(output)

        optimizer = SGD(momentum=0.9, lr=0.0045)
        optimizer = 'adadelta'

        model = Model(input=img_input, output=output)
	model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['rmse'])

        # Write model to designated path
        model.save(model_path)

        # Return model_config params compatible with constructor
        return {
            'type': RegressionModel.TYPE,
            'model_uri': model_path,
            'scale': scale,
        }


class MixtureModel(BaseModel):
    """
    """
    TYPE = 'mixture'

    def __init__(self, model_config):
        self.general_regression = load_from_config(
            model_config['general_regression'])
        self.sharp_regression = load_from_config(
            model_config['sharp_regression'])
        self.sharp_classifier = load_from_config(
            model_config['sharp_classifier'])
        self.sharp_bias = np.array([.05, 1, .05]).reshape(1, 3)

    def predict_on_batch(self, batch):
        shape = tuple([1] + list(batch.shape[1:]))
        output = np.empty((len(batch), 1), dtype='float64')
        p_cat = (
            self.sharp_bias * self.sharp_classifier.predict_on_batch(batch))

        for i, x in enumerate(batch):
            cat = np.argmax(p_cat[i])
            if cat == 1:
                p_angle = self.general_regression.predict_on_batch(
                    x.reshape(shape))[0, 0]
            else:
                sign = -1.0 if cat == 0 else 1.0
                x = x if sign == 1.0 else x[:, ::-1, :]
                p_angle = sign * self.sharp_regression.predict_on_batch(
                    x.reshape(shape))[0, 0]

            output[i] = p_angle / 16.

        return output

    def evaluate(self, dataset):
        batch_size = 32
        testing_size = dataset.get_testing_size()
        testing_generator = dataset.testing_generator(batch_size)
        n_batches = testing_size / batch_size

        err_sum = 0.
        err_count = 0.
        for _ in xrange(n_batches):
            X_batch, y_batch = testing_generator.next()
            y_pred = self.predict_on_batch(X_batch)
            err_sum += np.sum((y_batch - y_pred) ** 2)
            err_count += len(y_pred)
        mse = err_sum / err_count
        rmse = np.sqrt(mse)
        return [mse, rmse]

    def make_stateful_predictor(self,
                                smoothing=True,
                                smoothing_steps=10,
                                interpolation_weight=0.25,
                                max_abs_delta=0.026):
        if smoothing:
            assert smoothing_steps >= 2
            X = np.linspace(0, 1, smoothing_steps)
            x1 = 1 + X[1] - X[0]
            prev = deque()

        def predict_fn(x):
            p = self.predict_on_batch(x)[0, 0]

            if not smoothing:
                return p

            if len(prev) == smoothing_steps:
                m, b = np.polyfit(X, prev, 1)
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


class EnsembleModel(BaseModel):
    """
    """
    TYPE = 'ensemble'

    def __init__(self, model_config):
        self.input_model_config = model_config['input_model_config']
        self.input_model = load_from_config(
            self.input_model_config
        ).as_encoder()

        self.model = load_model_from_uri(model_config['model_uri'])

        self.timesteps = model_config['timesteps']
        self.timestep_noise = model_config['timestep_noise']
        self.timestep_dropout = model_config['timestep_dropout']

    def fit(self, dataset, training_args, callbacks=None):
        batch_size = training_args.get('batch_size', 100)
        epochs = training_args.get('epochs', 5)

        input_model = self.input_model
        timesteps = self.timesteps
        noise = self.timestep_noise
        dropout = self.timestep_dropout

        self.model.summary()

        batch, _ = (dataset
            .training_generator(batch_size)
            .next())

        # NOTE: for some reason if I don't call this then everything breaks
        # TODO: why?
        input_model.predict_on_batch(batch)

        training_generator = (dataset
            .training_generator(batch_size)
            .with_transform(input_model, timesteps, noise, dropout))

        validation_generator = (dataset
            .validation_generator(batch_size)
            .with_transform(input_model, timesteps, noise, dropout))

        validation_size = training_args.get(
            'validation_size', validation_generator.get_size())
        epoch_size = training_args.get(
            'epoch_size', training_generator.get_size())

        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=(callbacks or []))

    def evaluate(self, dataset):
        batch_size = 32
        testing_size = dataset.get_testing_size()
        testing_generator = dataset.testing_generator(batch_size)
        predictor = self.make_stateful_predictor()
        n_batches = testing_size / batch_size

        err_sum = 0.
        err_count = 0.
        for _ in xrange(n_batches):
            X_batch, y_batch = testing_generator.next()
            if self.timesteps == 0:
                y_pred = self.predict_on_batch(X_batch)
                err_sum += ((y_batch - y_pred) ** 2).sum()
                err_count += batch_size
            else:
                for i in xrange(batch_size):
                    transformed_batch = (self
                        .input_model
                        .predict_on_batch(X_batch))
                    y_pred = predictor(transformed_batch[i])
                    y_true = y_batch[i]
                    err_sum += (y_true - y_pred) ** 2
                    err_count += 1

        mse = err_sum / err_count
        return [mse, np.sqrt(mse)]

    def predict_on_batch(self, batch):
        transformed_batch = self.input_model.predict_on_batch(batch)
        if self.timesteps == 0:
            return self.model.predict_on_batch(transformed_batch)
        else:
            predictor = self.make_stateful_predictor()
            output_dim = get_output_dim(self.model)
            output = np.empty((len(batch), output_dim))

            for i in xrange(len(batch)):
                output[i] = predictor(transformed_batch[i])

            return output

    def make_stateful_predictor(self, apply_transform=False):
        default_prev = 0
        steps = deque([default_prev for _ in xrange(self.timesteps)])

        def predict_fn(input_features):
            if apply_transform:
                if len(input_features.shape) != 4:
                    input_features = input_features.reshape(
                        (1,) + input_features.shape)

                ensemble_features = (self.input_model
                    .predict([input_features])[0])
            else:
                ensemble_features = input_features.reshape(
                    (np.max(input_features.shape), ))

            input_dim = len(ensemble_features) + self.timesteps
            sample = (np
                .concatenate((ensemble_features, steps))
                .reshape((1, input_dim)))
            prediction = self.model.predict([sample])[0, 0]
            steps.popleft()
            steps.append(prediction)
            return prediction

        return predict_fn

    def save(self, model_path):
        save_model(self.model, model_path)

        return {
            'type': EnsembleModel.TYPE,
            'timesteps': self.timesteps,
            'timestep_noise': self.timestep_noise,
            'timestep_dropout': self.timestep_dropout,
            'model_uri': model_path,
            'input_model_config': self.input_model_config,
        }

    def output_dim(self):
        return get_output_dim(self.model)

    @classmethod
    def create(cls,
               model_path,
               input_model_config,
               timesteps=0,
               timestep_noise=0,
               timestep_dropout=0,
               layers=None,
               loss='mean_squared_error',
               learning_rate=0.001,
               momentum=0.9,
               W_l2=0.001,
               metrics=None):

        input_model = load_from_config(input_model_config).as_encoder()
        input_dim = get_output_dim(input_model) + timesteps
        layers = [input_dim] + (layers or [128, ])
        metrics = metrics or ['mse']
        sgd = SGD(lr=learning_rate,
                  momentum=momentum,
                  nesterov=False)

        model = Sequential()
        for input_dim, output_dim in zip(layers, layers[1:]):
            logger.info('Adding layer (%d, %d)', input_dim, output_dim)
            model.add(Dense(
                input_dim=input_dim,
                output_dim=output_dim,
                activation='relu',
                init='he_normal',
                bias=True,
                W_regularizer=l2(W_l2)))
            model.add(Dropout(0.5))

        model.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        model.compile(loss=loss, optimizer=sgd, metrics=metrics)

        # Write model to designated path
        model.save(model_path)

        return {
            'type': EnsembleModel.TYPE,
            'timesteps': timesteps,
            'timestep_noise': timestep_noise,
            'timestep_dropout': timestep_dropout,
            'model_uri': model_path,
            'input_model_config': input_model_config,
        }

class LstmModel(BaseModel):
    """
    """
    TYPE = 'lstm'

    def __init__(self, model_config):
        self.model = load_model_from_uri(model_config['model_uri'])
        self.timesteps = model_config['timesteps']
        self.scale = model_config.get('scale', 1.0)

    def fit(self, dataset, training_args, callbacks=None, final=False):
        batch_size = training_args.get('batch_size', 100)
        epochs = training_args.get('epochs', 5)

        timesteps = self.timesteps

        self.model.summary()

        training_generator = (
            dataset.final_generator(batch_size) if final
            else dataset.training_generator(batch_size))

        training_generator = (training_generator
            .scale_labels(self.scale)
            .with_timesteps(timesteps))

        validation_generator = (dataset
            .validation_generator(batch_size)
            .scale_labels(self.scale)
            .with_timesteps(timesteps))

        validation_size = training_args.get(
            'validation_size', validation_generator.get_size())
        epoch_size = training_args.get(
            'epoch_size', training_generator.get_size())

        callbacks = (callbacks or [])
        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=callbacks)

    def evaluate(self, dataset):
        batch_size = 32
        testing_size = dataset.get_testing_size()
        testing_generator = (dataset
            .testing_generator(batch_size)
            .with_timesteps(self.timesteps))
        n_batches = testing_size / batch_size

        err_sum = 0.
        err_count = 0.
        for _ in xrange(n_batches):
            X_batch, y_batch = testing_generator.next()
            y_pred = self.predict_on_batch(X_batch)
            err_sum += np.sum((y_batch - y_pred) ** 2)
            err_count += len(y_pred)
        mse = err_sum / err_count
        return [mse]

    def predict_on_batch(self, sequence):
        return self.model.predict_on_batch(sequence) / self.scale

    def save(self, model_path):
        save_model(self.model, model_path)

        return {
            'type': LstmModel.TYPE,
            'timesteps': self.timesteps,
            'model_uri': model_path,
            'scale': self.scale
        }

    def output_dim(self):
        return get_output_dim(self.model)

    @classmethod
    def create(cls,
               model_path,
               input_shape,
               batch_size=64,
               timesteps=0,
               loss='mean_squared_error',
               W_l2=0.001,
               metrics=None,
               scale=1.0):
        """
        Creates an LstmModel using a model in the input_model_config

        @param model_path - s3 uri to save the model
        @param input_shape - timestepped shape (timesteps, feature dims)
        @param batch_input_shape - (batch_size, feature dims)
        @param timesteps - timesteps inclusive of the current frame
                         (10 - current frame + 9 previous frames)
        @param loss - loss function on the model
        @param W_l2 - W_l2 regularization param
        @param metrics - metrics to track - (rmse, mse...)
        @param scale - factor by which to scale the labels
        """

        metrics = metrics or ['rmse']

        model = Sequential()
        model.add(TimeDistributed(Convolution2D(24, 5, 5,
            init= "he_normal",
            activation='relu',
            subsample=(5, 4),
            border_mode='valid'), input_shape=input_shape))
        model.add(TimeDistributed(Convolution2D(32, 5, 5,
            init= "he_normal",
            activation='relu',
            subsample=(3, 2),
            border_mode='valid')))
        model.add(TimeDistributed(Convolution2D(48, 3, 3,
            init= "he_normal",
            activation='relu',
            subsample=(1,2),
            border_mode='valid')))
        model.add(TimeDistributed(Convolution2D(64, 3, 3,
            init= "he_normal",
            activation='relu',
            border_mode='valid')))
        model.add(TimeDistributed(Convolution2D(128, 3, 3,
            init= "he_normal",
            activation='relu',
            subsample=(1,2),
            border_mode='valid')))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
        model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
        model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(
            output_dim=256,
            init='he_normal',
            activation='relu',
            W_regularizer=l2(W_l2)))
        model.add(Dropout(0.2))
        model.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        model.compile(loss=loss, optimizer='adadelta', metrics=metrics)

        # Write model to designated path
        model.save(model_path)

        return {
            'type': LstmModel.TYPE,
            'timesteps': timesteps,
            'model_uri': model_path,
            'scale': scale
        }


class TransferLstmModel(BaseModel):
    """
    """
    TYPE = 'transfer-lstm'

    def __init__(self, model_config):
        self.transform_model_config = model_config['transform_model_config']
        self.transform_model = load_from_config(self.transform_model_config)
        self.model = load_model_from_uri(model_config['model_uri'])
        self.timesteps = model_config['timesteps']
        self.scale = model_config.get('scale', 1.0)
        self.concat_original = model_config.get('concat_original', False)

    def fit(self, dataset, training_args, callbacks=None, final=False):
        batch_size = training_args.get('batch_size', 100)
        epochs = training_args.get('epochs', 5)
        timesteps = self.timesteps

        self.model.summary()

        training_generator = (
            dataset.final_generator(batch_size) if final
            else dataset.training_generator(batch_size))

        training_generator = (training_generator
            .with_concat_original(self.concat_original)
            .scale_labels(self.scale)
            .precompute_transform(self.transform_model)
            .with_timesteps(timesteps=timesteps))

        validation_generator = (dataset
            .validation_generator(batch_size)
            .with_concat_original(self.concat_original)
            .scale_labels(self.scale)
            .with_precomputed(training_generator.precomputed)
            .with_timesteps(timesteps=timesteps))

        validation_size = training_args.get(
            'validation_size', validation_generator.get_size())
        epoch_size = training_args.get(
            'epoch_size', training_generator.get_size())

        callbacks = (callbacks or [])
        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=callbacks)

    def evaluate(self, dataset):
        testing_generator = (dataset
            .testing_generator(32)
            .with_concat_original(self.concat_original)
            .precompute_transform(self.transform_model)
            .with_timesteps(timesteps=self.timesteps))

        return std_evaluate(self, testing_generator)

    def make_stateful_predictor(self):
        encoder = self.transform_model.as_encoder()
        steps = deque()

        def predict_fn(x):
            # apply feature extractor
            x_encoded = encoder.predict_on_batch(x)

            # initial fill of timesteps
            if not len(steps):
                for _ in xrange(self.timesteps):
                    steps.append(x_encoded)

            # put most recent features at end
            steps.popleft()
            steps.append(x_encoded)

            timestepped_x = np.empty((1, self.timesteps) + x_encoded.shape[1:])
            for i, img in enumerate(steps):
                timestepped_x[0, i] = img

            if self.concat_original:
                original_x = x.reshape((1, 120, 320, 3))
                batch = [timestepped_x, original_x]
            else:
                batch = timestepped_x

            return self.model.predict_on_batch(batch)[0, 0] / self.scale

        return predict_fn

    def save(self, model_path):
        save_model(self.model, model_path)

        return {
            'type': TransferLstmModel.TYPE,
            'timesteps': self.timesteps,
            'model_uri': model_path,
            'transform_model_config': self.transform_model_config,
            'scale': self.scale,
        }

    def output_dim(self):
        return get_output_dim(self.model)

    @classmethod
    def create(cls,
               model_path,
               transform_model_config,
               input_shape,
               batch_size=16,
               timesteps=3,
               W_l2=0.001,
               scale=16,
               metrics=None):
        """
        Creates a TransferLstmModel

        @param model_uri - s3 uri to save the model
        @param input_shape - timestepped shape (timesteps, feature dims)
        @param transform_model_config - transform model config
        @param input_shape - (batch_size, feature dims..)
        @param timesteps - timesteps inclusive of the current frame
                         (10 - current frame + 9 previous frames)
        @param W_l2 - l2 regularization weight
        @param metrics - metrics to track - (rmse, mse...)
        """
        metrics = metrics or ['rmse']

        transform_model = load_from_config(transform_model_config)
        input_dim = get_output_dim(transform_model.as_encoder())

        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(timesteps, input_dim),
            dropout_W=0.1,
            dropout_U=0.1,
            return_sequences=True))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        model.compile(
            loss='mean_squared_error',
            optimizer='adadelta',
            metrics=metrics)

        # Write model to designated path
        model.save(model_path)

        return {
            'type': TransferLstmModel.TYPE,
            'timesteps': timesteps,
            'model_uri': model_path,
            'transform_model_config': transform_model_config,
            'scale': scale,
            'concat_original': False,
        }

    @classmethod
    def create_cnn(cls,
                   model_path,
                   transform_model_config,
                   input_shape,
                   batch_size=16,
                   timesteps=3,
                   W_l2=0.001,
                   scale=16,
                   metrics=None):
        """
        Creates a TransferLstmModel

        @param model_uri - s3 uri to save the model
        @param input_shape - timestepped shape (timesteps, feature dims)
        @param transform_model_config - transform model config
        @param input_shape - (batch_size, feature dims..)
        @param timesteps - timesteps inclusive of the current frame
                         (10 - current frame + 9 previous frames)
        @param W_l2 - l2 regularization weight
        @param metrics - metrics to track - (rmse, mse...)
        """
        metrics = metrics or ['rmse']

        transform_model = load_from_config(transform_model_config)
        input_dim = get_output_dim(transform_model.as_encoder())

        lstm = Sequential()
        lstm.add(LSTM(
            256,
            input_shape=(timesteps, input_dim),
            dropout_W=0.1,
            dropout_U=0.1,
            return_sequences=True))
        lstm.add(Flatten())
        lstm.add(Dense(256, activation='relu'))

        cnn = Sequential()
        cnn.add(Convolution2D(
            24, 5, 5,
            input_shape=input_shape,
            subsample=(2, 2),
            activation='relu',
            init='he_normal',
            border_mode='valid'))
        cnn.add(Convolution2D(
            128, 3, 3,
            subsample=(2, 2),
            activation='relu',
            init='he_normal',
            border_mode='valid'))
        cnn.add(Convolution2D(
            256, 2, 2,
            subsample=(2, 2),
            activation='relu',
            init='he_normal',
            border_mode='valid'))
        cnn.add(Flatten())
        cnn.add(Dense(256, activation='relu'))

        merged = Sequential()
        merged.add(Merge([lstm, cnn], mode='concat'))
        merged.add(Dense(
            output_dim=1,
            init='he_normal',
            W_regularizer=l2(W_l2)))

        merged.compile(
            loss='mean_squared_error',
            optimizer='adadelta',
            metrics=metrics)

        # Write model to designated path
        merged.save(model_path)

        return {
            'type': TransferLstmModel.TYPE,
            'timesteps': timesteps,
            'model_uri': model_path,
            'transform_model_config': transform_model_config,
            'scale': scale,
            'concat_original': True,
        }



def inception_layer(model, tower_size):
    """
    """
    tower_1 = conv_1x1(tower_size, normal=False)(model)

    tower_2 = conv_1x1(tower_size, normal=False)(model)
    tower_2 = conv_3x3(tower_size, normal=False)(tower_2)

    tower_3 = conv_1x1(tower_size, normal=False)(model)
    tower_3 = conv_5x5(tower_size, normal=False)(tower_3)

    tower_4 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        border_mode='same')(model)
    tower_4 = conv_1x1(tower_size, normal=False)(tower_4)

    return merge(
        [tower_1, tower_2, tower_3, tower_4],
        mode='concat',
        concat_axis=3)


def inception_resnet_a():
    def fn(input):
        norm = BatchNormalization(axis=3)(input)
        relu = Activation('relu')(input)

        tower_1 = no_relu_conv_mxn(32, 1, 1)(relu)

        tower_2 = no_relu_conv_mxn(32, 1, 1)(relu)
        tower_2 = no_relu_conv_mxn(32, 3, 3)(tower_2)

        tower_3 = no_relu_conv_mxn(32, 1, 1)(relu)
        tower_3 = no_relu_conv_mxn(32, 3, 3)(tower_3)
        tower_3 = no_relu_conv_mxn(32, 3, 3)(tower_3)

        towers = merge(
            [tower_1, tower_2, tower_3],
            mode='concat',
            concat_axis=3)

        linear_conv = no_relu_conv_mxn(256, 1, 1)(towers)

        return merge([relu, linear_conv], mode='sum')

    return fn

def inception_resnet_b():
    def fn(input):
        norm = BatchNormalization(axis=3)(input)
        relu = Activation('relu')(input)

        tower_1 = no_relu_conv_mxn(128, 1, 1)(relu)

        tower_2 = no_relu_conv_mxn(128, 1, 1)(relu)
        tower_2 = no_relu_conv_mxn(128, 1, 5)(tower_2)
        tower_2 = no_relu_conv_mxn(128, 5, 1)(tower_2)

        towers = merge(
            [tower_1, tower_2],
            mode='concat',
            concat_axis=3)

        linear_conv = no_relu_conv_mxn(256, 1, 1)(towers)

        return merge([relu, linear_conv], mode='sum')

    return fn


def conv_5x5(n_filters, subsample=(1, 1), normal=True):
    return conv_mxn(n_filters, 5, 5, subsample, normal)


def conv_3x3(n_filters, subsample=(1, 1), normal=True):
    return conv_mxn(n_filters, 3, 3, subsample, normal)


def conv_2x2(n_filters, subsample=(1, 1), normal=True):
    return conv_mxn(n_filters, 2, 2, subsample, normal)


def conv_1x1(n_filters, subsample=(1, 1), normal=True):
    return conv_mxn(n_filters, 1, 1, subsample, normal)


def conv_mxn(n_filters, m, n, subsample=(1, 1), normal=True):
    def fn(input):
        conv = Convolution2D(
            n_filters, m, n,
            activation='relu',
            border_mode='same')(input)

        if normal:
            conv = BatchNormalization(axis=3)(conv)

        return conv

    return fn


def no_relu_conv_mxn(n_filters, m, n):
    def f(input):
        return Convolution2D(
            n_filters, m, n,
            init='he_normal',
            border_mode='same')(input)

    return f


MODEL_CLASS_BY_TYPE = {
    'simple': CategoricalModel,  # backwards compat
    CategoricalModel.TYPE: CategoricalModel,
    EnsembleModel.TYPE: EnsembleModel,
    RegressionModel.TYPE: RegressionModel,
    LstmModel.TYPE: LstmModel,
    TransferLstmModel.TYPE: TransferLstmModel,
    MixtureModel.TYPE: MixtureModel,
}


def load_from_config(model_config):
    """
    Loads a model from config by looking up it's model type and
    calling the right constructor.

    NOTE: Requires model_config to have 'type' key.

    @param model_config - appropriate model_config dict for model type
    @return - model object
    """
    model_type = model_config['type']
    return MODEL_CLASS_BY_TYPE[model_type](model_config)


def save_model(model, model_path):
    """
    Save a keras model to a local path.

    @param model - keras model
    @param model_path - local path to write to
    """
    try: os.makedirs(os.path.dirname(model_path))
    except: pass

    model.save(model_path)

def load_model_from_uri(model_path_or_uri, cache_dir='/tmp'):
    """
    Download, deserialize, and load a keras model into memory.

    @param model_path_or_uri - absolute path or s3_uri
    @param cache_dir - where to cache model files
    """
    if model_path_or_uri.startswith('s3'):
        bucket, key = parse_s3_uri(model_path_or_uri)
        model_path = os.path.join(cache_dir, key)
        download_file(bucket, key, model_path)
    else:
        model_path = model_path_or_uri

    return keras_load_model(model_path)


def deep_copy_model(model):
    """
    Create a deep copy of a tensorflow model.

    @param model - tensorflow model
    @return - copy of model
    """
    _, tmp_path = tempfile.mkstemp()
    try:
        model.save(tmp_path)
        input_model = keras_load_model(tmp_path)
        return input_model
    finally:
        os.remove(tmp_path)

def deep_copy_model_weights(model):
    """
    Create a deep copy of a tensorflow model weights.

    @param model - tensorflow model
    @return - copy of model
    """
    _, tmp_path_json = tempfile.mkstemp()
    _, tmp_path_weights = tempfile.mkstemp()
    try:
        # serialize model to JSON
        model_json = model.to_json()
        with open(tmp_path_json, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(tmp_path_weights)

        # load json and create model
        json_file = open(tmp_path_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(tmp_path_weights)
        return loaded_model
    finally:
        print(tmp_path_json)
        print(tmp_path_weights)

def get_output_dim(model):
    """
    Infer output dimension from model by inspecting its layers.

    @param model - tensorflow model
    @return - output dimension
    """
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape'):
            return layer.output_shape[-1]

    raise ValueError('Could not infer output dim')


def std_evaluate(model, generator):
    """
    """
    size = generator.get_size()
    batch_size = generator.get_batch_size()
    n_batches = size / batch_size

    err_sum = 0.
    err_count = 0.
    for _ in xrange(n_batches):
        X_batch, y_batch = generator.next()
        y_pred = model.predict_on_batch(X_batch)
        err_sum += np.sum((y_batch - y_pred) ** 2)
        err_count += len(y_pred)

    mse = err_sum / err_count
    return [mse, np.sqrt(mse)]


def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def top_2(y_true, y_pred):
    return K.mean(tf.nn.in_top_k(y_pred, K.argmax(y_true, axis=-1), 2))

metrics.rmse = rmse
metrics.top_2 = top_2
