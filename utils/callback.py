import warnings
import six
import os
import io
import csv
import numpy as np
from math import exp
from collections import Iterable
from collections import OrderedDict
import tensorflow as tf
from keras.callbacks import Callback
from .disorientation import disorientation

def ExponentialDecay(epoch, lr):
    k = 0.2
    lr_new = lr * exp(-k*epoch)
    return lr_new

def StepDecay(epoch, lr):
    decay_rate = 0.1
    decay_steps = 3
    lr_new = lr * decay_rate**np.floor(epoch / decay_steps)
    return lr_new

class TerminateOnZero(Callback):
    """Callback that terminates training when a Zero loss is encountered.
    """
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.count_nonzero(loss) == 0 or np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss (0), terminating training' % (batch))
                with tf.Session() as sess:
                    print('y_true: \n', sess.run([self.model.targets]))
                    print('y_pred: \n', sess.run([self.model.outputs]))
                self.model.stop_training = True

class MultiGPUCheckpointCallback(Callback):
    """
    Save the model after every epoch.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.

    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    """

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

class AdditionalValidationSets(Callback):
    def __init__(self, validation_generator, verbose=0):
        """
        place this callback before CSVLogger
        Callbacks that appear later in the list would receive a modified version of logs

        :param validation_generator:
        validation_generator
        :param verbose:
        verbosity mode, 1 or 0
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_generator = validation_generator
        self.epoch = []
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        # evaluate on the additional validation sets
        mean_loss = self.model.evaluate_generator(self.validation_generator,
                                        max_queue_size=5,
                                        use_multiprocessing=False,
                                        verbose=self.verbose)

        result = np.array([[0]])
        batches = len(self.validation_generator)
        for i in range(batches):
            X, y_true = self.validation_generator[i]
            y_pred = self.model.predict_on_batch(X)
            d = disorientation(y_true, y_pred)
            result = np.concatenate((result,d))
        result = result[1:]
        mean_disorientation = np.average(result)

        logs['hikari_loss'] = mean_loss
        logs['hikari_disorientation'] = mean_disorientation
