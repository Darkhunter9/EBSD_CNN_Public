from __future__ import print_function
from __future__ import absolute_import
 
import warnings
import numpy as np
import os
import h5py
import time
import sys

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
import keras.backend.tensorflow_backend as KTF
from keras import layers
from keras.layers import (Dense, LeakyReLU, Input, BatchNormalization, Activation,
                        Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D,
                        GlobalMaxPooling2D, Lambda, Flatten, Dropout, concatenate)
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger,
                            TerminateOnNaN, EarlyStopping, LearningRateScheduler)
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils import multi_gpu_model
from keras.constraints import maxnorm
from keras.applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape

from utils.data_generator_hough import DataGenerator, DataGenerator_hikari
from utils.callback import (MultiGPUCheckpointCallback, ExponentialDecay,
                            TerminateOnZero, AdditionalValidationSets)
from utils.loss import loss_qu, loss_disorientation
from utils.activation import sigmoid2
from utils.eu2qu import eu2qu
from utils.disorientation import disorientation
from utils.imgprocess import *

# assign gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
# fix random seed for reproducibility
# seed = 7
seed = int(time.time())
np.random.seed(seed)


def EBSD_CNN_Model_Small(include_top=True, weights=None,
                        input_tensor=None, input_shape=None,
                        pooling=None, *args, **kwargs):

    # pre-processing
    if K.backend() != 'tensorflow':
        raise RuntimeError('The model is only available with '
                           'the TensorFlow backend.')
    if K.image_data_format() != 'channels_last':
        warnings.warn('The model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right nlow will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=60,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor 


    # Entry flow
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='block1_conv1', kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(img_input)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='block1_conv2', kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    residual0 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block2_sepconv1', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block2_sepconv2', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = layers.add([x, residual0])

    residual1 = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block3_sepconv1', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block3_sepconv2', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = layers.add([x, residual1])

    residual2 = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False, kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), strides=(2,2), padding='same', use_bias=False, name='block4_sepconv2', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = layers.add([x, residual2])

    # Exit flow 
    residual3 = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block13_sepconv1', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3),strides=(2,2), padding='same', use_bias=False, name='block13_sepconv2', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = layers.add([x, residual3])
 
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block14_sepconv1', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block14_sepconv2', depthwise_constraint=maxnorm(5), pointwise_constraint=maxnorm(5), depthwise_initializer='he_normal', pointwise_initializer='he_normal')(x)
    x = Activation('relu')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dropout(0.2)(x)
        x = Dense(4096, name='dense1', kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(2048, name='dense2', kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, name='dense3', kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, name='dense4', kernel_constraint=maxnorm(5), kernel_initializer='he_normal')(x)
        x = Dense(4,  name='dense5', kernel_initializer='he_normal')(x)
        x = Lambda(lambda t: K.l2_normalize(100*t, axis=1), name='normalization')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='EBSD_CNN_Model_Small')

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


if __name__ == '__main__':
    params = {'dim': (60,60),
            'batch_size': 16,
            'n_channels': (1,),
            'shuffle': True}
    model = EBSD_CNN_Model_Small(include_top=True, weights=None, input_shape=params['dim']+params['n_channels'])

    imgprocesser1 = ['clahe(10, (4, 4))','circularmask()',]
    imgprocesser2 = ['clahe(10, (4, 4))','circularmask()',]
    training_generator = DataGenerator(data='dir/to/Ni_training_extended.h5',
                                        batch_size=params['batch_size'], dim=params['dim'], n_channels=1, shuffle=True, width=params['dim'][1], processing=imgprocesser1)
    testing_generator = DataGenerator(data='dir/to/Ni_training_test.h5',
                                        batch_size=params['batch_size'], dim=params['dim'], n_channels=1, shuffle=False, width=params['dim'][1], processing=imgprocesser1)
    hikari_generator = DataGenerator_hikari(data='dir/to//HikariNiSequence.h5', scan=10,
                                        batch_size=params['batch_size'], dim=params['dim'], n_channels=1, shuffle=False, width=params['dim'][1], processing=imgprocesser2)

    # Compile model
    epochs = 120
    lrate = 0.0005
    decay = 0.25 / len(training_generator) # To make lr at the final of each epoch = 0.8 * lr at the first of this epoch
    adam = Adam(lr=lrate, decay=decay)
    # if training in parallel
    parallel_model = multi_gpu_model(model, gpus=2)
    # if training on single gpu
    # parallel_model = model
    parallel_model.compile(loss=loss_qu, optimizer=adam)
    print(model.summary())

    # All the callbacks
    # define the checkpoint
    filepath = './EBSD_CNN_Model.h5'
    # check point changed to save only original model, with all the weights shared
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint = MultiGPUCheckpointCallback(filepath, model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # live training visualization through tensorflow
    tbCallBack = TensorBoard(log_dir='./logs',  # log path
                    histogram_freq=0,  # frequency of epochs to plot histogram, 0 means no plot
                    #  batch_size=32,   # batch_size for histogram
                    write_graph=True,  # visualiza graph (structure)
                    write_grads=True,  # visualize histogram
                    write_images=True, # visualize parameters
                    embeddings_freq=0, 
                    embeddings_layer_names=None, 
                    embeddings_metadata=None,
                    update_freq='epoch')
    # learning rate auto-reduction
    lrCallBack = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto', cooldown=1, min_lr=0)
    # performance on hikari
    hikari_test = AdditionalValidationSets(hikari_generator)
    # CSV logger
    csv = CSVLogger('./EBSD_CNN.csv', separator=',', append=False)
    # terminate on Nan
    terminateonnan = TerminateOnNaN()
    terminateonzero = TerminateOnZero()
    # construct callback_list
    callbacks_list = [hikari_test, checkpoint, tbCallBack, lrCallBack, csv]

    # Fit the model
    history = parallel_model.fit_generator(generator=training_generator,
                                validation_data=testing_generator,
                                max_queue_size=5, 
                                epochs=epochs,
                                callbacks=callbacks_list, 
                                shuffle=True, 
                                verbose=1, 
                                initial_epoch=0, 
                                use_multiprocessing=True, workers=10)

    training_generator.close()
    testing_generator.close()
