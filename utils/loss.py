from keras import backend as K
from keras import layers
import tensorflow as tf
import numpy as np
from math import sin, cos, pi
import time
from utils.disorientation import disorientation

'''
The use of acos will easily make loss 0 as its derivative has value restrictions.
'''

def loss_qu(y_true, y_pred):
    '''
    calculating 'distance' between two quaternions
    1 - <y_true, y_pred> = cos(theta/2)
    not considering the rotation axis
    '''

    loss = 1 - K.mean(K.batch_dot(y_true, y_pred, axes=1))
    # loss = 360./pi*K.mean(
    #     tf.math.acos(
    #         tf.clip_by_value(
    #             tf.reshape(K.batch_dot(y_true, y_pred, axes=1), shape=[-1,1]),
    #         clip_value_min=-1, clip_value_max=1)))

    return 1e4*loss

def loss_disorientation(y_true, y_pred):
    '''
    only works for cubic material
    point group higher than 432

    calculating disorientation angle between two quaternions
    not considering the rotation axis

    input: y_true: tf tensor, 2darray of quaternion labels, n*4
            y_pred: tf tensor, 2darray of quaternion predictions, n*4
    output: tf tensor, mean of disorientation angles between all labels and predictions, (1,)
            in unit of rad
    '''

    y_pred_star = tf.multiply(y_pred, tf.constant([[1.,-1.,-1.,-1.]]))

    temp = tf.reshape(K.sum(tf.multiply(y_true,y_pred),axis=-1),shape=[-1,1]) # w of misorientation
    temp2 = tf.multiply(y_pred_star[:,1:],y_true[:,0:1]) + tf.multiply(y_true[:,1:],y_pred_star[:,0:1]) + tf.cross(y_true[:,1:],y_pred_star[:,1:]) # x y z of misorientation
    misorientation = tf.sort(tf.abs(tf.concat([temp,temp2],axis=-1)), axis=-1, direction='DESCENDING') # sort w >= x >= y >= z >= 0

    temp3 = tf.concat((misorientation[:,0:1], # w0 == w
                    tf.reshape(0.5*K.sum(misorientation,axis=-1),shape=[-1,1]), # w1 == (w+x+y+z)/2
                    tf.reshape(0.5**0.5*K.sum(misorientation[:,0:2],axis=-1),shape=[-1,1])),axis=-1) # w2 == (w+x) * sqrt（2）/ 2
    
    # the mean disorientation angle is:
    return 36000./pi*K.mean(K.tf.acos(K.clip(K.max(temp3, axis=-1),-1.0+K.epsilon(),1.0-K.epsilon()))) # average(degrees(2*acos(max(w0, w1, w2))))


# test 
if __name__ == '__main__':
    pass
