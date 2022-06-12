from copy import deepcopy
from math import exp, floor, log10, ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image

# the use of class and multiprecessing in Keras 
# will cause multiple cpus read/write self.img at the same time, which may lead to a bug

# Adaptive histogram equalization
def clahe(img, limit=10, tileGridSize=(10, 10)):
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=tileGridSize)
    temp = np.zeros_like(img)
    for i in range(img.shape[0]):
        temp[i] = clahe.apply(img[i])
    return temp

# circular mask
def circularmask(img):
    center = [int(img.shape[2]/2), int(img.shape[1]/2)]
    radius = min(center[0], center[1], img.shape[2]-center[0], img.shape[1]-center[1])
    Y, X = np.ogrid[:img.shape[1], :img.shape[2]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask_array = dist_from_center <= radius
    temp = img
    temp[:,~mask_array] = 0
    return temp

# square mask
def squaremask(img):
    '''
    get the bigest square inside image with circular mask
    will change the input size
    '''
    n = img.shape[1]
    start = ceil(n*0.5*(1.-0.5**0.5))
    end = floor(n-n*0.5*(1.-0.5**0.5))
    return img[:,start-1:end,start-1:end]

def poisson_noise(img, c=1.):
    '''
    produce poisson noise on given images
    Smaller c brings higher noise level
    '''
    temp = np.zeros_like(img)

    for i in range(img.shape[0]):
        vals = len(np.unique(img[i]))
        vals = 2 ** np.ceil(np.log2(vals))
        temp[i] = np.random.poisson(img[i] * c * vals) / float(vals) / c
    
    return temp

def bac(img, a=1, b=0):
    temp = np.clip(a*img+b, 0., 255.)
    temp = temp.astype(np.uint8)
    return temp

def gamma_trans(img, gamma):
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    temp = np.zeros_like(img)
    for i in range(img.shape[0]):
        temp[i] = cv2.LUT(img[i], gamma_table)
    temp = temp.astype(np.uint8)
    return temp
