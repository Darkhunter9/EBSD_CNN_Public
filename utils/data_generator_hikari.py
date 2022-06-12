import numpy as np
import h5py
import cv2
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from .eu2qu import eu2qu
from .imgprocess import *
from .eu_txt_reader import eu_txt_reader

'''
data generator for hikari dataset
'''

class DataGenerator_hikari(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, scan=10, batch_size=32, dim=(60,60), n_channels=1, shuffle=False, processing=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.processing = processing
        self.f = h5py.File(data, 'r')
        self.X = self.f['Scan '+str(scan)]['EBSD']['Data']['Pattern']
        self.phi = self.f['Scan '+str(scan)]['EBSD']['Data']['Phi']
        self.phi1 = self.f['Scan '+str(scan)]['EBSD']['Data']['Phi1']
        self.phi2 = self.f['Scan '+str(scan)]['EBSD']['Data']['Phi2']
        print('load data successfully')
        self.on_epoch_end()

        # judge whether resize is needed
        self.resize = (dim == self.X.shape[1:])

        '''
        For now the new indexing result is used (with refined pattern center), so label is no longer read from h5 file, in stead from a seperate txt file
        '''

        self.y = eu_txt_reader('/home/zihaod/CNN/data/Marc/Hough_new.txt', unit='rad')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # To make sure indexing elements are in increasing order
        # Otherwise TypeError: Indexing elements must be in increasing order
        indexes = np.sort(indexes)

        # Generate data
        X = np.array(self.X[indexes,:,:])
        X = X.astype(float)

        # resize
        if not self.resize:
            X_new = np.zeros((len(X),self.dim[0],self.dim[1]))
            for i in range(len(X)):
                X_new[i] = cv2.resize(X[i],(self.dim[1],self.dim[0]),interpolation=cv2.INTER_LINEAR)
        else:
            X_new = X
        X_new = X_new.astype(np.uint8)

        # preprocessing
        if self.processing:
            for i in self.processing:
                X_new = eval(i.replace('(','(X_new,',1))

        X_new = X_new.astype(float)
        X_new = np.clip(np.nan_to_num(X_new),0.,255.)
        X_new = X_new / 255.0
        X_new = X_new[:,:,:,np.newaxis]

        y = self.y[indexes].reshape((-1,4))

        # shuffle inside the batch
        if self.shuffle:
            return shuffle(X_new, y)
        else:
            return X_new, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def close(self):
        self.f.close()

if __name__ == '__main__':
    # Generators
    training_generator = DataGenerator(data='dir/to/Ni_testing.h5',
                                        batch_size=32, dim=(480,640), n_channels=1, shuffle=False)
