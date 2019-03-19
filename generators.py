from tensorflow.keras.utils import Sequence
import numpy as np
import h5py

class chunkgenerator(Sequence):
    def __init__(self, obj, x_set, y_set, batch_size):
        self.file = obj;
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.num_samples = obj[self.x].shape[0]

    def __len__(self):
        return int(np.ceil(self.num_samples/float(self.batch_size)))

    def __getitem__(self, idx):

        
        x = self.file[self.x][idx*self.batch_size:(idx+1)*self.batch_size,...]
        y = self.file[self.y][idx*self.batch_size:(idx+1)*self.batch_size,...][...,np.newaxis]
        return x,y
