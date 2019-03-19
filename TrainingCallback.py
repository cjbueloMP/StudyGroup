from tensorflow.keras.callbacks import Callback
import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

'''
    This is a traing callback that the fitting algorithm will run during training
'''
class TraingCallback(Callback):
    def __init__(self,data):
        self.data = data
        
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure(figsize=(10,3))
        self.logs = []
        self.floor_epoch = 0

    #def on_train_end( self, logs={}):
        # Do nothing

    #def on_batch_begin(self, batch, logs={}): 
        # Do nothing 

    def on_batch_end(self, batch, logs={}):

        if batch%10==0:
            self.losses.append(logs.get('loss'))

            clear_output(wait=True)
            self.fig = plt.figure(figsize=(10,3))

            data = self.data

            '''
                Run a test case
            '''
            # Test with above image
            testim = data['Val_X'][0,...][np.newaxis,...]
            predicted_image = self.model.predict(x=testim)
            plt.subplot(132)
            predicted_slice = np.squeeze(predicted_image)
            plt.imshow(predicted_slice[...,8], cmap='gray',vmin=0,vmax=1)
            plt.title('Predicted Image')
            plt.axis('off')

            testout = data['Val_Y'][0,...][...,np.newaxis]
            plt.subplot(133)
            plt.imshow(testout[...,8,0],cmap='gray',vmin=0,vmax=1)
            plt.title('True Image')
            plt.axis('off')

            # Using just this one image to get a loss
            self.val_losses.append( self.model.evaluate(x=testim,y=testout[np.newaxis,...],verbose=False))

            '''
            Plot the Losses 
            '''
            plt.subplot(131)
            plt.semilogy(self.losses, label="Loss")
            plt.semilogy(self.val_losses, label="Loss (test image)")
            plt.legend()

            print('Epoch = ' + str(self.floor_epoch) + 'Loss = ' + str(logs.get('loss')) )
            plt.show();

    def on_epoch_begin(self,epoch,logs={}):
        self.floor_epoch = epoch

