import sys
import os
import argparse
import gc

import time
import tqdm
import csv 

import numpy as np
import pandas as pd
import scipy as sp
import cv2
import tensorflow as tf

import keras

class Trainer():

	def __init__(self):
		self.batch_size = 4
		self.train_file_path = 'driving_log.csv'
		self.model_name = 'model.h5'
		self.epochs = 1
		self.eras = 100


	def batch_creator(self,ix):
	    """Create batch with random samples and return appropriate format"""
	    batch = ix.sample(n = self.batch_size)
	    #print(batch)
	    temp_x = []
	    temp_y = []
	    for index, row in batch.iterrows():

	        center_image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), str(row["center"]))
	        center_image = cv2.imread(center_image_path)
        	center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

        	steering = float(row["steering"])

	        temp_x.append(center_image)
	        temp_y.append(steering)
	        del center_image
	        gc.collect()
	    batch_x = np.stack(temp_x)
	    batch_y = np.stack(temp_y)
	    
	    del batch, temp_x, temp_y
	    gc.collect()

	    return batch_x, batch_y


	def train(self):
	    """ Trains classifier in batches """
	    dir_path = os.path.dirname(os.path.realpath(__file__))
	    data_ix = pd.read_csv(os.path.join(dir_path,self.train_file_path), names=['center','left','right','steering','throttle','brake','speed'], header=0)


	    model = keras.models.Sequential()
	    model.add(keras.layers.Flatten(input_shape=(160,320,3)))
	    model.add(keras.layers.Dense(1))
	    model.compile(loss='mse', optimizer='adam')


	    for era in range(self.eras):
		    batch_x, batch_y = self.batch_creator(data_ix)
		    print("\nBatch has shape: {} and {}".format(batch_x.shape, batch_y.shape))
		    model.fit(batch_x, batch_y, validation_split=0.2, shuffle=True, epochs=self.epochs)


	    print("Saving model at {}".format(self.model_name))
	    model.save(self.model_name)


if __name__ == '__main__':
	trainer = Trainer()
	trainer.train()