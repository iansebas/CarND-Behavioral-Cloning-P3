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

from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Trainer():

	def __init__(self):

		self.model = None
		self.batch_size = None
		self.train_file_path = 'driving_log.csv'
		self.model_name = 'model.h5'
		self.epochs = 5

		self.computing_setting()
		self.define_model()

	def computing_setting(self):
	    if tf.test.is_gpu_available():
	        print('\nGPU FOUND')
	        self.batch_size = 128
	        print("Batch Size: {}".format(self.batch_size))
	        print("CUDA ENABLED: {}".format(tf.test.is_gpu_available(cuda_only=True)))
	    else:
	        print('\nGPU NOT FOUND, USING CPU')
	        self.batch_size = 4
	        print("Batch Size: {}".format(self.batch_size))


	def batch_generator(self,ix):
	    """Create batch with random samples and return appropriate format"""
	    while True:
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

		        temp_x.append(cv2.flip(center_image,1))
		        temp_y.append(steering*-1.0)

		        del center_image
		        gc.collect()
		    batch_x = np.stack(temp_x)
		    batch_y = np.stack(temp_y)
		    
		    del batch, temp_x, temp_y
		    gc.collect()

		    yield batch_x, batch_y

	def define_model(self):

		model = keras.models.Sequential()
		model.add(keras.layers.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
		model.add(keras.layers.Cropping2D(cropping=((70,25),(0,0))))

		model.add(keras.layers.convolutional.Convolution2D(24, (5, 5), strides=(2,2), activation='relu'))
		model.add(keras.layers.convolutional.Convolution2D(36, (5, 5), strides=(2,2), activation='relu'))
		model.add(keras.layers.convolutional.Convolution2D(48, (5, 5), strides=(2,2), activation='relu'))
		model.add(keras.layers.convolutional.Convolution2D(64, (3, 3), activation='relu'))
		model.add(keras.layers.convolutional.Convolution2D(64, (3, 3), activation='relu'))
		model.add(keras.layers.core.Flatten())
		model.add(keras.layers.core.Dense(100, activation='relu'))
		model.add(keras.layers.core.Dropout(0.5))
		model.add(keras.layers.core.Dense(50, activation='relu'))
		model.add(keras.layers.core.Dropout(0.5))
		model.add(keras.layers.core.Dense(10, activation='relu'))
		model.add(keras.layers.core.Dense(1, activation='softmax'))

		print("\nModel Defined")
		self.model = model


	def train(self):
		""" Trains classifier in batches """
		dir_path = os.path.dirname(os.path.realpath(__file__))
		data_ix = pd.read_csv(os.path.join(dir_path,self.train_file_path), names=['center','left','right','steering','throttle','brake','speed'], header=0)

		train_ix, test_ix = train_test_split(data_ix, test_size=0.2)

		train_generator = self.batch_generator(train_ix)
		validation_generator = self.batch_generator(test_ix)

		self.model.compile(loss='mse', optimizer='adam')
		print("\nStarting Training")
		history_object = self.model.fit_generator(train_generator, steps_per_epoch=int(train_ix.size/self.batch_size), validation_data=validation_generator, validation_steps=int(test_ix.size/self.batch_size), epochs=self.epochs)

		print("\nSaving model at {}".format(self.model_name))
		self.model.save(self.model_name)


		### print the keys contained in the history object
		print(history_object.history.keys())
		### plot the training and validation loss for each epoch
		plt.plot(history_object.history['loss'])
		plt.plot(history_object.history['val_loss'])
		plt.title('model mean squared error loss')
		plt.ylabel('mean squared error loss')
		plt.xlabel('epoch')
		plt.legend(['training set', 'validation set'], loc='upper right')
		plt.savefig('figures/loss.png')




if __name__ == '__main__':
	trainer = Trainer()
	trainer.train()