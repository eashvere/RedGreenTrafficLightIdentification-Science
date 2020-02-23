# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
import sys
from PIL import Image
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
import os
import pandas as pd
from scipy.misc import imread
import numpy as np

def load_image( infilename ) :
    #Method to help load pictures
    img = Image.open( infilename )
    img = img.resize((256,256), Image.ANTIALIAS)
    img.load()
    data = np.asarray( img, dtype="float32" )
    img = None
    return data

#Loading the Data
root_dir = os.path.abspath("/Users/eashver/Desktop/Python/nexar_traffic_light_train/")
train_file = os.path.join(root_dir, "labels.csv")
print("Loading Data...")
train = pd.read_csv(train_file)
batch_size = 50

print("Loaded Data")
    #Convert CSV file to numpy arrays
print("Reading Data...")
temp = []
temp2 = []
files = 0
#Iterate through the CSV file
for index, row in train.iterrows():
    image_path = os.path.join(root_dir, row['FileName'])
    #Append Information
    temp.append(load_image(image_path))
    temp2.append(row['Label'])
    files += 1
    sys.stdout.write('\rCurrent File: %d/18659' % files)
    sys.stdout.flush()
print('\n')
print("Done Reading Data")
print("Creating numpy array...")

#Converting to Numpy Arrays
X = np.array(temp)
Y = np.array(temp2)
temp = None
temp2 = None

print("Creating Validation Sets...")
split_size = int(X.shape[0]*0.9)
X, X_test = X[:split_size], X[split_size:]
Y, Y_test = Y[:split_size], Y[split_size:]

print(X.shape)
print(Y.shape)


#Reshaping the Numpy Arrays
X = X.reshape([-1, 256, 256, 3])
X_test = X_test.reshape([-1, 256, 256, 3])
Y = to_categorical(Y, 3)
Y_test = to_categorical(Y_test, 3)

print("Starting Data Manipulation...")
# Shuffle the data
print("Shuffling the Data...")
# Shuffle the data
X, Y = shuffle(X, Y)

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 256, 256, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
# Step 1: Convolution
network = conv_2d(network, 12, 5, strides = 1, activation='relu') #Network, number of filters, filter size sidelength

# Step 2: Max pooling
network = max_pool_2d(network, 3)

# Step 3: Convolution again
network = conv_2d(network, 48, 4, strides = 2, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 96, 4, strides = 2, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 3)
# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Another Fully-connected 2048 node neural network to add more neurons
network = fully_connected(network, 2048, activation='relu')

# Step 9: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 10: Fully-connected neural network with three outputs to make the final prediction
network = fully_connected(network, 3, activation='softmax')
print("Done Creating Neural Network")
# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001, name = 'target')

print("Wrapping Currently in progress...")
# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=2, checkpoint_path='traffic-light.tfl.ckpt')

print("Training is beginning...")
runtimes = 20
print("Running " + str(runtimes) + " times")
# Train it! We'll do 20 training passes with a batch size of 50 and monitor it as it goes.
model.fit(X, Y, n_epoch=runtimes, shuffle=True,
                validation_set=(X_test, Y_test),show_metric=True,
                batch_size = batch_size, snapshot_epoch=True,run_id='Traffic_Light Classifier')

print("Saving Model...")
# Save model when training is complete to a file
model.save("traffic-light.tfl")
print("Network trained and saved as traffic-light.tfl!")
