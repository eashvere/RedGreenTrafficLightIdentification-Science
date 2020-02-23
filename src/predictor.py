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

#Path to the model
modelfile = "/Users/eash/Documents/Results/No_Nothing_lights/traffic-light.tfl"

DevMode = False

#Method to load the picture
def load_image( infilename ) :
    img = Image.open( infilename )
    img = img.resize((256,256), Image.ANTIALIAS)
    img.load()
    data = np.asarray( img, dtype="float32" )
    img = None
    data = data.reshape([-1, 256, 256, 3])
    return data

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 256, 256, 3])
# Step 1: Convolution
network = conv_2d(network, 12, 5, strides = 1, activation='relu') #Network, number of filters, filter size sidelength

# Step 2: Max pooling
network = max_pool_2d(network, 3)

# Step 3: Convolution again
network = conv_2d(network, 48, 4, strides = 2, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 3)

# Step 4: Convolution yet again
network = conv_2d(network, 96, 4, strides = 2, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 3)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 6: Fully-connected 1024 node neural network
network = fully_connected(network, 2048, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with three outputs to make the final prediction
network = fully_connected(network, 3, activation='softmax')
print("Done Creating Neural Network")
# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001, name = 'target')

print("Wrapping Currently in progress...")
# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path='traffic-light.tfl.ckpt')

model.load(modelfile)

#Code to predict
yesDevmode = raw_input("Should Dev Mode be on (yes/no)")
if yesDevmode.lower() == "yes" or yesDevmode.lower() == 'y':
    yesDevmode = True

while(True):
    #User Interface to input picture and get a prediction back
    print("\n")
    picture = raw_input("What picture do you want to use: ")
    if picture == "exit" or picture == "":
        sys.exit("Shutting Traffic Classifer down...")
    picturefile = "/Users/eash/Documents/Demo/" + picture + ".jpg"
    pic = load_image(picturefile)

    pred = model.predict(pic)

    for list in pred:
        light = list.index(max(list))

    print("\n")

    if light == 0:
        print("No Traffic Lights")
    elif light == 1:
        print("There is a Red Light ahead")
        os.system("say 'There is a Red Light ahead'")
    elif light == 2:
        print("There is a Green Light Ahead")
        os.system("say 'There is a Green Light ahead'")
    if yesDevmode:
        print("Predictions: ", pred)
