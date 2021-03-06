#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:29:03 2020

@author: sinval
"""    

def create_ann():
    #Import Keras
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    #Initialising the ann
    classifier = Sequential()
    #Creating the input layer and the first hidden layer
    classifier.add(Dense(6, input_shape=(11,), activation="relu", kernel_initializer="uniform"))
    #Creating another hidden layer
    classifier.add(Dense(6, activation="relu", kernel_initializer="uniform"))
    #Creating the output layer
    classifier.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    #Compile de ann
    classifier.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"] )
    return classifier
