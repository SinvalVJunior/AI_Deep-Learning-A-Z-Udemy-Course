#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:39:22 2020

@author: sinval
"""

import preprocessing_data as data
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def create_ann():
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

#Preprocess the data
dataset, X, y, X_train, X_test, y_train, y_test = data.preprocessing_data()
#Build the ann
classifier=KerasClassifier(build_fn=create_ann,batch_size=10,epochs=5)
#Get the accuracies
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
