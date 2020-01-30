#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:25:07 2020

@author: sinval
"""
import preprocessing_data as data
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def create_ann(optimizer):
    #Initialising the ann
    classifier = Sequential()
    #Creating the input layer and the first hidden layer
    classifier.add(Dense(6, input_shape=(11,), activation="relu", kernel_initializer="uniform"))
    #Creating another hidden layer
    classifier.add(Dense(6, activation="relu", kernel_initializer="uniform"))
    #Creating the output layer
    classifier.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    #Compile de ann
    classifier.compile(optimizer=optimizer, loss = "binary_crossentropy", metrics = ["accuracy"] )
    return classifier

#Preprocess the data
dataset, X, y, X_train, X_test, y_train, y_test = data.preprocessing_data()
#Build the ann
classifier=KerasClassifier(build_fn=create_ann)
#Make de dictionary
parameters = {'batch_size' : [25,20],
              'epochs' : [10,15],
              'optimizer' : ['adam','rmsprop'],
              }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

