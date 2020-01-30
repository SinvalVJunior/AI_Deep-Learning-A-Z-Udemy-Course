#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:29:03 2020

@author: sinval
"""
import preprocessing_data as data
import ann as ann
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#Preprocess the data
dataset, X, y, X_train, X_test, y_train, y_test = data.preprocessing_data()
#Build the ann
classifier = ann.creat_ann()
#Training the ann
classifier.fit(X_train,y_train,batch_size = 10, nb_epoch = 10)
#Making the prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
#Reading the outputs
cm = confusion_matrix(y_test,y_pred)
#Homework: Making a single prediction
sc = StandardScaler()
sc.fit_transform(X_train)
costumer = np.matrix('0 0 600 1 40 3 60000 2 1 1 50000')
costumer = sc.transform(costumer)
costumer_predict = classifier.predict(costumer)
costumer_predict = (costumer_predict > 0.5)


