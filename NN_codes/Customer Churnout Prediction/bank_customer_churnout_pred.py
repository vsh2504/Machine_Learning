# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:27:55 2020

@author: vinayak
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y =  dataset.iloc[:,13].values

# Encoding categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# After this Female?Male shifts to 5Th column so we need to change below code index
# Male/Female
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

# Out of 3 dummy vars for countries we need to remove one as it is dependant on other two
X = X[:,1:]

#Splitting the dataset into the Training set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")) #new syntax keras 2 api

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu')) #old_syntax

#Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100) 

#old_syntax --> new_syntax
'''
output_dim --> units
init -->  kernel_initializer
nb_epoch -->  epochs
Convolution2D(32, 3, 3, …) -->  Conv2D(32, (3, 3), …)
samples_per_epoch -->  steps_per_epoch
nb_val_samples -->  validation_steps
'''

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Input always given as 2D array to predict
new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)