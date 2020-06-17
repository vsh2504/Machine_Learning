# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:16:18 2020

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

#old_syntax --> new_syntax
'''
output_dim --> units
init -->  kernel_initializer
nb_epoch -->  epochs
Convolution2D(32, 3, 3, …) -->  Conv2D(32, (3, 3), …)
samples_per_epoch -->  steps_per_epoch
nb_val_samples -->  validation_steps
'''


#Evaluating the ann
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim = 11, units = 6,kernel_initializer = 'uniform' ))
    classifier.add(Dense(activation='relu',units= 6,kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn= build_classifier)
parameters = {
    'batch_size': [25,32],
    'epochs': [100,500],
    'optimizer': ['adam','rmsprop']
}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_