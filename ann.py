# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


###############################################################################
#
#   IMPORT DATASET & PREPROCESSING
#
###############################################################################

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical data
# There are 2 pieces of categorical data: gender (in column 3) and country (in 
# colum 2).
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Country = LabelEncoder()
X[:, 1] = labelencoder_X_Country.fit_transform(X[:, 1])
labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# We want 80% of the data to be used for training. We'll test on 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scale features for normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


###############################################################################
#
#   BUILD ANN ARCHITECTURE
#
###############################################################################

# Import Keras and required packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import PReLU

"""
    The architecture:
        - Input layer with 11 features
        - First Hidden Layer with 6 nodes & RELU activation
        - Dropout with 10%
        - Second Hidden Layer with 6 nodes & RELU activation
        - Dropout with 10%
        - Output layer with sigmoid activation
        - OPTIMIZER: adam
        - LOSS FN: binary cross entropy
        - BATCH SIZE: 10
        - EPOCHS: 100
"""

# Initializing the ANN (i.e., defining it as a sequence of layers)
classifier = Sequential()

# Adding the input layer and first hidden layer
# Also add dropout to prevent overfitting
classifier.add(Dense(input_dim = 11, units = 6, kernel_initializer = 'uniform'))

prelu_activation = PReLU()
classifier.add(prelu_activation)

classifier.add(Dropout(rate = 0.1))

# Add the second hidden layer
# Also add dropout to prevent overfitting
classifier.add(Dense(units = 6, kernel_initializer = 'uniform'))

prelu_activation = PReLU()
classifier.add(prelu_activation)

classifier.add(Dropout(rate = 0.1))

# Add output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# FIT
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


###############################################################################
#
#   EVALUATION
#
###############################################################################

# Use the test set to evaluate ANN
# Set a threshold for predicting true (threshold = 0.5)
# Could be interesting to create an ROC curve
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# To evaulate the performance, create the confusion matrix from the thresholded
# test set
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


###############################################################################
#
#   k-FOLD VALIDATION
#
###############################################################################

# Evaluate ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# Recreate the classifier above in a function (necessary for the k-Fold validation)
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# We'll use a fold value of 10 and compoute the epochs in parallel for speed
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

# Compute mean and variance
mean = accuracies.mean() # 0.838124995045364
variance = accuracies.std() # 0.01369591544384601


###############################################################################
#
#   HYPERPARAMETER TUNING
#
###############################################################################

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Copy and pasted from above, with one difference
# We want to tune the optimizer, so we'll pass in the argument as a parameter
# to the build function
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# Declare a dictionary of parameters that we want to optimize.
# We think these will effect the performance the most, batch_size, # of epochs,
# and the optimizer.
parameters_to_optimize = {'batch_size': [25, 32], 
                          'epochs': [100, 500],
                          'optimizer': ['adam', 'rmsprop']}

# Create grid search object to handle the creation and testing
# We want a k-fold of 10 and to evaluate each iteration on its accuracy
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters_to_optimize, 
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X = X_train, y = y_train)

# Examine the optimal parameters
best_parameters = grid_search.best_params_
best_accuracies = grid_search.best_score_


