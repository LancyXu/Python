# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:20:41 2016

@author: Lancy(Lanxin) Xu
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#from __future__ import print_function
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import random
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Embedding, Merge, LSTM
from keras.models import Model
from keras.optimizers import RMSprop,SGD, Adam
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import csr_matrix

np.random.seed(2018)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Function
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def load_data(URL):
    data=pd.read_csv(URL,
                 delimiter=",", header =0,
                 names=['TripType','Weekday','DepartmentDescription','FinelineNumber','ScanCount','Check_problem'])
    return data

def preprocess(data):
    check_problem=data[data.Check_problem==0] #Delete all rows with missing values or with Null values
    drop_check=check_problem.drop('Check_problem',1) #Drop the Check-problem column since all problematic rows were deleted
    return drop_check

def get_NB_CLASSES(data):
    NB_CLASSES=len(pd.Series.unique(data.TripType)) 
    #The number of output classes should be the number of unique classes in response variable
    return NB_CLASSES

def Encode_X(X,area): 
    for i in area: 
        labelencoder_X = LabelEncoder()
        X[:, i] = labelencoder_X.fit_transform(X[:, i])
    return X

def Encode_Y(Y):
    labelencoder_Y = LabelEncoder()
    Y= labelencoder_Y.fit_transform(Y)
    Y=np_utils.to_categorical(Y)
    return Y

def create_dummy(X,area):
    onehotencoder = OneHotEncoder(categorical_features = area)
    X = onehotencoder.fit_transform(X)
    return X

def MaxAbs(X):
    X_MaxAbs = preprocessing.MaxAbsScaler()
    X = X_MaxAbs.fit_transform(X)
    return X

def create_model(optimizer='RMSprop',activation='relu',dropout_rate=0.1,neurons=50):
    model = Sequential()
    model.add(Dense(neurons,input_shape=(x_train.shape[1],)))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def tuning(model,param_grid,x_train,y_train):
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(x_train, y_train,verbose=1)
    return grid_result

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Set Parameters
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
VALIDATION_SPLIT=0.3 
TEST_SIZE=0.3
    
neurons = [50,100,150]
dropout_rate = [0.0, 0.1, 0.2]
activation = ['relu','sigmoid']
optimizer = ['SGD', 'RMSprop','Adam']
batch_size = [10, 50, 100]
epochs = [10, 20, 50]
param_grid = dict(batch_size=batch_size, epochs=epochs, 
                  optimizer=optimizer,activation=activation, 
                  dropout_rate=dropout_rate,neurons=neurons)
    
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Import trainset
data=load_data("D:/myfile/Spring/Artifitial Intelligence/Final/train.csv")
#Preprocess
data=preprocess(data)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x=data[data.columns[1:]].values
y=data.TripType.values
NB_CLASSES=len(pd.Series.unique(data.TripType))
#Encode x
area=[0,1,2] 
x=Encode_X(x,area)
#Encode y
y=Encode_Y(y)

#Create Dummy variables
x=create_dummy(x,area)
x=csr_matrix(x)
    
#Scale
#Using MaxAbs because my x is in a sparse matrix and can not be turned into dense matrix due to memory issue. 
#Therefore, I can not use MinMax scaler.
x=MaxAbs(x)
    
#Data slicing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = TEST_SIZE)    
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Build Model
model=KerasClassifier(build_fn=create_model, verbose=1)

#Tune model
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train,verbose=1)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

#Below shows the best_params
neurons = 150
dropout_rate = 0.1
activation = 'sigmoid'
optimizer = 'Rmsprop'
batch_size = 100
epochs = 50

#Build model
model = Sequential()
model.add(Dense(neurons,input_shape=(x_train.shape[1],)))
model.add(Activation(activation))
model.add(Dropout(dropout_rate))
model.add(Dense(neurons))
model.add(Activation(activation))
model.add(Dropout(dropout_rate))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=epochs, batch_size=batch_size,
                    verbose=1, validation_split=0.3)

print('Testing...')

#Evaluate Model
score = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

stop_time = datetime.datetime.now()
print ("Time required for optimization:",stop_time - start_time)
