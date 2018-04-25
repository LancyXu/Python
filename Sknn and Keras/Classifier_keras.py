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
from sknn.mlp import Classifier, Layer
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
import datetime
start_time = datetime.datetime.now()
import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


# network and training
NB_EPOCH = 250
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 11   #Since the quality score ranges from 0 to 10, the number of classes outputted should be 11
OPTIMIZER = RMSprop() 
N_HIDDEN = 128
VALIDATION_SPLIT=0.3 
DROPOUT = 0.3 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                 delimiter=";", header =0,
                 names=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'])

data=data.dropna() 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x=data[data.columns[0:11]]
y=data["quality"]


x_MinMax = preprocessing.MinMaxScaler()

y = np.array(y).reshape((len(y),1))
x = x_MinMax.fit_transform(x)


np.random.seed(2018)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 11 outputs
# final stage is softmax

model = Sequential()
model.add(Dense(N_HIDDEN,input_shape=(11,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print('Test accuracy:', score[1])
print('Test error:', 1-score[1])

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
