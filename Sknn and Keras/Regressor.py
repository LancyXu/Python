# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:20:41 2016

@author: Lancy(Lanxin) Xu
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sknn.mlp import Regressor, Layer
from sklearn.metrics import mean_squared_error
import random
import datetime
import numpy as np
import pandas as pd

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                 delimiter=";", header =0,
                 names=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'])

data=data.dropna() 

x=data[data.columns[0:11]]
y=data["quality"]

x_MinMax = preprocessing.MinMaxScaler()
y_MinMax = preprocessing.MinMaxScaler()
y = np.array(y).reshape((len(y),1))
x = x_MinMax.fit_transform(x)
y = y_MinMax.fit_transform(y)

np.random.seed(2018)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

f = open('regressor _output.csv','w')
start_time = datetime.datetime.now()
li = ["Rectifier", "Sigmoid", "Tanh", "ExpLin"]
f.write ('2nd Layer Units, 2nd Layer Activator,3rd Layer Units, 3rd Layer Activator, Train MSE , Test MSE \n')
for L2 in range(2, 8):
    for L3 in range(4, 15):
        for activator in li:
            for activator2 in li:
                fit3 = Regressor(
                    layers=[
                        Layer(activator, units=L2),
                        Layer(activator2, units=L3),
                        Layer("Linear")],
                    learning_rate=0.02,
                    random_state =2018,
                    batch_size=100,
                    #dropout_rate=0.3,
                    n_iter=100)
                
                print ("fitting model right now", activator, activator2)
                fit3.fit(x_train,y_train)
                pred3_train=fit3.predict(x_train)
                pred3_test=fit3.predict(x_test)
                error3 = mean_squared_error(pred3_train, y_train)
                error4 = mean_squared_error(pred3_test,y_test)
                outstring= str(L2)+","+ str(activator)+","+str(L3)+ ","+str(activator2)+","+ str(error3)+","+str(error4)+"\n"
                f.write(outstring)
                print(outstring)

stop_time = datetime.datetime.now()
print ("Time required for optimization:",stop_time - start_time)

f.close()
