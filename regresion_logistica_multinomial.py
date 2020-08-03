# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:12:55 2020

@author: Ken
"""


import cv2 
import numpy as np
from collections import Counter 
import itertools 
import glob
import re
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pandas as pd


path = glob.glob("*.jpg")
cv_img = []
vec=[]
for img in path:
   
    cv_img.append(img)
    x=int(re.findall("\d+",img)[0])
    vec.append(x)

vec=np.array(vec)


x=np.load('matriz_histograma.npy')

anto=np.load('vectorAnthony.npy')
ken=np.load('ken.npy')

pru=np.zeros((1,len(x[0])))

for i in range(len(x[0])):
    pru[0][i]=x[0][i]

y=vec

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.20, random_state=100)
model = LogisticRegression()
model.fit(X_train, Y_train)


result = model.score(X_test, Y_test)
print("Accuracy: %.2f%%" % (result*100.0))
print(model.fit(X_train, Y_train))
kfold = model_selection.KFold(n_splits=5, random_state=150)
model_kfold = LogisticRegression()
results_kfold = model_selection.cross_val_score(model_kfold, x, y, cv=kfold)
model_kfold.fit(X_train, Y_train)
prediciones = model_kfold.predict(X_test)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))




prediciones_cinco = model_kfold.predict(ken)









