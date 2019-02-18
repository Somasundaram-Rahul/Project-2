# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
import scikitplot as skplt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from sklearn.svm import SVC
np.random.seed(0)
import random
random.seed(0)

#first stage: data reading/collection
df = pd.read_excel("default of credit card clients.xls",header=[0,1],index_col=0 )

#structure the data
X = df.iloc[:,0:23]
y = df.iloc[:,23]

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=1)

# data preprocessing
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#sci-kit learn support vector machine
clf = SVC(gamma='auto',probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

#computation and plotting of the cumulative gain chart
skplt.metrics.plot_cumulative_gain(y_test, y_pred)
defaults = sum(y_test)
total = len(y_test)
defaultRate = defaults/total
def bestCurve(defaults, total, defaultRate):
    x = np.linspace(0, 1, total)
    
    y1 = np.linspace(0, 1, defaults)
    y2 = np.ones(total-defaults)
    y3 = np.concatenate([y1,y2])
    return x, y3

x, best = bestCurve(defaults=defaults, total=total, defaultRate=defaultRate)    
plt.plot(x, best)  
plt.show()




