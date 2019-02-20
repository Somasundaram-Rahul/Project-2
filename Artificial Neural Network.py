# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#first stage: data reading/collection 
df = pd.read_excel("default of credit card clients.xls",header=[0,1],index_col=0 )

#structure the data
X = df.iloc[:,0:23]
y = df.iloc[:,23]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)

# data preprocessing
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#keras neural networks
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(40,activation="sigmoid",input_dim=X.shape[1]))
model.add(tf.keras.layers.Dense(20,activation="sigmoid"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam')

print (model.summary())

model.fit(
    X_train,
    y_train,
    epochs = 10,
    batch_size=16)
    
probas = model.predict(X_test) 
y_pred = np.c_[1-probas,probas]


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


#Area calculation
score = y_pred[:,1] 
pos = pd.get_dummies(y_test).as_matrix()
pos = pos[:,1]
npos = np.sum(pos)
index = np.argsort(score)
index = index[::-1] 
sort_pos = pos[index] 
cpos = np.cumsum(sort_pos) 
gain = cpos/npos
n = y_test.shape[0]
abscissa = np.arange(start=1,stop=n+1,step=1) 
abscissa = abscissa / n 
a=metrics.auc(abscissa, gain)-metrics.auc(abscissa, abscissa)
b=metrics.auc(abscissa, best)-metrics.auc(abscissa, abscissa)
print ("The area ratio is=",a/b)

