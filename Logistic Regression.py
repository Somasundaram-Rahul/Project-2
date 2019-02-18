#risk evaluation through credit card default analysis

import tensorflow as tf
import numpy as np
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#we set the seed

np.random.seed(1)
import random
random.seed(1)

#first stage: data reading/collection

df = pd.read_excel("default of credit card clients.xls", header=1, skiprows=0,
index_col=0)

df.rename(index=str, columns={"default payment next month": 
"defaultPaymentNextMonth"}, inplace=True)

#structure the data

X = df.iloc[:,  df.columns != 'defaultPaymentNextMonth']
y = df.iloc[:,  df.columns == 'defaultPaymentNextMonth']

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2)

#data preprocessing: Standardization of a dataset is a 
#common requirement for many machine learning estimators: 
#they might behave badly if the individual features do not
#more or less look like standard normally distributed data 
#(e.g. Gaussian with 0 mean and unit variance).

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#sci-kit learn logistic regression

lr = LogisticRegression(solver = "lbfgs", max_iter = 500)
model = lr.fit(X_train, np.array(y_train).ravel()) 

y_pred = model.predict(X_test)

#extracting and sorting (in decreasing order) the default 
#probabilities for different points in the explanatory
#variable space

prob_inst = model.predict_proba(X_test)

y_testo = np.array(y_test).ravel()

inds = np.argsort(prob_inst[:,0])
prob_sort = prob_inst[inds]
y_testo_sort = y_testo[inds]
   
#computation and plotting of the cumulative gain chart

skplt.metrics.plot_cumulative_gain(y_testo, prob_inst)
defaults = sum(y_testo == 1)
total = len(y_testo)
defaultRate = defaults/total
def bestCurve(defaults, total, defaultRate):
    x = np.linspace(0, 1, total)
    
    y1 = np.linspace(0, 1, defaults)
    y2 = np.ones(total-defaults)
    y3 = np.concatenate([y1,y2])
    return x, y3

x, best = bestCurve(defaults=defaults, total=total,
defaultRate=defaultRate)    
plt.plot(x, best) 
plt.show()

#area ratio calculation

npos = np.sum(y_testo)

#cumulated sum

cpos = np.cumsum(y_testo_sort)

#recall column

logclas = cpos/npos # logistic classifier

abscissa = np.arange(start=1,stop=total+1,step=1)
abscissa = abscissa/total

#using the area under curve sci-kit learn feature to extract area ratio

a=metrics.auc(abscissa, logclas)-metrics.auc(abscissa, abscissa)
b=metrics.auc(abscissa, best)-metrics.auc(abscissa, abscissa)
print ("The area ratio is",a/b)