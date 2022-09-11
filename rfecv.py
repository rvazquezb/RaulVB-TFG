import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import sklearn
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import time
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

csv = pd.read_csv('semana_final_tras_isolation.csv')

labels = []

for x in csv.index:
        if csv['Label'][x] != 'BENIGN':
                labels.append('ATTACK')
        else:
                labels.append('BENIGN')

csv['Label'] = labels
le = preprocessing.LabelEncoder()
le.fit(csv["Label"].unique())
etiquetas = le.transform(csv["Label"])

csv.drop(columns=['Flow ID'],axis=1,inplace=True)

estimator = RanfomForestClassifier(n_estimators=100, n_jobs=-1)
X = csv.drop(['Label'],axis=1)
y = etiquetas

 
rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(n_splits=5,shuffle=True), verbose=1, scoring='f1')
rfecv.fit(X,y)

print('------------ Results: ----------------')
print('>>>> Optimal number of features : %d' % rfecv.n_features_)
print('>>>> grid scores:')
print(rfecv.cv_results_)

print('************************************************************************************')
for i in range(X.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfecv.support_[i], rfecv.ranking_[i]))


print('************************************************************************************')

