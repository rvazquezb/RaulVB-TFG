import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import sklearn
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import time
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold

start_time = time.monotonic()

csv = pd.read_csv('semana_final_definitiva.csv')

le = preprocessing.LabelEncoder()
le.fit(csv["Label"].unique())
etiqueta = le.transform(csv["Label"])

X = csv.drop(['Label'],axis=1)
y = etiqueta

    
rfecv = RFECV(estimator=DecisionTreeClassifier(max_depth=3), step=1, cv=StratifiedKFold(n_splits=5,shuffle=True), verbose=1, scoring='accuracy')
rfecv.fit(X,y)

print('------------ Results: ----------------')
print('>>>> Optimal number of features : %d' % rfecv.n_features_)
print('>>>> grid scores:')
print(rfecv.grid_scores_)

print('************************************************************************************')
for i in range(X.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, rfecv.support_[i], rfecv.ranking_[i]))

end_time = time.monotonic()
print('************************************************************************************')
print(timedelta(seconds=end_time - start_time))


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
