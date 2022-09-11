import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import sys

np.set_printoptions(threshold=sys.maxsize)

csv = pd.read_csv('semana_final.csv',encoding='latin1')

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

csv.drop(columns=['Label'],axis=1,inplace=True)
csv.drop(columns=['Flow ID'],axis=1,inplace=True)

x = csv.values.tolist()
y = etiquetas

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


 # Create a random forest classifier
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
print(np.mean(cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=3,shuffle=True))))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f1_score(y_test, y_pred))


mydict=dict()

for feature in zip(csv.columns.values, model.feature_importances_):
	mydict[feature[0]] = feature[1]

a = sorted(mydict.items(), key=lambda x: x[1], reverse=True)
print(a)