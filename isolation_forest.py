import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
import sys
import seaborn as sns


csv = pd.read_csv('semana_final_tras_correlacion.csv',encoding='latin1')
csv.drop(columns=['Label'],axis=1,inplace=True)
csv.drop(columns=['Flow ID'],axis=1,inplace=True)

model=IsolationForest(n_estimators=100,max_features=1,n_jobs=-1,contamination=float(0.001),random_state=1)

model.fit(csv)

scores = model.decision_function(csv)
anomaly_score = model.predict(csv)
csv['scores'] = scores
csv['anomaly_score'] = anomaly_score
