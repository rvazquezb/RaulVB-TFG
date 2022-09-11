from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
import os

csv = pd.read_csv('semana_final_definitiva.csv',encoding='latin1')
csv2 = pd.read_csv('paquetes.csv',encoding='latin1')

le = preprocessing.LabelEncoder()
le.fit(csv["Label"].unique())
etiquetas = le.transform(csv["Label"])

x = csv.drop(['Label'],axis=1)
y = pd.DataFrame(etiquetas, columns = ['Target'])


flow_ids = csv2['flow_id'].unique()
flow_ids_lista = flow_ids.tolist()
count = 1
cv_externo = StratifiedKFold(n_splits=3, shuffle=True)
for train_index, test_index in cv_externo.split(X,y):
	X_train,X_test = X.iloc[train_index,:], X.iloc[test_index,:]
	y_train,y_test = y.iloc[train_index], y.iloc[test_index]
	
	model = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=None, max_features='sqrt', n_jobs=-1)
	#model = LogisticRegression(random_state=1, C=100, tol=0.001, solver='saga', n_jobs=-1)
	#model = AdaBoostClassifier(random_state=1, n_estimators=100, learning_rate=1.0)

	model.fit(X_train.drop(columns=['Flow ID'],axis=1),np.ravel(y_train))


	xtest = X_test[X_test['Flow ID'].isin(flow_ids_lista)]

	y_pred = model.predict(xtest.drop(columns=['Flow ID'],axis=1))

	flow_id = []
	prediction = []
	y_pos = 0
	for x in xtest.index:
		if X['Flow ID'][x] not in flow_id:	
			flow_id.append(X['Flow ID'][x])
			prediction.append(y_pred[y_pos])
		y_pos = y_pos + 1

	df = pd.DataFrame({
		'flow_id' : flow_id,
		'prediction' : prediction,
	})

	csv3 = csv2.merge(df, how='inner', on='flow_id')

	name = 'RF_modelo' + str(count) + '.tsv'
	#name = 'LR_modelo' + str(count) + '.tsv'
	#name = 'Ada_modelo' + str(count) + '.tsv'

	csv3.to_csv(name,sep="\t",index=False)

	os.system('python3 evaluator_chunks_RaulTFG_v2.py -gpath ./golden_truth.tsv -ppath ./' + name + ' -opath ./salida_evaluacion/RF' + str(count) + ' -o 8 -p 0.15694 -n_decision 5')
	count = count + 1


