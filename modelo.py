import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline

param_grids = [
{
	'estimator': [LogisticRegression(random_state=1)],
	'estimator__solver': ['lbfgs','saga'],
	'estimator__C': [1,10,100],
	'estimator__tol': [0.01, 0.001, 0.0001]
},{
	'estimator': [RandomForestClassifier(random_state=1)],
	'estimator__n_estimators': [100,200,300],
	'estimator__max_depth': [3,5,7,None]
},{
	'estimator': [AdaBoostClassifier(random_state=1)],
	'estimator__n_estimators': [10,50,100,500],
	'estimator__learning_rate': [0.5,1.0] 
}]

pipes = Pipeline(steps=[('estimator', LogisticRegression(random_state=1))])

csv = pd.read_csv('semana_final_definitiva.csv',encoding='latin1')

le = preprocessing.LabelEncoder()
le.fit(csv["Label"].unique())
etiquetas = le.transform(csv["Label"])

csv.drop(columns=['Flow ID'],axis=1,inplace=True)

X = csv.drop(['Label'],axis=1)
y = pd.DataFrame(etiquetas, columns = ['Label'])


cv_externo = StratifiedKFold(n_splits=5, shuffle=True)
count=1
for train_index, test_index in cv_externo.split(X,y):
	X_train,X_test = X.iloc[train_index,:], X.iloc[test_index,:]
	y_train,y_test = y.iloc[train_index], y.iloc[test_index]

	grid = GridSearchCV(pipes,param_grid=param_grids, refit=True, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1), n_jobs=-1, scoring='f1', verbose=4)
	grid.fit(X_train,np.ravel(y_train))

	print('Best cross-validation accuraccy: {:.12f}'.format(grid.best_score_))
	print('Test set score: {:.12f}'.format(grid.score(X_test,y_test)))
	print('Best parameters: {}'.format(grid.best_params_))

	df = pd.DataFrame(grid.cv_results_)

	df.to_csv('cv_results' + str(count) + '.csv',index=False)
	count = count + 1