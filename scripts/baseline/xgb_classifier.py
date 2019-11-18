import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# path to training data file
train_file = 'training.csv'

# path to test data file
test_file = 'test.csv'

sys.path.append("../utils/")
import dataset_splitter
import evaluation_metrics as ev

train_dfs, test_dfs = dataset_splitter.split_dataset(train_file, 0.2, 5)
print("loaded and splitted training data")

# 5-fold cv
print('building cross-validation models')
auc = []
ba = []
sens = []
spec = []
kappa = []

for k in range(5):

	train = train_dfs[k]
	test = test_dfs[k]

	X_train = train.iloc[:,3:]
	y_train = train.iloc[:,2:3]
	y_train = np.array(y_train).ravel()

	model = XGBClassifier(objective="binary:logistic", random_state=42)
	model.fit(X_train, y_train)

	X_test = test.iloc[:,3:]
	y_test = test.iloc[:,2:3]
	y_test = np.array(y_test).ravel()

	predictions = model.predict(X_test)
	y_pred = np.round(predictions,0)

	auc.append(ev.auc_roc(y_test, predictions))
	ba.append(ev.balanced_accuracy(y_test, y_pred))
	se, sp = ev.sensitivity_specificity(y_test, y_pred)
	sens.append(se)
	spec.append(sp)
	kappa.append(ev.kappa_score(y_test, y_pred))


print('cross-validation performance')
print('AUC:\t%.2f +/- %.2f' % (np.array(auc).mean(),np.array(auc).std()))
print('BACC:\t%.2f +/- %.2f' % (np.array(ba).mean(),np.array(ba).std()))
print('Sens:\t%.2f +/- %.2f' % (np.array(sens).mean(),np.array(sens).std()))
print('Spec:\t%.2f +/- %.2f' % (np.array(spec).mean(),np.array(spec).std()))
print('Kappa:\t%.2f +/- %.2f' % (np.array(kappa).mean(),np.array(kappa).std()))

# external validation

# load training and test datasets
dataset = pd.read_csv('training.csv', delimiter=',')
X_train = dataset.iloc[:,3:]
y_train = dataset.iloc[:,2:3]
y_train = np.array(y_train).ravel()

#print("loaded training data: %s, %s" % (X_train.shape, y_train.shape))

dataset = pd.read_csv('test.csv', delimiter=',')
X_test = dataset.iloc[:,3:]
y_test = dataset.iloc[:,2:3]
y_test = np.array(y_test).ravel()

print('loaded test data')
print('building final model')
#print("loaded test data: %s, %s" % (X_test.shape, y_test.shape))

# fit model on training data
model = XGBClassifier(objective="binary:logistic", random_state=42)
model.fit(X_train, y_train)

# make predictions for test data
predictions = model.predict(X_test)
y_pred = np.round(predictions,0)

# calculate performance metrics

auc = ev.auc_roc(y_test, predictions)
ba = ev.balanced_accuracy(y_test, y_pred)
sens, spec = ev.sensitivity_specificity(y_test, y_pred)
kappa = ev.kappa_score(y_test, y_pred)

print('external validation performance')

print('AUC:\t%s' % auc)
print('BACC:\t%s' % ba)
print('Sens:\t%s' % sens)
print('Spec:\t%s' % spec)
print('Kappa:\t%s' % kappa)