import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# load training and test datasets
dataset = pd.read_csv('training.csv', delimiter=',')
X_train = dataset.iloc[:,3:]
y_train = dataset.iloc[:,2:3]
y_train = np.array(y_train).ravel()

print("loaded training data: %s, %s" % (X_train.shape, y_train.shape))

dataset = pd.read_csv('test.csv', delimiter=',')
X_test = dataset.iloc[:,3:]
y_test = dataset.iloc[:,2:3]
y_test = np.array(y_test).ravel()

print("loaded test data: %s, %s" % (X_test.shape, y_test.shape))

# fit model on training data
model = XGBClassifier(objective="binary:logistic", random_state=42)
model.fit(X_train, y_train)

# make predictions for test data
predictions = model.predict(X_test)
y_pred = np.round(predictions,0)

# calculate performance metrics
import sys
sys.path.append("../utils/")
import evaluation_metrics as ev

auc = ev.auc_roc(y_test, predictions)
ba = ev.balanced_accuracy(y_test, y_pred)
sens, spec = ev.sensitivity_specificity(y_test, y_pred)
kappa = ev.kappa_score(y_test, y_pred)

print('\nModel Performance')
print('AUC:\t%s' % auc)
print('BACC:\t%s' % ba)
print('Sens:\t%s' % sens)
print('Spec:\t%s' % spec)
print('Kappa:\t%s' % kappa)