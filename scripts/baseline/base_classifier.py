import os, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# path to training data file
train_file = '../../data/train_valid/descriptors/training_set_desc.csv'
print('Training File: %s' % os.path.basename(train_file))

# path to test data file
test_file = '../../data/train_valid/descriptors/validation_set_desc.csv'
print('Test File: %s' % os.path.basename(test_file))

# load training and test datasets
dataset = pd.read_csv(train_file, delimiter=',')
X_train = dataset.iloc[:,2:121] # rdkit - 2:121; morganfp - 121:1145; latent1 - 1145:1657; latent2 - 1657:
y_train = dataset.iloc[:,1:2]
y_train = np.array(y_train).ravel()

print("loaded training data: %s, %s" % (X_train.shape, y_train.shape))

dataset = pd.read_csv(test_file, delimiter=',')
X_test = dataset.iloc[:,2:121] # rdkit - 2:121; morganfp - 121:1145; latent1 - 1145:1657; latent2 - 1657:
y_test = dataset.iloc[:,1:2]
y_test = np.array(y_test).ravel()

#print('building final model')
print("loaded test data: %s, %s" % (X_test.shape, y_test.shape))

# fit model on training data
model = RandomForestClassifier(n_estimators = 300, random_state = 42)
#model = XGBClassifier(n_estimators = 300, random_state = 42, objective='binary:logistic', learning_rate=0.05)
model.fit(X_train, y_train)


'''
# save the model file
output = open('xgboost_classifier.pkl', 'wb')
pickle.dump(model, output)
output.close()

'''

# get predictions for test data
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test).T[1]
y_pred_prob = y_pred_prob.ravel()
y_pred_prob = np.round(y_pred_prob, 2)

# calculate performance metrics
import sys
sys.path.append("../utils/")
import evaluation_metrics as ev

auc = ev.auc_roc(y_test, y_pred_prob)
ba = ev.balanced_accuracy(y_test, y_pred)
sens, spec = ev.sensitivity_specificity(y_test, y_pred)
kappa = ev.kappa_score(y_test, y_pred)

print('model performance')
print('AUC:\t%s' % auc)
print('BACC:\t%s' % ba)
print('Sens:\t%s' % sens)
print('Spec:\t%s' % spec)
print('Kappa:\t%s' % kappa)