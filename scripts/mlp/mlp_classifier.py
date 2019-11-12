import warnings
warnings.filterwarnings('ignore')

import numpy as np
seed = 7
np.random.seed(seed)
import pandas as pd

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam

from sklearn.utils import class_weight

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

# calculate class weights for highly imbalanced datasets
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weight_dict = dict(enumerate(class_weights))

print('class weights: %s' % class_weight_dict)

# create a multi-layer perceptron
model = Sequential()

model = Sequential([
        Dense(2000, input_dim=204, kernel_initializer='uniform', activation='relu'),
        Dense(2000, kernel_initializer='uniform', activation='relu'),
        Dense(1000, kernel_initializer='uniform', activation='relu'),
        Dense(700, kernel_initializer='uniform', activation='relu'),
        Dense(1, activation='sigmoid')])

model.compile(loss="binary_crossentropy", optimizer=Adam(0.0001))
model.fit(X_train, y_train, epochs=5, batch_size=64)
score = model.evaluate(X_test, y_test, batch_size=64)

print("Results: %.2f (%.2f) MSE" % (score.mean(), score.std()))

# get predictions for test data
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