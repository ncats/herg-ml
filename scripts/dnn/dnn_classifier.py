import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.utils import class_weight
import pickle

seed = 7
np.random.seed(seed)

# load training and test datasets
dataset = pd.read_csv('../../data/train_valid/descriptors/training_set_desc.csv', delimiter=',')
X_train = dataset.iloc[:,121:1145] # rdkit - 2:121; morganfp - 121:1145; latent1 - 1145:1657; latent2 - 1657:
y_train = dataset.iloc[:,1:2].values
y_train = np.array(y_train).ravel()

print("loaded training data: %s, %s" % (X_train.shape, y_train.shape))

dataset = pd.read_csv('../../data/train_valid/descriptors/validation_set_desc.csv', delimiter=',')
X_test = dataset.iloc[:,121:1145] # rdkit - 2:121; morganfp - 121:1145; latent1 - 1145:1657; latent2 - 1657:
y_test = dataset.iloc[:,1:2].values
y_test = np.array(y_test).ravel()

print("loaded test data: %s, %s" % (X_test.shape, y_test.shape))

# calculate class weights for highly imbalanced datasets
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weight_dict = dict(enumerate(class_weights))

print('class weights: %s' % class_weight_dict)

# create a multi-layer perceptron
model = Sequential()

model = Sequential([
        Dense(2000, input_dim=1024, kernel_initializer='uniform', activation='relu'),
        Dense(2000, kernel_initializer='uniform', activation='relu'),
        Dense(1000, kernel_initializer='uniform', activation='relu'),
        Dense(700, kernel_initializer='uniform', activation='relu'),
        Dense(1, kernel_initializer='uniform', activation='sigmoid')])

model.compile(loss="binary_crossentropy", optimizer=Adam(0.00001))
model.fit(X_train, y_train, epochs=30, batch_size=128, class_weight=class_weight_dict)
score = model.evaluate(X_test, y_test, batch_size=128)
#print("Results: %.2f (%.2f) MSE" % (score.mean(), score.std()))

'''
# save the model file
output = open('dnn_classifier.pkl', 'wb')
pickle.dump(model, output)
output.close()

'''

# get predictions for test data
predictions = model.predict(X_test)
y_pred = np.round(predictions, 0)
predictions = np.array(predictions).ravel()
predictions = np.round(predictions, 2)

# calculate performance metrics
import sys
sys.path.append("../utils/")
import evaluation_metrics as ev

auc = ev.auc_roc(y_test, predictions)
ba = ev.balanced_accuracy(y_test, y_pred)
sens, spec = ev.sensitivity_specificity(y_test, y_pred)
kappa = ev.kappa_score(y_test, y_pred)

print('model performance')
print('AUC:\t%s' % auc)
print('BACC:\t%s' % ba)
print('Sens:\t%s' % sens)
print('Spec:\t%s' % spec)
print('Kappa:\t%s' % kappa)