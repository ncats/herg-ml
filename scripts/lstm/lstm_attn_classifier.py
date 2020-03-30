import warnings
warnings.filterwarnings('ignore')

import numpy as np
seed = 7
np.random.seed(seed)
import pandas as pd

import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Embedding, Conv1D, MaxPooling1D, GRU, Flatten
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from attention import Attention
from keras_self_attention import SeqSelfAttention
from sklearn.utils import class_weight
import pickle

max_len = 100
batch_size = 128

# read training data
dataset = pd.read_csv("../../data/train_valid/descriptors/training_set_desc.csv", delimiter=",")
X_train = dataset.iloc[:,0:1].values
y_train = dataset.iloc[:,1:2].values
y_train = np.array(y_train).ravel()

print("loaded training data: %s, %s" % (X_train.shape, y_train.shape))

for p in range (X_train.shape[0]):
  s = X_train[p,0]
  s = s.replace("[nH]","A")
  s = s.replace("Cl","L")
  s = s.replace("Br","R")
  s = s.replace("[C@]","C")
  s = s.replace("[C@@]","C")
  s = s.replace("[C@@H]","C")
  s =[s[i:i+1] for i in range(0,len(s),1)]
  s = " ".join(s)
  X_train[p,0] = s
X_train = X_train[:,0]
X_train = X_train.tolist()


tokenizer = Tokenizer(num_words=max_len)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')


# read test data
dataset = pd.read_csv("../../data/train_valid/descriptors/validation_set_desc.csv", delimiter=",")
X_test = dataset.iloc[:,0:1].values
y_test = dataset.iloc[:,1:2].values
y_test = np.array(y_test).ravel()

print("loaded test data: %s, %s" % (X_test.shape, y_test.shape))

for p in range (X_test.shape[0]):
  s = X_test[p,0]
  s = s.replace("[nH]","A")
  s = s.replace("Cl","L")
  s = s.replace("Br","R")
  s = s.replace("[C@]","C")
  s = s.replace("[C@@]","C")
  s = s.replace("[C@@H]","C")
  s =[s[i:i+1] for i in range(0,len(s),1)]
  s = " ".join(s)
  X_test[p,0] = s
X_test = X_test[:,0]  
X_test = X_test.tolist()

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')


# calculate class weights for highly imbalanced datasets
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weight_dict = dict(enumerate(class_weights))

print('class weights: %s' % class_weight_dict)

model = Sequential()
model.add(Embedding(32, 64, input_length=max_len))
model.add(LSTM(64, return_sequences=True))
model.add(SeqSelfAttention(
    attention_width=2,
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation=None,
    kernel_regularizer=keras.regularizers.l2(1e-6),
    use_attention_bias=False,
    name='Attention',
))
model.add(Flatten())
model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(0.01))

model.fit(X_train, y_train, batch_size=batch_size, epochs=10, class_weight=class_weight_dict)
score = model.evaluate(X_test, y_test, batch_size=batch_size)
#print("Results: %.2f (%.2f) MSE" % (score.mean(), score.std()))

'''
# save the model file
model.save("lstm_attn_classifier.h5")

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