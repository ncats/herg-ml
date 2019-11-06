import keras
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Embedding, Conv1D, MaxPooling1D, GRU, Flatten
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM
import numpy
import csv
import pandas as pd
import hashlib
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from Attention import Attention
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras_multi_head import MultiHeadAttention
from keras_multi_head import MultiHead

seed = 7
numpy.random.seed(seed)

maxlen = 100
batch_size = 128

dataset = pd.read_csv("/Users/siramshettyv2/work/herg/scripts/dataset/chembl_full_deepsmi.csv", delimiter=",")
X_train = dataset.iloc[:,2:3].values
y_train = dataset.iloc[:,3:4].values


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


#y_train = y_train[:,0]
X_train = X_train.tolist()
#print(X_train)
#print(X_train.shape)


tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=100)
print(X_train)


dataset = pd.read_csv("/Users/siramshettyv2/work/herg/scripts/dataset/ncats_test_deepsmi.csv", delimiter=",")
X_test = dataset.iloc[:,2:3].values
y_test = dataset.iloc[:,3:4].values

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
#y_test = y_test[:,0]
X_test = X_test.tolist()
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=100)
#print(X_test)
#print(X_test.shape)

model = Sequential()
model.add(Embedding(100, 32, input_length=100))
#model.add(LSTM(128, return_sequences=True))
#model.add(MultiHeadAttention(head_num=4, name='Multi-Head1',))
model.add(MultiHead(
    layer=keras.layers.Bidirectional(keras.layers.LSTM(units=128), name='LSTM'),
    layer_num=5,
    reg_index=[1, 4],
    reg_slice=(slice(None, None), slice(32, 48)),
    reg_factor=0.1,
    name='Multi-Head-Attention',
))
model.add(Flatten())
model.add(Dense(100, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))
#model.compile(loss='mse', optimizer=RMSprop(lr=0.0001))



model.fit(X_train, y_train, batch_size=32, epochs=20)

score = model.evaluate(X_test, y_test, batch_size=32)

# get prediction probabilities and labels
predictions = model.predict(X_test)
labels = numpy.round(predictions,0)

# calculate accuracy and balanced accuracy
#acc = accuracy_score(y_test, labels)
#acc = float("{0:.2f}".format(acc))
bacc = balanced_accuracy_score(y_test, labels)
bacc = float("{0:.2f}".format(bacc))

# claculate sensitivity and specificity
confusion = confusion_matrix(y_test, labels)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

sensitivity = TP / float(FN + TP)
sensitivity = float("{0:.2f}".format(sensitivity))

specificity = TN / float(TN + FP)
specificity = float("{0:.2f}".format(specificity))

# calculate AUC-ROC
auc = roc_auc_score(y_test, predictions)
auc = float("{0:.2f}".format(auc))

# calculate kappa
##kappa = cohen_kappa_score(y_test, labels)
#kappa = float("{0:.2f}".format(kappa))

print(str(auc),str(bacc),str(sensitivity),str(specificity))