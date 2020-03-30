import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.layers import Input, RepeatVector, Bidirectional, Dense, Dropout, Activation, BatchNormalization, Embedding, Conv1D, MaxPooling1D, GRU, Concatenate
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, Multiply, Flatten
import numpy as np
import csv
import pandas as pd
import hashlib
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from matplotlib import pyplot as plt

seed = 7
np.random.seed(seed)

dataset = pd.read_csv("../../data/train_valid/validation_set.csv", delimiter=",")
X_train = dataset.iloc[:,0:1].values # smiles

X_train = X_train[:,0]
print(X_train.shape)
#print(X_train)


X_train1 = np.copy(X_train)

charset = 40
embed = 101
max_len = embed-1

pkl_file = open('ae/char_to_int.pkl', 'rb')
char_to_int = pickle.load(pkl_file)
pkl_file.close()

print(len(char_to_int))

pkl_file = open('ae/int_to_char.pkl', 'rb')
int_to_char = pickle.load(pkl_file)
pkl_file.close()

print(char_to_int)
print(int_to_char)

model = load_model("ae/smiles2smiles.h5")


def vectorize(smiles):
        one_hot =  np.zeros((smiles.shape[0], embed , charset),dtype=np.int8)
        for i,smile in enumerate(smiles):
            one_hot[i,0,char_to_int["!"]] = 1
            for j,c in enumerate(smile):
                if (j < (max_len-1))and(c in char_to_int):
                    one_hot[i,j+1,char_to_int[c]] = 1
            one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
        return one_hot[:,0:-1,:], one_hot[:,1:,:]
X_train, XX_train = vectorize(X_train)


print(X_train.shape)

v = model.predict([X_train[1:2], X_train[1:2]])

print(X_train[1:2])

idxs = np.argmax(v, axis=2)
pred=  "".join([int_to_char[h] for h in idxs[0]])[:-1]
idxs2 = np.argmax(X_train[1:2], axis=2)
true =  "".join([int_to_char[k] for k in idxs2[0]])[1:]
print(pred)
print(true)


smiles_to_latent_model = load_model("ae/smi2lat.h5")
x_latent = smiles_to_latent_model.predict(X_train)
#print(x_latent.shape)


#x_latent1 = np.resize(x_latent, (x_latent.shape[0],x_latent.shape[1]*x_latent.shape[2]))
x_latent1 = np.resize(x_latent, (x_latent.shape[0],x_latent.shape[1]))
print(x_latent1.shape)

'''
with open('validation_set_latent1.csv', 'w', newline = "") as csvfile:
 fieldnames = ['Pred']
 writer = csv.writer(csvfile,delimiter=',',)
 for p in x_latent1: writer.writerow(p)
'''
