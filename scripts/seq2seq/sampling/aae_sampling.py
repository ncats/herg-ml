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
from rdkit import Chem

seed = 7
np.random.seed(seed)

dataset = pd.read_csv("training_set_blockers.csv", delimiter=",")
X_train = dataset.iloc[:,0:1].values # smiles

X_train.shape

X_train = X_train[:,0]
X_train1 = np.copy(X_train)

charset = 40
embed = 101
max_len = embed-1

pkl_file = open('../aae/char_to_int.pkl', 'rb')
char_to_int = pickle.load(pkl_file)
pkl_file.close()

print(len(char_to_int))

pkl_file = open('../aae/int_to_char.pkl', 'rb')
int_to_char = pickle.load(pkl_file)
pkl_file.close()

print(len(int_to_char))


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


# load models

smiles_to_latent_model = load_model("../aae/smi2lat.h5")
sample_model = load_model("../aae/samplemodel.h5")


def latent_to_smiles(latent):
    #decode states and set Reset the LSTM cells with them
    #Prepare the input char
    #print(latent)
    startidx = char_to_int["!"]
    samplevec = np.zeros((1,max_len,charset))
    samplevec[0,0,startidx] = 1
    smiles = ""
    #Loop and predict next char
    for i in range(1,max_len):
        o  = sample_model.predict([samplevec,latent]).argmax(axis=2)
        #print(o)
        sampleidx = int(o[:,i-1])
        #print(sampleidx)
        samplechar = int_to_char[sampleidx]
        #print(samplechar)
        if samplechar != "E":
           smiles = smiles + int_to_char[sampleidx]
           #samplevec = np.zeros((1,1,58))
           samplevec[0,i,sampleidx] = 1
        else:
            break
    return smiles

X_latent = smiles_to_latent_model.predict(X_train)
X_latent.shape

scale = 0.20
mols = []

for y in range (X_latent.shape[0]):
    latent = X_latent[y:y+1]
    print(latent.shape)
    for i in range(30):
        latent_r = latent + scale*(np.random.standard_normal(latent.shape)) #TODO, try with
        smiles = latent_to_smiles(latent_r)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mols.append(Chem.MolToSmiles(mol))
mols = np.asarray(mols)
#print(len(mols))

with open('blockers_sample_space.txt', 'w') as f:
    for mol in mols:
        f.write("%s\n" % mol)
