import keras
from keras import backend as K
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import Input, RepeatVector, Bidirectional, TimeDistributed, Dense, Dropout, Activation, BatchNormalization, Embedding, Conv1D, MaxPooling1D, GRU, Concatenate
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, Multiply, Flatten, dot, concatenate, Reshape
import numpy as np
import csv
import pandas as pd
import hashlib
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from Attention import Attention

seed = 7
np.random.seed(seed)

# reading training dataset
dataset = pd.read_csv("/work/vishal/herg/data/guacamol/chembl_25/chembl25_train.smiles", delimiter=",", header=None)
X_train = dataset.iloc[:,0:1].values

# reading test dataset
dataset = pd.read_csv("/work/vishal/herg/data/guacamol/chembl_25/chembl25_test.smiles", delimiter=",", header=None)
X_test = dataset.iloc[:,0:1].values


X_train = X_train[:,0]
X_test = X_test[:,0]

X_train1 = np.copy(X_train)
print(X_train.shape)

charset = set("".join(list(X_train))+"".join(list(X_test))+"!E")
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in X_train])+1
print (charset)
print('total characters: '+str(len(charset)))
print(int_to_char)
max_len = embed-1
print('max_len: '+str(max_len))

output = open('/work/vishal/herg/models/ae_guacamol/chembl_25/char_to_int.pkl', 'wb')
pickle.dump(char_to_int, output)
output.close()
print('saved char to int model...')

output = open('/work/vishal/herg/models/ae_guacamol/chembl_25/int_to_char.pkl', 'wb')
pickle.dump(int_to_char, output)
output.close()
print('saved int to char model...')


def vectorize(smiles):
        one_hot =  np.zeros((smiles.shape[0], embed , len(charset)),dtype=np.int8)
        for i,smile in enumerate(smiles):
            one_hot[i,0,char_to_int["!"]] = 1
            for j,c in enumerate(smile):
                one_hot[i,j+1,char_to_int[c]] = 1
            one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
        return one_hot[:,0:-1,:], one_hot[:,1:,:]
X_train, XX_train = vectorize(X_train)
X_test, XX_test = vectorize(X_test)

input_shape = X_train.shape[1:]
output_dim = XX_train.shape[-1]

lstm_dim = 64
latent_dim = 512

print(input_shape)
print(output_dim)


unroll = False

encoder_inputs = Input(shape=input_shape)
encoder = LSTM(lstm_dim, return_sequences=True, unroll=unroll)
encoder_1 = encoder(encoder_inputs)
encoder_2 = Flatten()(encoder_1)
encoder_outputs = Dense(latent_dim, activation="relu")(encoder_2)

decoder_inputs = Input(shape=input_shape)
decoder_lstm = LSTM(lstm_dim,return_sequences=True, unroll=unroll)
decoder_outputs = decoder_lstm(decoder_inputs)

encoder_out1 = Dense(6400, activation="relu") (encoder_outputs)
encoder_out = Reshape((100, 64), input_shape=(6400,))(encoder_out1)

attention = dot([decoder_outputs, encoder_out], axes=[2, 2])
attention = Activation('softmax')(attention)

context = dot([attention, encoder_out], axes=[2,1])
decoder_combined_context = concatenate([context, decoder_outputs])

output = (Dense(lstm_dim, activation="tanh"))(decoder_combined_context)

decoder_dense = Dense(output_dim, activation='softmax')
decoder_outputs = decoder_dense(output)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


print (model.summary())


opt=Adam(lr=0.001) #Default 0.001
model.compile(optimizer=opt, loss='categorical_crossentropy')


model.fit([X_train, X_train],[XX_train],
                    epochs=5,
                    batch_size=256,
                    #shuffle=True,
                    validation_data=[[X_test,X_test],[XX_test] ])

smiles_pred = []
smiles_act = []

model.save("/work/vishal/herg/models/ae_guacamol/chembl_25/smiles2smiles.h5")

# check output (i.e. compare input SMILES with output SMILES)

b = 0
for i in range(100):
    v = model.predict([X_test[i:i+1], X_test[i:i+1]])
    idxs = np.argmax(v, axis=2)
    pred=  "".join([int_to_char[h] for h in idxs[0]])[:-1]
    pred= pred.replace("E","")
    smiles_pred.append(pred)
    idxs2 = np.argmax(X_test[i:i+1], axis=2)
    true =  "".join([int_to_char[k] for k in idxs2[0]])[1:]
    true= true.replace("E","")
    smiles_act.append(true)
    if true != pred: b = b +1
    
print ("Number of errors: %s" % b)

with open('/work/vishal/herg/results/ae_guacamol/results_pred_chembl_25.csv', 'w', newline = '') as afile:
    for p in range (len(smiles_pred)):
       aa= str(smiles_act[p])+","+str(smiles_pred[p])+"\n"
       afile.write(aa)
afile.close()

smiles_to_latent_model = Model(encoder_inputs, encoder_outputs)
smiles_to_latent_model.save("/work/vishal/herg/models/ae_guacamol/chembl_25/smi2lat.h5")
