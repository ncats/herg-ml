import numpy as np
import keras as ke
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import Input, RepeatVector, Bidirectional, Dense, Dropout, Activation, BatchNormalization, Embedding, Conv1D, MaxPooling1D, GRU, Concatenate
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, Multiply, Flatten, dot, concatenate, Reshape
import pandas as pd
from keras.datasets import mnist
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.utils import shuffle

seed = 7
np.random.seed(seed)

dataset = pd.read_csv("../../data/chembl25/chembl25_train.smiles", delimiter=",", header=None)
X_train = dataset.iloc[:,0:1].values


X_train = X_train[:,0]
X_train1 = X_train

charset = set("".join(list(X_train))+"!E")
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in X_train])+1
print (charset)
print(len(charset), embed)
print(int_to_char)
max_len = embed-1

output = open('aae/char_to_int.pkl', 'wb')
pickle.dump(char_to_int, output)
output.close()

output = open('aae/int_to_char.pkl', 'wb')
pickle.dump(int_to_char, output)
output.close()


def vectorize(smiles):
        one_hot =  np.zeros((smiles.shape[0], embed , len(charset)),dtype=np.int8)
        for i,smile in enumerate(smiles):
            one_hot[i,0,char_to_int["!"]] = 1
            for j,c in enumerate(smile):
                one_hot[i,j+1,char_to_int[c]] = 1
            one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
        return one_hot[:,0:-1,:], one_hot[:,1:,:]
X_train, XX_train = vectorize(X_train)

input_shape = X_train.shape[1:]
output_dim = XX_train.shape[-1]
latent_dim = 512
lstm_dim = 64
encoder_dim = (latent_dim)

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

model_ae = Model([encoder_inputs, decoder_inputs], decoder_outputs)


model_enc = Model(encoder_inputs, encoder_outputs)

desc = Input(shape=(encoder_dim,))
#x = Flatten()(desc)
x = Dense(500, activation="relu")(desc)
x = Dense(200, activation="relu")(x)
out = Dense(1, activation="sigmoid")(x)
model_disc = Model(desc, out)

input_enc = Input(shape=input_shape)
output_enc = model_enc(input_enc)
output_disc = model_disc(output_enc)
model_enc_disc = Model(input=input_enc, output=output_disc)


model_enc.summary()
#model_dec.summary()
model_disc.summary()
model_ae.summary()
model_enc_disc.summary()

model_disc.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")
model_disc.trainable = False
model_enc_disc.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")
model_ae.compile(optimizer=ke.optimizers.Adam(lr=0.001), loss="categorical_crossentropy")

batchsize=256

for epochnumber in range(5):
    X_train, XX_train = shuffle(X_train, XX_train)

    for i in range(int(len(X_train) / batchsize)):

        batch = X_train[i*batchsize:i*batchsize+batchsize]
        batch2 = XX_train[i*batchsize:i*batchsize+batchsize]
        ae_loss = model_ae.train_on_batch([batch, batch],[batch2])

        batchpred = model_enc.predict(batch)
        fakepred = np.random.standard_normal(size=(batchsize,latent_dim))
        discbatch_x = np.concatenate([batchpred, fakepred])
        discbatch_y = np.concatenate([np.zeros(batchsize), np.ones(batchsize)])
        model_disc.train_on_batch(discbatch_x, discbatch_y)

        model_enc_disc.train_on_batch(batch, np.ones(batchsize))
        if i % 10 == 0: print(i, ae_loss, epochnumber)


smiles_pred = []
smiles_act = []

model_ae.save("aae/smiles2smiles.h5")


b = 0
for i in range(1000):
    v = model_ae.predict([X_train[i:i+1], X_train[i:i+1]])
    idxs = np.argmax(v, axis=2)
    pred=  "".join([int_to_char[h] for h in idxs[0]])[:-1]
    pred= pred.replace("E","")
    smiles_pred.append(pred)
    idxs2 = np.argmax(X_train[i:i+1], axis=2)
    true =  "".join([int_to_char[k] for k in idxs2[0]])[1:]
    true= true.replace("E","")
    smiles_act.append(true)
    if true != pred: b = b +1

print ("Number of errors: %s" % b)

model_enc.save("aae/smi2lat.h5")


inf_decoder_inputs = Input(shape=(None, output_dim))
inf_decoder_inputs2 = Input(shape=(None, latent_dim))
inf_decoder_lstm = LSTM(lstm_dim,return_sequences=True, unroll=unroll)
inf_decoder_outputs  = inf_decoder_lstm(inf_decoder_inputs)

encoder_out1 = Dense(6400, activation="relu") (inf_decoder_inputs2)
encoder_out = Reshape((100, 64), input_shape=(6400,))(encoder_out1)


inf_attention = dot([inf_decoder_outputs, encoder_out], axes=[2, 2])
inf_attention = Activation('softmax')(inf_attention)
inf_context = dot([inf_attention, inf_decoder_inputs2], axes=[2,1])
inf_decoder_combined_context = concatenate([inf_context, inf_decoder_outputs])
inf_output = (Dense(lstm_dim, activation="tanh"))(inf_decoder_combined_context)
inf_decoder_dense = Dense(output_dim, activation='softmax')
inf_decoder_outputs = inf_decoder_dense(inf_output)

sample_model = Model([inf_decoder_inputs,inf_decoder_inputs2] , inf_decoder_outputs)
for i in range(1,7):
    sample_model.layers[i+2].set_weights(model_ae.layers[i+3].get_weights())
sample_model.layers[1].set_weights(model_ae.layers[2].get_weights())

sample_model.save("aae/samplemodel.h5")
