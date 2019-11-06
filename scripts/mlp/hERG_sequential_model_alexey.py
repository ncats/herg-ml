import numpy as np
seed = 7
np.random.seed(seed)

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
import csv
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score

from sklearn.utils import class_weight
from collections import Counter

def BA(y_true, y_pred):
   y_pred_pos = np.round(np.clip(y_pred, 0, 1))
   y_pred_neg = 1 - y_pred_pos
   y_pos = np.round(np.clip(y_true, 0, 1))
   y_neg = 1 - y_pos
   tp = np.sum(y_pos * y_pred_pos)
   tn = np.sum(y_neg * y_pred_neg)
   fp = np.sum(y_neg * y_pred_pos)
   fn = np.sum(y_pos * y_pred_neg)
   Sensitivity = tp/(tp + fn)
   Specificity = tn/(tn + fp)
   return (Sensitivity + Specificity)/2




dataset = pd.read_csv('/Users/siramshettyv2/work/herg/data/chembl_lib/train_test/latent/chembl_full.csv', delimiter=',')
X_train = dataset.iloc[:,2:514].values
y_train = dataset.iloc[:,1:2].values


print("first data loaded")
print(X_train.shape)
print(y_train.shape)

dataset = pd.read_csv('//Users/siramshettyv2/work/herg/data/chembl_lib/train_test/latent/ncats_test.csv', delimiter=',')
X_test = dataset.iloc[:,2:514].values
y_test = dataset.iloc[:,1:2].values


print("second data loaded")
print(X_test.shape)
print(y_test.shape)


#class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.ravel()), y_train.ravel())

#weights = {}
#weights[0] = class_weights[0]
#weights[1] = class_weights[1]

#print(weights)

weights = {0: 0.65, 1: 0.35}


model = Sequential()

model = Sequential([
        Dense(2000, input_dim=512, kernel_initializer='uniform', activation='selu'),
        #Dropout(0.5),
        #BatchNormalization(),
        Dense(2000, kernel_initializer='uniform', activation='selu'),
        #Dropout(0.5),
        #BatchNormalization(),
        Dense(700, kernel_initializer='uniform', activation='selu'),
        #Dropout(0.5),
        #BatchNormalization(),
        Dense(500, kernel_initializer='uniform', activation='selu'),
        #BatchNormalization(),
        Dense(1, activation='sigmoid')])


#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss="binary_crossentropy",
              optimizer=Adam(0.0001))

model.fit(X_train, y_train,
          epochs=20,
          batch_size=128, class_weight=weights)


score = model.evaluate(X_test, y_test, batch_size=64)

print("Results: %.2f (%.2f) MSE" % (score.mean(), score.std()))

predictions = model.predict(X_test)

#ba = BA(y_test,predictions)
#print(str(ba))


labels = np.round(predictions,0)


# calculate accuracy and balanced accuracy
acc = accuracy_score(y_test, labels)
acc = float("{0:.2f}".format(acc))
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