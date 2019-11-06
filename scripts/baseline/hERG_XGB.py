# imports

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv('/Users/siramshettyv2/work/herg/scripts/dataset/latent/chembl_full_latent.csv', delimiter=',')
X_train = dataset.iloc[:,2:514]
y_train = dataset.iloc[:,1:2]

print("training data loaded")
print(X_train.shape)
print(y_train.shape)

dataset = pd.read_csv('/Users/siramshettyv2/work/herg/scripts/dataset/latent/ncats_test_latent.csv', delimiter=',')
X_test = dataset.iloc[:,2:514]
y_test = dataset.iloc[:,1:2]

print("test data loaded")
print(X_test.shape)
print(y_test.shape)

# fit model on training data
#model = RandomForestClassifier(n_estimators = 300, random_state = 42)
model = XGBClassifier(objective="binary:logistic", random_state=42)

model.fit(X_train, y_train)
# make predictions for test data
predictions = model.predict(X_test)
labels = np.round(predictions,0)


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