import pandas as pd
import numpy as np

# load prediction file
df = pd.read_csv('validation_predherg.csv', delimiter=',')
y_test = np.array(df['activity']).ravel()
y_pred = np.array(df['prediction']).ravel()

# calculate performance metrics
import sys
sys.path.append("../scripts/utils/")
import evaluation_metrics as ev

ba = ev.balanced_accuracy(y_test, y_pred)
sens, spec = ev.sensitivity_specificity(y_test, y_pred)
kappa = ev.kappa_score(y_test, y_pred)

print('Model Performance')
print('BACC:\t%s' % ba)
print('Sens:\t%s' % sens)
print('Spec:\t%s' % spec)
#print('Kappa:\t%s' % kappa)
