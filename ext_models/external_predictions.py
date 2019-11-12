import pandas as pd
import numpy as np

# load prediction file
df = pd.read_csv('predictions_startdrop.csv', delimiter=',')
y_test = np.array(df['activity']).ravel()
y_pred = np.array(df['prediction']).ravel()

# calculate performance metrics
import sys
sys.path.append("../scripts/utils/")
import evaluation_metrics as ev

ba = ev.balanced_accuracy(y_test, y_pred)
sens, spec = ev.sensitivity_specificity(y_test, y_pred)

print('\nModel Performance')
print('BACC:\t%s' % ba)
print('Sens:\t%s' % sens)
print('Spec:\t%s' % spec)