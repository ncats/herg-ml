from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score

def accuracy(y_true, y_pred):
	acc = accuracy_score(y_true, y_pred)
	acc = float("{0:.2f}".format(acc))
	return acc


def balanced_accuracy(y_true, y_pred):
	bacc = balanced_accuracy_score(y_true, y_pred)
	bacc = float("{0:.2f}".format(bacc))
	return bacc

def sensitivity_specificity(y_true, y_pred):
	confusion = confusion_matrix(y_true, y_pred)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]

	sensitivity = TP / float(FN + TP)
	sensitivity = float("{0:.2f}".format(sensitivity))
	specificity = TN / float(TN + FP)
	specificity = float("{0:.2f}".format(specificity))
	return sensitivity, specificity

def auc_roc(y_true, y_pred_prob):
	auc = roc_auc_score(y_true, y_pred_prob)
	auc = float("{0:.2f}".format(auc))
	return auc

def kappa_score(y_true, y_pred):
	kappa = cohen_kappa_score(y_true, y_pred)
	kappa = float("{0:.2f}".format(kappa))
	return kappa