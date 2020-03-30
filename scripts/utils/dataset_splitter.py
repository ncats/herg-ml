import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(path_to_dataset, split_size, n_folds):
	df = pd.read_csv(path_to_dataset, delimiter=',')
	#print("dataset loaded")
	train_dfs = {}
	test_dfs = {}
	for k in range(n_folds):
		#print("performing splitting, fold: %s" % k)
		train, test = train_test_split(df, test_size=split_size) # 'shuffle = True' by default
		train_dfs[k] = train
		test_dfs[k] = test
	#print("finished splitting datasets\n")
	return train_dfs, test_dfs

#dataset = '../../data/train_valid/descriptors/training_set_desc.csv'
#train_dfs, test_dfs = split_dataset(dataset, 0.2, 5) # 0.2 

#print('Training Sets')
#for k, d in train_dfs.items():
#	print('Training Set: %s, %s'% (k, d.shape))

#print('Test Sets')
#for k, d in test_dfs.items():
#	print('Test Set: %s, %s'% (k, d.shape))